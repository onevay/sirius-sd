"""Подготовка обучения классификатора циклов (руководство §4): вектор признаков цикла, группы, метки.

Единица классификации — ЦИКЛ жеста, найденный автоматом с мягкими порогами (высокий recall). Таблица строится по запускам пайплайна
(`outputs/runs/...`), метки хранятся в `labels/cycle_labels.csv`. Правила:
  * пространственные признаки — в ширинах плеч, временные — в секундах; пропуски — NaN (бустинг их понимает);
  * группа = видео (актёр+камера+дубль): один и тот же человек/сцена НЕ должны попадать и в train, и в val (иначе «валидация 0.98»);
  * слабые метки по папке (`курение` -> smoke?, `лжекурение` -> neg) годятся только для запуска конвейера; оценка — по ручным меткам.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import LABELS, OUTPUTS, video_id, weak_label

# значения label в файле меток -> бинарная цель (1 = затяжка / «курение»)
LABEL_CLASSES = {
    "smoke": 1,          # затяжка сигаретой/вейпом
    "drink": 0, "phone": 0, "eat": 0, "touch_face": 0, "other_neg": 0,   # сложные негативы
    "ignore": None, "unsure": None,                                       # не используем в обучении
}
LABELS_CSV = LABELS / "cycle_labels.csv"
LABEL_COLS = ["video", "tid", "peak_t", "cx", "cy", "label", "source", "note", "labeler", "ts"]

KINEMATIC = ["hold", "d_min", "d_med_hold", "d_std_hold", "t_approach", "t_retract", "v_approach_max", "v_retract_max", "amplitude",
             "elbow_hold", "elbow_range", "hand", "head_tilt_hold", "head_tilt_delta", "wrist_h_hold", "nose_disp", "wrist_disp"]
RHYTHM = ["n_cycles_pm20", "interval_prev", "interval_next", "interval_std", "same_hand_frac", "time_at_mouth_frac60"]
QUALITY = ["H", "s_px", "wrist_conf", "nose_conf", "missing_frac", "overlap_iou", "edge_dist", "track_len"]
CONTEXT = ["brightness"]
OBJECT = ["obj_max_conf", "obj_hit_frames", "obj_hit"]
APPEARANCE_X = ["x_smoking", "x_vaping", "x_drinking", "x_phone", "x_eating", "x_touching", "x_talking"]
APPEARANCE_V = ["v_smoking", "v_drinking_any", "v_eating_any", "v_top1_p"]
VLM = ["vlm_yesno", "vlm_letter"]   # оценка «затяжка» от VLM (вероятность по логитам одного токена), режимы — configs/prompts/vlm_yesno.txt / vlm_letter.txt
FEATURE_GROUPS = {"A_kinematics": KINEMATIC, "B_rhythm": RHYTHM, "C_quality": QUALITY, "F_context": CONTEXT, "D_object": OBJECT,
                  "E_xclip": APPEARANCE_X, "E_videomae": APPEARANCE_V, "G_vlm": VLM}


def _win(s: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    return s[(s.t >= t0) & (s.t <= t1)]


def _nanmed(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def cycle_vector(s: pd.DataFrame, c: pd.Series, tid_cycles: pd.DataFrame, summary: pd.Series, iou_other: float, frame_wh: tuple[int, int],
                 brightness: float = np.nan) -> dict:
    """Один цикл -> словарь признаков групп A, B, C, F. `s` — ряд признаков ЭТОГО трека."""
    hold = _win(s, c.mouth_in, c.mouth_out)
    appr = _win(s, c.start, c.mouth_in)
    retr = _win(s, c.mouth_out, c.end)
    pre = _win(s, c.start - 1.0, c.start)
    whole = _win(s, c.start, c.end)
    h = int(c.hand)
    elbow = hold["elbowL" if h == 0 else "elbowR"] if h in (0, 1) else pd.Series(dtype=float)
    v = s.v
    nose_xy = whole[["nose_x", "nose_y"]].values
    wx, wy = ("wristL_x", "wristL_y") if h == 0 else ("wristR_x", "wristR_y")
    wr_xy = whole[[wx, wy]].values if h in (0, 1) else np.full((len(whole), 2), np.nan)
    sc = float(np.nanmedian(whole.s)) if len(whole) else np.nan

    def disp(a: np.ndarray) -> float:
        a = a[np.isfinite(a).all(1)] if len(a) else a
        return float(np.hypot(*(a[-1] - a[0])) / sc) if len(a) >= 2 and sc else np.nan

    out = dict(
        hold=float(c.hold), d_min=float(c.d_min), d_med_hold=_nanmed(hold.d), d_std_hold=float(np.nanstd(hold.d)) if len(hold) else np.nan,
        t_approach=float(c.mouth_in - c.start), t_retract=float(c.end - c.mouth_out),
        v_approach_max=float(np.nanmax(np.abs(appr.v))) if len(appr) and appr.v.notna().any() else np.nan,
        v_retract_max=float(np.nanmax(np.abs(retr.v))) if len(retr) and retr.v.notna().any() else np.nan,
        amplitude=float(np.nanmax(_win(s, c.start - 0.5, c.mouth_in).d) - c.d_min) if len(_win(s, c.start - 0.5, c.mouth_in)) else np.nan,
        elbow_hold=_nanmed(elbow), elbow_range=float(np.nanmax(elbow) - np.nanmin(elbow)) if len(elbow) and elbow.notna().any() else np.nan,
        hand=h, head_tilt_hold=_nanmed(hold.head_tilt), head_tilt_delta=_nanmed(hold.head_tilt) - _nanmed(pre.head_tilt),
        wrist_h_hold=_nanmed(hold.wrist_h), nose_disp=disp(nose_xy), wrist_disp=disp(wr_xy),
    )
    # ритм: соседние циклы этого трека в ±20 с
    others = tid_cycles[tid_cycles.peak_t != c.peak_t]
    near = others[(others.peak_t - c.peak_t).abs() <= 20.0]
    prev_ = others[others.peak_t < c.peak_t]
    next_ = others[others.peak_t > c.peak_t]
    ints = np.diff(np.sort(np.r_[near.peak_t.values, c.peak_t])) if len(near) else np.array([])
    out.update(
        n_cycles_pm20=int(len(near)),
        interval_prev=float(c.peak_t - prev_.peak_t.max()) if len(prev_) else np.nan,
        interval_next=float(next_.peak_t.min() - c.peak_t) if len(next_) else np.nan,
        interval_std=float(np.std(ints)) if len(ints) >= 2 else np.nan,
        same_hand_frac=float((near.hand == h).mean()) if len(near) else np.nan,
        time_at_mouth_frac60=float(tid_cycles[(tid_cycles.peak_t - c.peak_t).abs() <= 30.0].hold.sum() / 60.0),
    )
    # качество наблюдения
    W, Hh = frame_wh
    rows = _win(s, c.start, c.end)
    d_ok = rows.d.notna().mean() if len(rows) else np.nan
    cx = float(s.loc[(s.t - c.peak_t).abs().idxmin(), ["mouth_x"]].iloc[0]) if len(s) else np.nan
    cy = float(s.loc[(s.t - c.peak_t).abs().idxmin(), ["mouth_y"]].iloc[0]) if len(s) else np.nan
    h_med = float(_nanmed(rows.H))
    out.update(H=h_med, s_px=sc, wrist_conf=float(rows.wrist_conf.mean()) if len(rows) else np.nan,
               nose_conf=float(rows.nose_conf.mean()) if len(rows) else np.nan, missing_frac=float(1 - d_ok) if d_ok == d_ok else np.nan,
               overlap_iou=float(iou_other), edge_dist=float(min(cx, W - cx, cy, Hh - cy) / max(h_med, 1.0)) if np.isfinite(cx) else np.nan,
               track_len=float(summary.dur), brightness=float(brightness))
    out.update(cx=cx, cy=cy)
    return out


def build_cycle_table(video: Path, tr, series: pd.DataFrame, cycles: pd.DataFrame, brightness: float = np.nan) -> pd.DataFrame:
    """Таблица циклов одного запуска с векторами признаков и служебными колонками (video, group, weak_label)."""
    from .evaluate import iou as box_iou

    if cycles.empty:
        return pd.DataFrame()
    info = tr.meta["video_info"]
    summ = tr.summary().set_index("tid")
    vid = video_id(video)
    rows = []
    for tid, g in cycles.groupby("tid"):
        s = series[series.tid == tid].reset_index(drop=True)
        for c in g.itertuples():
            cs = pd.Series(c._asdict())
            # максимальное перекрытие с другими людьми на момент пика
            b = tr.box_at(int(tid), c.peak_t)
            ov = 0.0
            for other in tr.tids:
                if other == tid:
                    continue
                ob = tr.box_at(other, c.peak_t)
                rr, _ = tr.of(other)
                if ob is not None and len(rr) and rr.t.min() - 0.5 <= c.peak_t <= rr.t.max() + 0.5 and b is not None:
                    ov = max(ov, box_iou(b, ob))
            vec = cycle_vector(s, cs, g, summ.loc[tid], ov, (info["width"], info["height"]), brightness)
            rows.append(dict(video=vid, group=vid, weak_label=weak_label(video) or "-", tid=int(tid), start=c.start, mouth_in=c.mouth_in,
                             mouth_out=c.mouth_out, end=c.end, peak_t=c.peak_t, **vec))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ метки
def load_labels() -> pd.DataFrame:
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV)
    return pd.DataFrame(columns=LABEL_COLS)


def save_label(video: str, tid: int, peak_t: float, cx: float, cy: float, label: str, note: str = "", labeler: str = "me", source: str = "manual") -> None:
    """Добавляет/обновляет метку цикла (идентификатор метки = видео + момент пика + положение, а не tid — он меняется при смене модели)."""
    assert label in LABEL_CLASSES, f"label ∈ {list(LABEL_CLASSES)}"
    LABELS.mkdir(parents=True, exist_ok=True)
    df = load_labels()
    m = (df.video == video) & ((df.peak_t - peak_t).abs() < 0.5) & (np.hypot(df.cx - cx, df.cy - cy) < 100)
    df = df[~m]
    new = pd.DataFrame([dict(video=video, tid=tid, peak_t=round(peak_t, 3), cx=round(cx, 1), cy=round(cy, 1), label=label, source=source,
                             note=note, labeler=labeler, ts=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))])
    out = new if df.empty else pd.concat([df, new], ignore_index=True)
    out[LABEL_COLS].to_csv(LABELS_CSV, index=False)


def visual_labels_to_project(csv: Path, table: pd.DataFrame, tol_start: float = 0.06) -> pd.DataFrame:
    """Метки в формате визуального просмотра (`video, tid, start, label`; start округлён до 0.1 с) -> формат проекта (`video, peak_t, cx, cy, label`).

    Стыкуются по (video, tid, ближайший start ≤ tol_start) с таблицей циклов, откуда берутся peak_t и положение. Метки без пары пропускаются.
    Файл `docs/review/assistant_visual_labels.csv` — суждения ассистента, а не эталон: `source = "visual"`.
    """
    lab = pd.read_csv(csv)
    rows = []
    for r in lab.itertuples():
        c = table[(table.video == r.video) & (table.tid == r.tid) & ((table.start - r.start).abs() <= tol_start)]
        if len(c):
            c = c.iloc[int((c.start - r.start).abs().values.argmin())]
            rows.append(dict(video=r.video, tid=int(r.tid), peak_t=float(c.peak_t), cx=float(c.cx), cy=float(c.cy), label=r.label, source="visual",
                             note=getattr(r, "note", ""), labeler=getattr(r, "reviewer", "visual"), ts=""))
    return pd.DataFrame(rows, columns=LABEL_COLS)


def attach_labels(table: pd.DataFrame, tol_t: float = 1.0, tol_px: float = 150.0, use_weak: bool = True, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Добавляет колонки label / y / label_source. Ручные метки имеют приоритет над слабыми (по папке).

    `labels` — таблица меток в формате проекта вместо `labels/cycle_labels.csv` (например, из `visual_labels_to_project`); метки из неё считаются «ручными».
    """
    t = table.copy()
    t["label"], t["y"], t["label_source"] = None, np.nan, ""
    lab = load_labels() if labels is None else labels
    for i, r in t.iterrows():
        m = lab[(lab.video == r.video) & ((lab.peak_t - r.peak_t).abs() <= tol_t) & (np.hypot(lab.cx - r.cx, lab.cy - r.cy) <= tol_px)]
        if len(m):
            row = m.iloc[(m.peak_t - r.peak_t).abs().argmin()]
            t.at[i, "label"], t.at[i, "y"], t.at[i, "label_source"] = row.label, LABEL_CLASSES.get(row.label, np.nan), "manual"
        elif use_weak and r.weak_label == "fake":
            t.at[i, "label"], t.at[i, "y"], t.at[i, "label_source"] = "other_neg", 0, "weak"
        elif use_weak and r.weak_label == "smoking":
            t.at[i, "label"], t.at[i, "y"], t.at[i, "label_source"] = "smoke", 1, "weak"
    return t


def attach_model_features(table: pd.DataFrame, evidence: pd.DataFrame | None = None, tube: pd.DataFrame | None = None,
                          vlm: pd.DataFrame | None = None) -> pd.DataFrame:
    """Присоединяет к таблице циклов признаки групп D (предмет), E (X-CLIP / VideoMAE) и G (VLM) по ключу (video, tid, start).

    X-CLIP и VideoMAE лежат в разных parquet: функцию вызывают последовательно, по одному `tube`.
    `vlm` — результат `vlm_eval.run_vlm`: колонки `score_<режим>` становятся `vlm_<режим>`.
    """
    t = table.copy()
    t["start_key"] = t.start.round(3)
    if vlm is not None and len(vlm):
        g = vlm.rename(columns={"start": "start_key"}).copy()
        g["start_key"] = g.start_key.round(3)
        sc = {c: f"vlm_{c[len('score_'):]}" for c in g.columns if c.startswith("score_")}
        t = t.merge(g[["video", "tid", "start_key", *sc]].rename(columns=sc), on=["video", "tid", "start_key"], how="left")
    if evidence is not None and len(evidence):
        e = evidence.rename(columns={"start": "start_key"})
        hit_cols = [c for c in e.columns if c.endswith("_max_conf")]
        e["obj_max_conf"] = e[hit_cols].max(axis=1) if hit_cols else 0.0
        e["obj_hit_frames"] = e[[c for c in e.columns if c.endswith("_hit_frames")]].max(axis=1)
        e["obj_hit"] = e[[c for c in e.columns if c.endswith("_hit") and not c.endswith("_max_hit")]].any(axis=1).astype(float)
        t = t.merge(e[["video", "tid", "start_key", "obj_max_conf", "obj_hit_frames", "obj_hit"]], on=["video", "tid", "start_key"], how="left")
    if tube is not None and len(tube):
        x = tube.rename(columns={"start": "start_key"}).copy()
        names = {"a person smoking a cigarette": "x_smoking", "a person vaping": "x_vaping", "a person drinking from a bottle": "x_drinking",
                 "a person talking on a mobile phone": "x_phone", "a person eating": "x_eating", "a person touching their face": "x_touching",
                 "a person standing and talking": "x_talking"}
        for src, dst in names.items():
            if f"p_{src}" in x.columns:
                x[dst] = x[f"p_{src}"]
        if "videomae_k_smoking" in x.columns:
            x["v_smoking"] = x["videomae_k_smoking"]
            dr = [c for c in x.columns if c.startswith("videomae_k_drinking") or c == "videomae_k_tasting beer"]
            ea = [c for c in x.columns if c.startswith("videomae_k_eating")]
            x["v_drinking_any"] = x[dr].sum(axis=1) if dr else np.nan
            x["v_eating_any"] = x[ea].sum(axis=1) if ea else np.nan
            x["v_top1_p"] = x.get("videomae_top1_p")
        keep = ["video", "tid", "start_key"] + [c for c in x.columns if c.startswith(("x_", "v_"))]
        t = t.merge(x[keep], on=["video", "tid", "start_key"], how="left")
    return t.drop(columns=["start_key"])


def feature_columns(groups: list[str] | None = None, table: pd.DataFrame | None = None) -> list[str]:
    cols = []
    for g, c in FEATURE_GROUPS.items():
        if groups is None or g in groups:
            cols += c
    return [c for c in cols if table is None or c in table.columns]


def collect_runs(root: Path = OUTPUTS / "runs") -> list[tuple[str, Path]]:
    """Все запуски с готовыми циклами: [(video_id, каталог запуска)]."""
    out = []
    if not root.exists():
        return out
    for vdir in sorted(root.iterdir()):
        for rd in sorted(vdir.iterdir()):
            if (rd / "pose" / "tracks.parquet").exists() and (rd / "cycles").exists():
                out.append((vdir.name, rd))
    from .stages import drop_overlapping_runs   # локальный импорт: dataset остаётся лёгким при импорте (stages тянет позу и трекинг)

    return drop_overlapping_runs(out)
