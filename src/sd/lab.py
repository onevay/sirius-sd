"""Лаборатория, часть 1: этапы конвейера по отдельности на одном клипе — без записи в журнал прогонов и эталон.

Поза и трекинг уже посчитаны и лежат в кэше (`outputs/runs/<клип>/<запуск>/pose`); всё, что идёт после (ряд d_t, автомат циклов, признаки цикла, оценка классификатора, события),
пересчитывается за секунды при любой правке параметров. Результаты `recognize` пишутся в `outputs/lab/recognize` — отдельно от `outputs/recognize`, чтобы лабораторные пробы
не попадали в таблицы обучения (`oof_eval.collect` берёт последний запуск каждого клипа именно оттуда).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import copy
import dataclasses
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import calibrate as CAL
from . import evaluation as EV
from . import gt as GT
from . import profiles as PR
from . import stages
from .config import _wrap
from .paths import OUTPUTS
from .tracks import Tracks

LAB_RECOGNIZE = OUTPUTS / "lab" / "recognize"

#: параметры этапов, доступные ползунками (ключ → (этап, подпись, мин, макс, шаг)); остальные ключи правятся через «Конструктор»
STAGE_KNOBS: dict[str, tuple[str, str, float, float, float]] = {
    "features.mouth_offset": ("series", "Положение рта (доля от носа к плечам)", 0.0, 0.8, 0.05),
    "features.median_sec": ("series", "Медианный фильтр d_t, с", 0.0, 1.0, 0.05),
    "features.smooth_sec": ("series", "Сглаживание Савицкого–Голея, с", 0.0, 1.5, 0.05),
    "features.hand_extend": ("series", "Продление кисти от локтя", 0.0, 1.0, 0.05),
    "cycles.th_in": ("cycles", "Порог входа в «у рта»", 0.2, 1.5, 0.05),
    "cycles.th_out": ("cycles", "Порог выхода из «у рта»", 0.3, 2.0, 0.05),
    "cycles.hold_min_sec": ("cycles", "Минимальная пауза, с", 0.1, 2.0, 0.1),
    "cycles.hold_max_sec": ("cycles", "Максимальная пауза по регламенту, с", 2.0, 12.0, 0.5),
    "cycles.hold_cap_sec": ("cycles", "Верхний предел удержания, с", 5.0, 30.0, 1.0),
    "events.cycle_th": ("events", "Порог оценки цикла", 0.05, 0.95, 0.05),
    "events.confidence_threshold": ("events", "Порог уверенности события", 0.05, 0.95, 0.05),
    "events.merge_gap_sec": ("events", "Склейка циклов одного человека, с", 1.0, 40.0, 1.0),
    "events.window_sec": ("events", "Окно «два цикла», с", 5.0, 40.0, 1.0),
}


# ---------------------------------------------------------------------------------------------- клипы с кэшем позы
def list_clips() -> pd.DataFrame:
    """Запуски позы, пригодные для лаборатории: video_id, run_id, окно, к/с, модель, время создания, каталог."""
    rows = []
    base = OUTPUTS / "runs"
    for f in sorted(base.glob("*/*/pose/tracks.parquet")) if base.exists() else []:
        rd = f.parent.parent
        a, b = stages.run_window(rd)
        m = re.search(r"_(\d+(?:\.\d+)?)fps$", rd.name)
        parts = rd.name.split("_")
        rows.append(dict(video_id=rd.parent.name, run_id=rd.name, start=a, end=b, fps=float(m.group(1)) if m else np.nan, pose=parts[1] if len(parts) > 1 else "",
                         created=pd.to_datetime(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"), path=str(rd)))
    df = pd.DataFrame(rows, columns=["video_id", "run_id", "start", "end", "fps", "pose", "created", "path"])
    return df.sort_values(["video_id", "created"], ascending=[True, False], ignore_index=True)


def is_full(row: pd.Series) -> bool:
    return str(row.run_id).startswith("0-ends")


def load_clip(rd: str | Path) -> tuple[Tracks, Path]:
    rd = Path(rd)
    return Tracks.load(rd / "pose"), rd


# ---------------------------------------------------------------------------------------------- конфигурация
def set_path(cfg: Any, key: str, value: Any) -> None:
    """Присваивание по пути `a.b.c` (прямое: у старых запусков в сохранённом конфиге может не быть новых ключей)."""
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d:
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def build_cfg(rd: str | Path, profile: str | None = None, overrides: dict[str, Any] | None = None):
    """Конфигурация для этапов после позы: параметры позы, трекинга и частоты — из запуска (иначе кэш не сойдётся), остальное — из профиля и правок."""
    base = PR.load_named(profile).cfg() if profile else None
    cfg = CAL.cfg_current_for_run(Path(rd), base)
    cfg = _wrap(copy.deepcopy(dict(cfg)))
    for k, v in (overrides or {}).items():
        set_path(cfg, k, v)
    return cfg


def changed(cfg_a, cfg_b, keys: list[str] | None = None) -> dict[str, tuple[Any, Any]]:
    """Ключи ручек, значения которых в двух конфигурациях различаются: {ключ: (было, стало)}."""
    out = {}
    for k in keys or list(STAGE_KNOBS):
        a, b = cfg_a, cfg_b
        try:
            for p in k.split("."):
                a, b = a[p], b[p]
        except (KeyError, TypeError):
            continue
        if a != b:
            out[k] = (a, b)
    return out


def knob_value(cfg, key: str) -> float:
    d = cfg
    for p in key.split("."):
        d = d[p]
    return float(d)


# ---------------------------------------------------------------------------------------------- этапы
def stage_frames(tr: Tracks, rd: Path, cfg) -> dict[str, pd.DataFrame]:
    """Ряд d_t, циклы, отброшенные жесты и состояния автомата для конфигурации (кэшируется по хэшу параметров этапов — повтор мгновенный)."""
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, st = stages.stage_cycles(ser, cfg, rd)
    return dict(series=ser, cycles=cyc, rejected=rej, states=st)


def recognize_clip(rd: str | Path, cfg, profile: str, *, bundle: str | None = None, light: bool = False) -> dict:
    """Полный проход `pipeline.recognize` по кэшу позы (оценка циклов и события). `bundle` подменяет классификатор профиля; `light` — без признаков предмета, фото и VLM
    (быстро, но пакетам, которым они нужны, не хватит признаков — оценки приблизительные; в предупреждениях это видно)."""
    from . import pipeline as PL

    rd = Path(rd)
    tr, _ = load_clip(rd)
    prof = PR.load_named(profile)
    opts = prof.options_obj(render=False)
    if bundle:
        opts = dataclasses.replace(opts, cycle_bundle=bundle, cycle_bundle_full=None)
    if light:
        opts = dataclasses.replace(opts, objects=(), photo_bundle=None, vlm_model=None, vlm_mode="off")
    a, b = stages.run_window(rd)
    end = None if rd.name.split("_")[0].endswith("-ends") else b          # "0-ends_…" — окно до конца видео
    res = PL.recognize(tr.meta.get("video"), a, end, cfg, opts, out_root=LAB_RECOGNIZE, force_pose=False)
    return dict(scored=res.cycles, events=res.events, meta=res.meta, out_dir=res.out_dir)


# ---------------------------------------------------------------------------------------------- оценка одного клипа
def clip_eval(events: pd.DataFrame, video_id: str, duration: float, threshold: float) -> dict:
    """Event F1 одного клипа по эталону: метрики, список событий системы со статусом (TP / FP / игнор) и интервалы эталона."""
    gt_df = GT.load()
    mine = GT.for_clip(gt_df, video_id)
    gts = GT.to_eval_gts(gt_df, {video_id})
    preds = EV.to_preds(events, video_id) if len(events) else []
    st = EV.Settings(threshold=threshold)
    m, res, kept = EV.metrics_at(preds, gts, st, {video_id: max(duration, 1.0)}, threshold)
    tp = {pi for pi, _ in res.matches}
    rows = []
    for i, p in enumerate(kept):
        status = "TP" if i in tp else ("игнор" if i not in res.fp_idx else "FP")
        rows.append(dict(start=p.start, end=p.end, peak=p.peak, confidence=p.confidence, статус=status))
    miss = [dict(start=gts[j].start, end=gts[j].end, метка=gts[j].label) for j in res.fn_idx]
    return dict(metrics=m, events=pd.DataFrame(rows, columns=["start", "end", "peak", "confidence", "статус"]), gt=mine, missed=pd.DataFrame(miss, columns=["start", "end", "метка"]),
                has_gt=len(mine) > 0)


def diff_events(base: pd.DataFrame, new: pd.DataFrame, tol: float = 3.0) -> pd.DataFrame:
    """Что изменилось в событиях: появились / пропали / остались (совпадение по перекрытию интервалов с допуском)."""
    rows = []
    used = set()
    for r in new.itertuples():
        hit = [i for i, b in enumerate(base.itertuples()) if b.start_sec - tol <= r.end_sec and r.start_sec - tol <= b.end_sec and i not in used]
        if hit:
            used.add(hit[0])
            rows.append(dict(статус="осталось", start=r.start_sec, end=r.end_sec, confidence=r.confidence))
        else:
            rows.append(dict(статус="появилось", start=r.start_sec, end=r.end_sec, confidence=r.confidence))
    for i, b in enumerate(base.itertuples()):
        if i not in used:
            rows.append(dict(статус="пропало", start=b.start_sec, end=b.end_sec, confidence=b.confidence))
    return pd.DataFrame(rows, columns=["статус", "start", "end", "confidence"]).sort_values("start", ignore_index=True)


# ---------------------------------------------------------------------------------------------- картинки
def pose_frame(video: str | Path, tr: Tracks, cfg, t: float) -> tuple[np.ndarray, float]:
    """Размеченный кадр (RGB) на момент ближайшего обработанного кадра: рамки, ID, скелет."""
    import cv2

    from .render import render_frame_img

    img, tt = render_frame_img(video, tr, cfg, t)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), tt


def cycle_strip(video: str | Path, tr: Tracks, cyc: pd.Series, size: int = 170) -> np.ndarray | None:
    """Полоса кропов человека по фазам цикла (до, старт, рот вх., пик, рот вых., после), RGB. None — кадры не прочитались."""
    import cv2

    from .review import _box_at, _crop, _label
    from .video_io import iter_frames

    rows, _ = tr.of(int(cyc.tid))
    if rows.empty:
        return None
    ts = [max(cyc.start - 0.5, 0.0), cyc.start, cyc.mouth_in, cyc.peak_t, cyc.mouth_out, cyc.end + 0.3]
    names = ["до", "старт", "рот вх.", "пик", "рот вых.", "после"]
    frames = {fr.idx: fr for fr in iter_frames(video, max(ts[0] - 0.1, 0), ts[-1] + 0.1, 1)}
    if not frames:
        return None
    idxs = np.array(sorted(frames))
    tt = np.array([frames[i].t for i in idxs])
    cells = []
    for t, nm in zip(ts, names):
        fr = frames[int(idxs[np.abs(tt - t).argmin()])]
        cells.append(_label(_crop(fr.img, _box_at(rows, fr.t), size), f"{nm} {fr.t:.1f}s"))
    return cv2.cvtColor(np.hstack(cells), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------------------------- графики
def dt_figure(series: pd.DataFrame, tid: int, cycles: pd.DataFrame, rejected: pd.DataFrame, cfg, gestures: pd.DataFrame | None = None, video_id: str = "",
              extra: dict[str, pd.Series] | None = None):
    """d_t выбранного человека: сырой и сглаженный ряд, пороги автомата, найденные циклы (зелёные полосы), отброшенные (серые, с причиной), метки жестов (ромбы)."""
    import plotly.graph_objects as go

    s = series[series.tid == tid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.t, y=s.d_raw, name="d сырой", line=dict(color="rgba(120,120,120,0.45)", width=1)))
    fig.add_trace(go.Scatter(x=s.t, y=s.d, name="d сглаженный", line=dict(color="#0f766e", width=2)))
    for name, y in (extra or {}).items():
        fig.add_trace(go.Scatter(x=s.t, y=y, name=name, line=dict(width=1, dash="dot")))
    fig.add_hline(y=float(cfg["cycles"]["th_in"]), line_dash="dash", line_color="#d97706", annotation_text="th_in")
    fig.add_hline(y=float(cfg["cycles"]["th_out"]), line_dash="dash", line_color="#b45309", annotation_text="th_out")
    for r in cycles[cycles.tid == tid].itertuples():
        fig.add_vrect(x0=r.start, x1=r.end, fillcolor="rgba(22,163,74,0.18)", line_width=0)
        fig.add_vline(x=r.peak_t, line_color="rgba(22,163,74,0.8)", line_width=1)
    for r in rejected[rejected.tid == tid].itertuples():
        fig.add_vrect(x0=r.t0, x1=r.t1, fillcolor="rgba(120,120,120,0.16)", line_width=0)
        fig.add_annotation(x=(r.t0 + r.t1) / 2, y=0.02, yref="paper", text=str(r.reason), showarrow=False, font=dict(size=9, color="gray"), textangle=-90)
    if gestures is not None and len(gestures):
        g = gestures[gestures.video == video_id]
        if len(g):
            from .gestures import LABEL_RU

            fig.add_trace(go.Scatter(x=g.peak_t, y=[0.12] * len(g), mode="markers", marker=dict(symbol="diamond", size=9, color=["#16a34a" if x == "smoke" else "#7c3aed" for x in g.label]),
                                     customdata=[LABEL_RU.get(x, x) for x in g.label], hovertemplate="%{x:.1f} с · %{customdata}<extra>метка жеста</extra>", name="метки жестов (зелёные — затяжка)"))
    ymax = float(np.nanpercentile(s.d_raw.dropna(), 98)) if s.d_raw.notna().any() else 3.0
    fig.update_yaxes(range=[0, min(max(ymax, 2.0), 6.0)], title="расстояние кисть–рот, ширин плеч")
    fig.update_xaxes(title="время, с")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12))
    return fig


def events_timeline(duration: float, rows: list[tuple[str, pd.DataFrame, str]], height: int = 220):
    """Дорожки интервалов: [(название, таблица start/end, цвет)] — эталон и события системы друг под другом."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (name, df, color) in enumerate(rows):
        for r in df.itertuples():
            fig.add_shape(type="rect", x0=r.start, x1=max(r.end, r.start + 0.2), y0=i - 0.35, y1=i + 0.35, fillcolor=color, line_width=0, opacity=0.8)
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=[r[0] for r in rows], range=[-0.6, len(rows) - 0.4])
    fig.update_xaxes(range=[0, max(duration, 1.0)], title="время, с")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig
