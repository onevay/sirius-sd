"""Качество детекции НАЧАЛА цикла (кандидатов-жестов): recall по эталону жестов, точность, точность времени старта, вклад гибридных триггеров.

Проблема: эталон начала цикла не может строиться тем же автоматом, который проверяется. Поэтому используется метод «пула» (как при оценке поиска):
кандидаты берутся из НЕСКОЛЬКИХ независимых генераторов (автомат при разных порогах, продление кисти, триггеры по предмету и по фото-модели),
объединяются по времени, и каждого кандидата размечают глазами: настоящий жест «рука ко рту» или нет, какой (затяжка / питьё / телефон / касание…),
и корректно ли найден старт. Recall генератора = доля настоящих жестов пула, которые он нашёл. Пул неполон (жест, который не нашёл ни один генератор, в нём не
виден) — это оценивается отдельно проверкой случайных окон «вне кандидатов» (`audit_windows`).
Метки — суждения ассистента по кадрам, не эталон организаторов; цифры — «насколько один генератор лучше другого», не итоговое качество.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd

from . import stages
from .calibrate import cfg_current_for_run, list_runs, with_params
from .paths import OUTPUTS, ROOT
from .tracks import Tracks

GT_CSV = ROOT / "docs" / "review" / "gesture_gt.csv"
POOL = OUTPUTS / "analysis" / "gesture_pool.parquet"
GESTURE_LABELS = ("smoke", "drink", "phone", "eat", "touch_face", "other_neg")
NON_GESTURE = ("no_gesture",)
# генераторы кандидатов: имя -> (th_in, th_out, hand_extend); main = рабочие пороги из default.yaml
GENERATORS = {"main": None, "guide": (0.35, 0.55, 0.0), "guide_k05": (0.35, 0.55, 0.5), "lenient": (1.0, 1.2, 0.0), "lenient_k05": (0.8, 1.0, 0.5)}


def generator_cycles(runs: list[tuple[str, Path]] | None = None, generators: dict | None = None, progress=None) -> pd.DataFrame:
    """Циклы каждого генератора по всем запускам (поза не пересчитывается). Колонки: video, run, gen, tid, start, mouth_in, mouth_out, end, peak_t, hold, hand, d_min."""
    runs = runs if runs is not None else list_runs(pose_sig="rtmpose")
    gens = generators or GENERATORS
    rows = []
    for i, (vname, rd) in enumerate(runs):
        tr = Tracks.load(rd / "pose")
        base = cfg_current_for_run(rd)
        summ = tr.summary(base["video"]["min_person_height_px"])
        ok_tids = set(summ[~summ.ignore_small].tid.tolist())
        for gname, p in gens.items():
            cfg = base if p is None else with_params(base, *p)
            ser = stages.stage_features(tr, cfg, rd)
            cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
            for r in cyc[cyc.tid.isin(ok_tids)].itertuples():
                rows.append(dict(video=vname, run=str(rd), gen=gname, tid=int(r.tid), start=round(float(r.start), 3), mouth_in=float(r.mouth_in), mouth_out=float(r.mouth_out),
                                 end=float(r.end), peak_t=float(r.peak_t), hold=float(r.hold), hand=int(r.hand), d_min=float(r.d_min)))
        if progress:
            progress(i + 1, len(runs))
    return pd.DataFrame(rows)


def build_pool(cyc: pd.DataFrame, min_overlap: float = 0.3, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Кластеры циклов разных генераторов одного трека, пересекающихся во времени (≥ min_overlap с или центр одного внутри другого) → строка пула.

    Эталонный интервал кандидата — цикл генератора `main`, если он есть, иначе самый «ранний» по приоритету генераторов. Колонка `found_by` — множество генераторов.
    `extra` — кандидаты внешних триггеров (video, tid, start, end, peak_t, gen): они участвуют в кластеризации так же.
    """
    c = pd.concat([cyc, extra], ignore_index=True) if extra is not None and len(extra) else cyc.copy()
    prio = {g: i for i, g in enumerate(["main", "guide", "guide_k05", "lenient", "lenient_k05"])}
    c["_p"] = c.gen.map(lambda g: prio.get(g, 99))
    out = []
    for (v, tid), g in c.groupby(["video", "tid"]):
        g = g.sort_values(["start", "_p"]).reset_index(drop=True)
        cl: list[list[int]] = []
        spans: list[tuple[float, float]] = []
        for i, r in g.iterrows():
            placed = False
            for k, (a, b) in enumerate(spans):
                ov = min(b, r.end) - max(a, r.start)
                if ov >= min_overlap or (a <= r.peak_t <= b) or (r.start <= (a + b) / 2 <= r.end):
                    cl[k].append(i)
                    spans[k] = (min(a, r.start), max(b, r.end))
                    placed = True
                    break
            if not placed:
                cl.append([i])
                spans.append((r.start, r.end))
        for members in cl:
            m = g.loc[members].sort_values("_p")
            ref = m.iloc[0]
            out.append(dict(video=v, run=ref.run if "run" in m else "", tid=int(tid), start=float(ref.start), mouth_in=float(ref.get("mouth_in", np.nan)),
                            mouth_out=float(ref.get("mouth_out", np.nan)), end=float(ref.end), peak_t=float(ref.peak_t), hold=float(ref.get("hold", np.nan)),
                            hand=int(ref.get("hand", -1)) if pd.notna(ref.get("hand", np.nan)) else -1, d_min=float(ref.get("d_min", np.nan)),
                            ref_gen=ref.gen, found_by=",".join(sorted(set(m.gen), key=lambda x: prio.get(x, 99))), n_cycles=len(m)))
    pool = pd.DataFrame(out).sort_values(["video", "start"]).reset_index(drop=True)
    pool["start"] = pool.start.round(3)
    return pool


def attach_gt(pool: pd.DataFrame, gt: pd.DataFrame, tol: float = 1.0) -> pd.DataFrame:
    """Метки жестов к кандидатам пула: по (video, tid) и ближайшему peak_t в пределах tol секунд. Колонки gt: video, tid, peak_t, label, start_quality, note."""
    p = pool.copy()
    p["label"], p["start_quality"], p["gt_note"] = None, None, ""
    for i, r in p.iterrows():
        m = gt[(gt.video == r.video) & (gt.tid == r.tid) & ((gt.peak_t - r.peak_t).abs() <= tol)]
        if len(m):
            b = m.iloc[int((m.peak_t - r.peak_t).abs().values.argmin())]
            p.at[i, "label"], p.at[i, "start_quality"], p.at[i, "gt_note"] = b.label, b.get("start_quality"), b.get("note", "")
    return p


def candidate_montage(cands: pd.DataFrame, out_dir: Path, size: int = 140, per_image: int = 6, ids: list[int] | None = None) -> pd.DataFrame:
    """Монтажи для разметки кандидатов пула: по строке на кандидата, 7 кропов человека: старт−0.6, старт−0.2, старт, старт+0.25, вход ко рту, пик, выход.

    Подпись ASCII (шрифт OpenCV): `#id`, номер видео, ID трека, интервал, пауза, d_min; соответствие id → видео/время пишется в `cand_index.csv`.
    Возвращает таблицу индекса (id, video, tid, start, peak_t, png).
    """
    import cv2

    from .review import FONT, _box_at, _crop, _label
    from .video_io import iter_frames

    out_dir.mkdir(parents=True, exist_ok=True)
    ids = ids if ids is not None else list(range(len(cands)))
    vids = {v: i for i, v in enumerate(sorted(cands.video.unique()))}
    cache: dict[str, Tracks] = {}
    index, rows_img, png_names = [], [], []
    for cid, r in zip(ids, cands.itertuples()):
        if r.run not in cache:
            cache = {r.run: Tracks.load(Path(r.run) / "pose")}
        tr = cache[r.run]
        rows, _ = tr.of(int(r.tid))
        if rows.empty:
            continue
        t_in = r.mouth_in if np.isfinite(r.mouth_in) else r.start + 0.5
        ts = [r.start - 0.6, r.start - 0.2, r.start, r.start + 0.25, t_in, r.peak_t, (r.mouth_out if np.isfinite(r.mouth_out) else r.end) + 0.1]
        names = ["-0.6", "-0.2", "start", "+0.25", "in", "peak", "out"]
        frames = {fr.idx: fr for fr in iter_frames(tr.meta["video"], max(min(ts) - 0.1, 0), max(ts) + 0.1, 1)}
        if not frames:
            continue
        idxs = np.array(sorted(frames))
        tt = np.array([frames[i].t for i in idxs])
        cells = []
        for t, nm in zip(ts, names):
            fr = frames[int(idxs[np.abs(tt - t).argmin()])]
            cells.append(_label(_crop(fr.img, _box_at(rows, fr.t), size), f"{nm} {fr.t:.1f}s"))
        row = np.hstack(cells)
        cap = np.zeros((16, row.shape[1], 3), np.uint8)
        cv2.putText(cap, f"#{cid} v{vids[r.video]} ID{int(r.tid)} {r.start:.1f}-{r.end:.1f}s hold={r.hold:.1f} dmin={r.d_min:.2f} by:{str(r.found_by)[:28]}", (3, 12), FONT, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        rows_img.append(np.vstack([cap, row]))
        index.append(dict(id=cid, video=r.video, v=vids[r.video], tid=int(r.tid), start=float(r.start), peak_t=float(r.peak_t), found_by=r.found_by))
        if len(rows_img) == per_image:
            p = out_dir / f"cand_{len(png_names) + 1:02d}.png"
            cv2.imwrite(str(p), np.vstack(rows_img))
            png_names += [p.name]
            for k in range(len(index) - per_image, len(index)):
                index[k]["png"] = p.name
            rows_img = []
    if rows_img:
        p = out_dir / f"cand_{len(png_names) + 1:02d}.png"
        cv2.imwrite(str(p), np.vstack(rows_img))
        for k in range(len(index) - len(rows_img), len(index)):
            index[k]["png"] = p.name
    idx = pd.DataFrame(index)
    idx.to_csv(out_dir / "cand_index.csv", index=False, encoding="utf-8-sig")
    return idx


def audit_windows(pool: pd.DataFrame, n: int = 30, win: float = 2.0, step: float = 3.0, margin: float = 0.7, seed: int = 0,
                  runs: list[tuple[str, Path]] | None = None) -> pd.DataFrame:
    """Случайные окна `win` с по людям (≥ min_person_height_px), которые НЕ пересекаются ни с одним кандидатом пула (± margin с).

    Если в таких окнах есть настоящий жест «рука ко рту», пул (а значит и все генераторы) его пропустил: доля таких окон оценивает неполноту пула.
    Возвращает строки в формате кандидата (для `candidate_montage`): video, run, tid, start, mouth_in, mouth_out, end, peak_t, hold, hand, d_min, found_by="audit".
    """
    runs = runs if runs is not None else list_runs(pose_sig="rtmpose")
    rng = np.random.default_rng(seed)
    cands = []
    for vname, rd in runs:
        tr = Tracks.load(rd / "pose")
        cfg = cfg_current_for_run(rd)
        summ = tr.summary(cfg["video"]["min_person_height_px"])
        for r in summ[~summ.ignore_small].itertuples():
            busy = pool[(pool.video == vname) & (pool.tid == r.tid)]
            t = r.t0 + 0.5
            while t + win < r.t1 - 0.5:
                if not ((busy.start < t + win + margin) & (busy.end > t - margin)).any():
                    cands.append(dict(video=vname, run=str(rd), tid=int(r.tid), start=round(float(t), 3), mouth_in=t + 0.5, mouth_out=t + 1.5, end=t + win, peak_t=t + 1.0,
                                      hold=1.0, hand=-1, d_min=float("nan"), found_by="audit"))
                t += step
    c = pd.DataFrame(cands)
    if len(c) > n:
        c = c.iloc[np.sort(rng.choice(len(c), n, replace=False))].reset_index(drop=True)
    return c


def is_gesture(label) -> bool:
    return label in GESTURE_LABELS


def _found(s: pd.Series, gen: str) -> pd.Series:
    """Найден ли кандидат генератором `gen`; «main+lenient_k05» — объединение генераторов."""
    parts = gen.split("+")
    return s.map(lambda v: any(x in str(v).split(",") for x in parts))


def generator_table(pool: pd.DataFrame, gens: list[str], n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Для каждого генератора (или объединения «a+b»): найдено настоящих жестов (recall) и затяжек, доля настоящих среди его кандидатов (precision), число кандидатов,
    с 95% интервалом бутстрэпа по видео. Кандидаты без метки и с меткой «неясно» не учитываются ни в recall, ни в precision."""
    lab = pool[pool.label.notna() & (pool.label != "unsure")].copy()
    lab["gesture"] = lab.label.map(is_gesture)
    lab["puff"] = lab.label == "smoke"
    rng = np.random.default_rng(seed)
    vids = lab.video.unique()
    by_v = {v: lab[lab.video == v] for v in vids}

    def stat(d: pd.DataFrame, f: pd.Series) -> dict:
        n_g, n_p, n_f = int(d.gesture.sum()), int(d.puff.sum()), int(f.sum())
        hit_g, hit_p = int((f & d.gesture).sum()), int((f & d.puff).sum())
        return dict(recall_gesture=hit_g / n_g if n_g else np.nan, recall_puff=hit_p / n_p if n_p else np.nan,
                    precision_gesture=hit_g / n_f if n_f else np.nan, precision_puff=hit_p / n_f if n_f else np.nan, cand=n_f, gestures=n_g, puffs=n_p)

    rows = []
    for g in gens:
        pt = stat(lab, _found(lab.found_by, g))
        boots = []
        for _ in range(n_boot):
            d = pd.concat([by_v[v] for v in rng.choice(vids, size=len(vids), replace=True)], ignore_index=True)
            boots.append(stat(d, _found(d.found_by, g)))
        b = pd.DataFrame(boots)
        row = dict(generator=g, **pt)
        for k in ("recall_gesture", "recall_puff", "precision_gesture", "precision_puff"):
            row[f"{k}_lo"], row[f"{k}_hi"] = (float(np.nanpercentile(b[k], 2.5)), float(np.nanpercentile(b[k], 97.5))) if b[k].notna().sum() > 20 else (np.nan, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def start_quality_table(pool: pd.DataFrame, gen: str = "main") -> pd.DataFrame:
    """Точность старта у найденных генератором жестов: доли ok / late / early по меткам start_quality (оценка по кадрам с шагом ≈ 0.5 с)."""
    d = pool[_found(pool.found_by, gen) & pool.label.map(is_gesture) & pool.start_quality.notna()]
    if d.empty:
        return pd.DataFrame(columns=["start_quality", "n", "share"])
    t = d.start_quality.value_counts().rename_axis("start_quality").reset_index(name="n")
    t["share"] = t.n / t.n.sum()
    return t


def plot_generators(t: pd.DataFrame, out: Path, info: str = "") -> Path:
    """Точки с усами (95% по видео): recall затяжек, recall любых жестов и precision затяжек по генераторам кандидатов; справа — число кандидатов."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, muted, grid, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    series = [("recall_puff", "recall затяжек", "#2a78d6", -0.22), ("recall_gesture", "recall любых жестов «рука ко рту»", "#eb6834", 0.0), ("precision_puff", "precision затяжек", "#1baf7a", 0.22)]
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(t) + 2.0), facecolor=surface)
    ax.set_facecolor(surface)
    y = np.arange(len(t))[::-1]
    for col, name, color, dy in series:
        v, lo, hi = t[col].to_numpy(float), t[f"{col}_lo"].to_numpy(float), t[f"{col}_hi"].to_numpy(float)
        ax.errorbar(v, y + dy, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)], fmt="o", color=color, ecolor=color, elinewidth=1.4, capsize=2.5, ms=6, label=name, zorder=3)
        for xi, yi in zip(v, y + dy):
            ax.text(xi, yi + 0.13, f"{xi:.2f}", ha="center", va="bottom", fontsize=7.5, color=ink)
    ax.set_yticks(y, [f"{g}\\n{int(c)} канд." for g, c in zip(t.generator, t.cand)], fontsize=8.5, color=ink2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.7, len(t) - 0.2)
    ax.set_xlabel("доля (95% интервал бутстрэпом по видео); эталон — метки глазами по пулу кандидатов, не эталон организаторов", fontsize=8.5, color=ink2)
    ax.grid(axis="x", color=grid, lw=0.7, zorder=0)
    ax.tick_params(axis="x", labelsize=8, colors=muted, length=0)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8.5, labelcolor=ink2)
    fig.suptitle("Генераторы кандидатов-циклов: что находят и сколько лишнего", x=0.01, ha="left", fontsize=12, fontweight="bold", color=ink)
    fig.text(0.01, 0.935, info, ha="left", fontsize=8.5, color=ink2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=surface)
    plt.close(fig)
    return Path(out)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))
