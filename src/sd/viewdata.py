"""Данные для просмотра и разметки в браузере: компактные JSON-представления треков, эталона, событий и ошибок одного клипа. Без Streamlit — тестируется отдельно."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import OUTPUTS

RUNS_DIR = OUTPUTS / "runs"


def find_pose_runs(clip_id: str, root: Path | None = None) -> list[Path]:
    """Каталоги `pose/` с готовыми треками клипа (новые сверху): `outputs/runs/<clip_id>/<run_id>/pose`."""
    base = (root or RUNS_DIR) / clip_id
    runs = [p.parent for p in base.glob("*/pose/tracks.parquet")] if base.exists() else []
    return sorted(runs, key=lambda p: (p / "tracks.parquet").stat().st_mtime, reverse=True)


def tracks_payload(pose_dir: str | Path, min_height_px: float = 0.0) -> dict:
    """{'tracks': {tid: [[t, x1, y1, x2, y2], …]}, 'src_w', 'src_h'} — рамки в пикселях исходного кадра (время округлено до 0.01 с)."""
    from .tracks import Tracks

    tr = Tracks.load(pose_dir)
    df = tr.df
    out: dict[str, list] = {}
    for tid, g in df.groupby("tid"):
        if float(g["h"].median()) < min_height_px:
            continue
        g = g.sort_values("t")
        out[str(int(tid))] = [[round(float(r.t), 2), int(r.x1), int(r.y1), int(r.x2), int(r.y2)] for r in g.itertuples()]
    vi = tr.meta.get("video_info", {})
    return dict(tracks=out, src_w=int(vi.get("width") or 0), src_h=int(vi.get("height") or 0))


def box_at(tracks: dict, tid: str | int, t: float) -> tuple[float, float, float, float] | None:
    """Рамка человека в ближайший к `t` момент (для рамки эталона в peak)."""
    pts = tracks.get(str(tid))
    if not pts:
        return None
    ts = np.array([p[0] for p in pts])
    p = pts[int(np.abs(ts - t).argmin())]
    return float(p[1]), float(p[2]), float(p[3]), float(p[4])


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def gt_payload(gt: pd.DataFrame, clip_id: str) -> list[dict]:
    out = []
    for r in gt[gt["clip_id"] == clip_id].sort_values("start_sec").itertuples():
        box = [_f(r.x1), _f(r.y1), _f(r.x2), _f(r.y2)]
        out.append(dict(id=str(r.id), start=float(r.start_sec), end=float(r.end_sec), label=str(r.label), person=str(r.person_gt_id or ""), peak=_f(r.peak_sec),
                        box=box if all(v is not None for v in box) else None, note=str(r.note or "")))
    return out


def events_payload(events: pd.DataFrame, clip_id: str, threshold: float = 0.0) -> list[dict]:
    if events.empty:
        return []
    e = events[(events["clip_id"] == clip_id) & (events["confidence"] >= threshold)]
    return [dict(tid=int(r.person_track_id), start=float(r.start_sec), end=float(r.end_sec), conf=_f(r.confidence), note="") for r in e.itertuples()]


def errors_payload(errors: pd.DataFrame, clip_id: str) -> list[dict]:
    if errors.empty:
        return []
    e = errors[errors["clip_id"] == clip_id].sort_values("start")
    return [dict(kind=str(r.kind), start=_f(r.start), end=_f(r.end), conf=_f(r.confidence), cause=str(r.cause), text=str(r.cause_ru)) for r in e.itertuples() if _f(r.start) is not None]
