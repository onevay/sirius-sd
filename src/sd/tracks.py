"""Контейнер треков (рамки + ключевые точки) с сохранением на диск и офлайн-склейкой фрагментов одного человека."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import iou


@dataclass
class Tracks:
    df: pd.DataFrame            # frame, t, tid, x1, y1, x2, y2, score, h  — по строке на (кадр, человек)
    kp: np.ndarray              # (N, K, 3): x, y, conf в пикселях ИСХОДНОГО кадра; порядок строк = порядок df
    frame_t: pd.DataFrame       # frame, t — ВСЕ обработанные кадры (чтобы пропуски детекций стали NaN)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    def save(self, d: str | Path) -> Path:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(d / "tracks.parquet", index=False)
        self.frame_t.to_parquet(d / "frame_t.parquet", index=False)
        np.save(d / "keypoints.npy", self.kp.astype(np.float32))
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return d

    @classmethod
    def load(cls, d: str | Path) -> "Tracks":
        d = Path(d)
        return cls(pd.read_parquet(d / "tracks.parquet"), np.load(d / "keypoints.npy"), pd.read_parquet(d / "frame_t.parquet"),
                   json.loads((d / "meta.json").read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ access
    @property
    def tids(self) -> list[int]:
        return sorted(self.df.tid.unique().tolist())

    def of(self, tid: int) -> tuple[pd.DataFrame, np.ndarray]:
        m = (self.df.tid == tid).values
        return self.df[m].reset_index(drop=True), self.kp[m]

    def box_at(self, tid: int, t: float) -> tuple[float, float, float, float] | None:
        """Рамка ВСЕГО человека на кадре, ближайшем к моменту t (для peak_sec)."""
        rows, _ = self.of(tid)
        if rows.empty:
            return None
        r = rows.iloc[int(np.abs(rows.t.values - t).argmin())]
        return float(r.x1), float(r.y1), float(r.x2), float(r.y2)

    def summary(self, min_height_px: float = 80.0) -> pd.DataFrame:
        g = self.df.groupby("tid")
        s = g.agg(t0=("t", "min"), t1=("t", "max"), n_det=("frame", "count"), h_med=("h", "median"), h_min=("h", "min"),
                  h_max=("h", "max"), score=("score", "mean")).reset_index()
        step = float(np.median(np.diff(self.frame_t.t.values))) if len(self.frame_t) > 2 else 0.1
        s["dur"] = s.t1 - s.t0
        s["coverage"] = (s.n_det / np.maximum(s.dur / step + 1, 1)).clip(upper=1.0)   # доля кадров трека с детекцией
        s["ignore_small"] = s.h_med < min_height_px
        return s


def _center_dist(a, b) -> float:
    return float(np.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2))


def stitch_tracks(tr: Tracks, cfg_tracking: dict) -> Tracks:
    """Склейка фрагментов одного человека (offline): A закончился, B начался в пределах max_gap, рамки близки.

    Критерий близости: IoU(последняя рамка A, первая рамка B) >= min_iou ИЛИ расстояние центров <= max_center_dist_h·высота.
    Одновременно живущие треки не склеиваются (A.t1 < B.t0), цепочки допустимы. Короткие треки (< min_track_sec) удаляются.
    """
    s = cfg_tracking["stitch"]
    df = tr.df
    if df.empty:
        return tr
    first = df.sort_values("t").groupby("tid").first()
    last = df.sort_values("t").groupby("tid").last()
    tids = sorted(first.index, key=lambda i: first.loc[i, "t"])
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for b in tids:
        best, best_sc = None, -1e9
        fb = first.loc[b]
        box_b = (fb.x1, fb.y1, fb.x2, fb.y2)
        for a in tids:
            if a == b or a in succ or last.loc[a, "t"] >= fb.t or fb.t - last.loc[a, "t"] > s["max_gap_sec"]:
                continue
            la = last.loc[a]
            box_a = (la.x1, la.y1, la.x2, la.y2)
            i = iou(box_a, box_b)
            dist = _center_dist(box_a, box_b) / max(la.h, 1.0)
            if i >= s["min_iou"] or dist <= s["max_center_dist_h"]:
                sc = i - dist
                if sc > best_sc:
                    best, best_sc = a, sc
        if best is not None and b not in pred:
            succ[best], pred[b] = b, best
    root: dict[int, int] = {}
    for t0 in tids:
        if t0 in pred:
            continue
        cur = t0
        while True:
            root[cur] = t0
            if cur not in succ:
                break
            cur = succ[cur]
    new = df.copy()
    new["tid"] = new.tid.map(lambda x: root.get(x, x))
    # короткие треки — долой
    dur = new.groupby("tid").t.agg(lambda x: x.max() - x.min())
    keep = new.tid.map(dur >= cfg_tracking["min_track_sec"]).values
    out = Tracks(new[keep].reset_index(drop=True), tr.kp[keep], tr.frame_t, dict(tr.meta))
    out.meta["stitch"] = dict(merged={int(k): int(v) for k, v in root.items() if k != v}, dropped_short=int((~keep).sum()),
                              tracks_before=len(tids), tracks_after=len(out.tids))
    return out
