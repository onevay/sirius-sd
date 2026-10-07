"""Визуальная проверка: монтажи кадров по циклам и «ленты» кропа человека по времени.

Нужны, чтобы проверять автомат циклов глазами, не имея разметки: что именно детектор принял за жест (затяжка / питьё / телефон / касание лица),
и какие жесты он пропустил. Те же монтажи удобно просматривать при ручной разметке.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .tracks import Tracks
from .video_io import iter_frames

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _crop(img: np.ndarray, box, size: int, upper: float = 0.7, margin: float = 0.25) -> np.ndarray:
    """Квадратный кроп верхней части человека (рамка + запас) -> size×size."""
    x1, y1, x2, y2 = box
    h_up = (y2 - y1) * upper
    side = max(x2 - x1, h_up) * (1 + margin)
    cx, cy = (x1 + x2) / 2, y1 + h_up / 2
    X1, Y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    X2, Y2 = int(round(X1 + side)), int(round(Y1 + side))
    H, W = img.shape[:2]
    pad = max(0, -X1, -Y1, X2 - W, Y2 - H)
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img
    c = src[Y1 + pad:Y2 + pad, X1 + pad:X2 + pad]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_CUBIC) if c.size else np.zeros((size, size, 3), np.uint8)


def _box_at(rows: pd.DataFrame, t: float):
    ts = rows.t.values
    return tuple(float(np.interp(t, ts, rows[c].values)) for c in ("x1", "y1", "x2", "y2"))


def _label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 15), (0, 0, 0), -1)
    cv2.putText(img, text, (3, 11), FONT, 0.38, color, 1, cv2.LINE_AA)
    return img


def cycle_montage(video: str | Path, tr: Tracks, cycles: pd.DataFrame, out_png: Path, size: int = 150, per_image: int = 8,
                  series: pd.DataFrame | None = None, title: str = "") -> list[Path]:
    """По строке на цикл: 6 кропов (старт подъёма, вход ко рту, пик, выход, конец + ближайший кадр до начала) с подписью d и временем.

    Возвращает список PNG (по `per_image` циклов на изображение).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stride = int(tr.meta.get("stride", 1))
    paths = []
    rows_img: list[np.ndarray] = []
    for k, c in enumerate(cycles.itertuples()):
        rows, _ = tr.of(int(c.tid))
        if rows.empty:
            continue
        ts = [max(c.start - 0.5, 0.0), c.start, c.mouth_in, c.peak_t, c.mouth_out, c.end + 0.3]
        names = ["до", "старт", "рот вх.", "пик", "рот вых.", "после"]
        frames = {fr.idx: fr for fr in iter_frames(video, max(ts[0] - 0.1, 0), ts[-1] + 0.1, 1)}
        if not frames:
            continue
        idxs = np.array(sorted(frames))
        tt = np.array([frames[i].t for i in idxs])
        cells = []
        for t, nm in zip(ts, names):
            j = int(idxs[np.abs(tt - t).argmin()])
            fr = frames[j]
            cell = _crop(fr.img, _box_at(rows, fr.t), size)
            dval = ""
            if series is not None:
                s = series[(series.tid == c.tid)]
                if len(s):
                    sv = s.iloc[int(np.abs(s.t.values - fr.t).argmin())]
                    dval = f" d={sv.d:.2f}" if np.isfinite(sv.d) else " d=NaN"
            cells.append(_label(cell, f"{nm} {fr.t:.1f}s{dval}"))
        row = np.hstack(cells)
        cap = np.zeros((16, row.shape[1], 3), np.uint8)
        cv2.putText(cap, f"#{k} ID{int(c.tid)} {Path(str(getattr(c, 'video', title))).name[:26]} hold={c.hold:.1f}s dmin={c.d_min:.2f} hand={'LR'[int(c.hand)] if c.hand in (0, 1) else '?'}",
                    (3, 12), FONT, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        rows_img.append(np.vstack([cap, row]))
        if len(rows_img) == per_image or k == len(cycles) - 1:
            p = out_png.with_name(f"{out_png.stem}_{len(paths) + 1:02d}.png")
            cv2.imwrite(str(p), np.vstack(rows_img))
            paths.append(p)
            rows_img = []
    if rows_img:
        p = out_png.with_name(f"{out_png.stem}_{len(paths) + 1:02d}.png")
        cv2.imwrite(str(p), np.vstack(rows_img))
        paths.append(p)
    return paths


def timeline_sheet(video: str | Path, tr: Tracks, tid: int, t0: float, t1: float, out_png: Path, step: float = 1.0, size: int = 120,
                   cols: int = 10, series: pd.DataFrame | None = None, cycles: pd.DataFrame | None = None) -> Path:
    """«Лента» кропа человека каждые `step` секунд: быстро увидеть, где рука у рта, и сверить с найденными циклами."""
    rows, _ = tr.of(tid)
    times = np.arange(max(t0, rows.t.min()), min(t1, rows.t.max()), step)
    frames = {fr.idx: fr for fr in iter_frames(video, times[0] - 0.1, times[-1] + 0.1, 1)} if len(times) else {}
    idxs = np.array(sorted(frames))
    tt = np.array([frames[i].t for i in idxs]) if len(idxs) else np.array([])
    cells = []
    for t in times:
        fr = frames[int(idxs[np.abs(tt - t).argmin()])]
        cell = _crop(fr.img, _box_at(rows, fr.t), size)
        txt = f"{fr.t:.0f}s"
        col = (255, 255, 255)
        if series is not None:
            s = series[series.tid == tid]
            sv = s.iloc[int(np.abs(s.t.values - fr.t).argmin())]
            txt += f" d={sv.d:.2f}" if np.isfinite(sv.d) else " d=-"
        if cycles is not None and len(cycles) and ((cycles.tid == tid) & (cycles.start <= fr.t) & (cycles.end >= fr.t)).any():
            col = (0, 0, 255)
            txt += " CYCLE"
        cells.append(_label(cell, txt, col))
    if not cells:
        raise RuntimeError("нет кадров в интервале")
    while len(cells) % cols:
        cells.append(np.zeros((size, size, 3), np.uint8))
    sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    return out_png
