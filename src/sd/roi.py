"""Зона интереса (ROI, руководство §6, шаг 1): человек «в зоне», если нижняя середина его рамки (точка стоп) лежит внутри полигона в момент пика события.

Файл `roi.json`:  {"polygon": [[x, y], ...]}  или  {"zones": [[[x, y], ...], ...]}  (координаты — в пикселях исходного кадра; событие проходит, если оно в любой зоне).
Без файла фильтра нет. Чистый Python, без cv2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

Poly = Sequence[Sequence[float]]


def load(path: str | Path) -> list[list[tuple[float, float]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = raw.get("zones") or ([raw["polygon"]] if "polygon" in raw else None)
    if not zones:
        raise ValueError("roi.json: нужен ключ polygon или zones")
    out = []
    for z in zones:
        if len(z) < 3:
            raise ValueError("полигон ROI: минимум 3 точки")
        out.append([(float(x), float(y)) for x, y in z])
    return out


def point_in_polygon(x: float, y: float, poly: Poly) -> bool:
    """Чётно-нечётное правило (луч вправо); точка на границе считается внутри."""
    n, inside = len(poly), False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x2 - x1) * (y - y1) == (y2 - y1) * (x - x1) and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def box_in_zones(box: tuple[float, float, float, float], zones: list[Poly]) -> bool:
    x1, y1, x2, y2 = box
    if not np.isfinite([x1, y1, x2, y2]).all():
        return False
    fx, fy = (x1 + x2) / 2, y2
    return any(point_in_polygon(fx, fy, z) for z in zones)


def filter_events(events: pd.DataFrame, zones: list[Poly] | None) -> pd.DataFrame:
    """Оставляет события, у которых точка стоп в `peak_sec` внутри ROI. `zones=None` — без фильтра."""
    if not zones or events.empty:
        return events
    keep = [box_in_zones((r.x1, r.y1, r.x2, r.y2), zones) for r in events.itertuples()]
    return events[np.array(keep, bool)].reset_index(drop=True)
