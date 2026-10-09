"""Проверка файла предсказаний перед сдачей (руководство §10.1): формат, уникальность, времена, рамки внутри кадра. Своя копия проверок — валидатор организаторов может быть строже,
поэтому после получения его нужно прогнать тоже. Возвращает список проблем (пусто — формат в порядке)."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["camera_id", "clip_id", "event_id", "start_sec", "end_sec", "confidence", "label", "person_track_id", "peak_sec", "x1", "y1", "x2", "y2"]


def validate(df: pd.DataFrame, frame_wh: dict[str, tuple[int, int]] | None = None) -> list[str]:
    out: list[str] = []
    miss = [c for c in COLUMNS if c not in df.columns]
    if miss:
        return [f"нет колонок: {miss}"]
    if df.empty:
        return out
    if list(df.columns) != COLUMNS:
        out.append("порядок колонок отличается от формата организаторов")
    num = df[["start_sec", "end_sec", "confidence", "peak_sec", "x1", "y1", "x2", "y2"]].apply(pd.to_numeric, errors="coerce")
    bad_nan = int((~np.isfinite(num.to_numpy(float))).any(axis=1).sum())
    if bad_nan:
        out.append(f"{bad_nan} строк с пустыми или нечисловыми значениями (в т.ч. нет рамки)")
    if df["event_id"].duplicated().any():
        out.append(f"event_id не уникален: {int(df['event_id'].duplicated().sum())} повторов")
    dup_tid = df.groupby("clip_id")["person_track_id"].apply(lambda s: s.map(type).nunique() > 1)
    if dup_tid.any():
        out.append("person_track_id разного типа внутри клипа")
    if (num.start_sec < 0).any() or (num.end_sec <= num.start_sec).any():
        out.append("есть события с start_sec < 0 или end_sec ≤ start_sec")
    if ((num.peak_sec < num.start_sec) | (num.peak_sec > num.end_sec)).any():
        out.append("peak_sec вне интервала события")
    if ((num.confidence < 0) | (num.confidence > 1)).any():
        out.append("confidence вне [0, 1]")
    if ((num.x2 <= num.x1) | (num.y2 <= num.y1)).any():
        out.append("рамка: нужно x1 < x2 и y1 < y2")
    if frame_wh:
        for r in df.itertuples():
            wh = frame_wh.get(r.clip_id)
            if wh and not (0 <= r.x1 and 0 <= r.y1 and r.x2 <= wh[0] and r.y2 <= wh[1]):
                out.append(f"рамка вне кадра {wh[0]}×{wh[1]} в {r.event_id}")
                break
    return out
