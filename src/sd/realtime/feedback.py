"""Решения оператора → эталон и обучение: подтверждённая тревога становится POSITIVE, ложная — NEGATIVE (сложный негатив для дообучения и проверки).

Так мониторинг сам пополняет `labels/events_gt.csv` (то же, что размечается на странице «Разметка»), и отзывы идут в обучение классификатора (руководство §8.4, активное обучение).
Времена пересчитываются из секунд потока в секунды ФРАГМЕНТА (клип эталона = фрагмент). Повторный экспорт не дублирует строки (ключ — заметка `alert:<id>`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import gt as GT
from ..paths import from_portable, video_id


def alert_to_gt_row(a: dict) -> dict | None:
    """Строка эталона для решённой тревоги или None (нет файла фрагмента / решение не однозначно)."""
    if a["status"] not in ("confirmed", "false") or not a.get("chunk"):
        return None
    chunk = from_portable(a["chunk"])
    if chunk is None:
        return None
    off = float(a.get("chunk_offset") or 0.0)
    s, e = max(0.0, float(a["start_sec"]) - off), max(0.0, float(a["end_sec"]) - off)
    if e - s < 0.2:
        return None
    ok = a["status"] == "confirmed"
    box = a.get("box")
    if ok and not box:
        return None
    return dict(clip_id=video_id(chunk), start=s, end=e, label="POSITIVE" if ok else "NEGATIVE", person=a.get("tid"), peak=max(s, float(a["peak_sec"]) - off), box=box if ok else None,
                camera_id=str(a.get("camera_id") or ""), note=f"alert:{a['id']}", labeler=a.get("reviewer") or "")


def export_reviewed(store, path: str | Path | None = None) -> dict:
    """Переносит решения оператора в эталон. Возвращает {added, skipped_no_clip, already}."""
    gt = GT.load(path)
    have = set(gt["note"].astype(str))
    out = dict(added=0, skipped=0, already=0)
    df = store.list_alerts(status=["confirmed", "false"], limit=100000)
    for _, r in df.iterrows():
        a = {k: (None if pd.isna(v) else v) for k, v in r.items() if k != "box"}
        a["box"] = r["box"]
        row = alert_to_gt_row(a)
        if row is None:
            out["skipped"] += 1
        elif row["note"] in have:
            out["already"] += 1
        else:
            GT.add(row["clip_id"], row["start"], row["end"], row["label"], person=row["person"] if row["person"] is not None else "", peak=row["peak"], box=row["box"], note=row["note"],
                   camera_id=row["camera_id"], labeler=row["labeler"], path=path)
            out["added"] += 1
    return out
