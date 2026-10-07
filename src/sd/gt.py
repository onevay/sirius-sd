"""Эталон событий (ground truth) для оценки: интервалы POSITIVE / NEGATIVE / IGNORE по клипам. Формат — как у организаторов (руководство §9.2).

Файл `labels/events_gt.csv` (в git не попадает: приватность). Одна строка = один интервал одного человека:

    id, clip_id, camera_id, person_gt_id, start_sec, end_sec, label, peak_sec, x1, y1, x2, y2, note, labeler, ts

* `clip_id` — `paths.video_id` (папка__имя): уникален между папками, в отличие от имени файла.
* Время — секунды от начала видео (как `start_sec/end_sec` в предсказаниях).
* Рамка — рамка ВСЕГО человека в `peak_sec` в пикселях исходного кадра; нужна только для POSITIVE (по ней сопоставляется предсказание, IoU ≥ 0.30).
* Клип считается просмотренным, если в нём есть хотя бы одна строка. «Весь клип без курения» — NEGATIVE на всю длительность.

Модуль без тяжёлых зависимостей (pandas + numpy), чтобы разметка и оценка работали на любой машине.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .evaluate import GT
from .paths import LABELS as LABELS_DIR

LABEL_SET = ("POSITIVE", "NEGATIVE", "IGNORE")
COLUMNS = ["id", "clip_id", "camera_id", "person_gt_id", "start_sec", "end_sec", "label", "peak_sec", "x1", "y1", "x2", "y2", "note", "labeler", "ts"]
ORGANIZER_COLUMNS = ["camera_id", "clip_id", "person_gt_id", "start_sec", "end_sec", "label", "peak_sec", "x1", "y1", "x2", "y2"]
NUM = ["start_sec", "end_sec", "peak_sec", "x1", "y1", "x2", "y2"]


def default_path() -> Path:
    return LABELS_DIR / "events_gt.csv"


def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float if c in NUM else object) for c in COLUMNS})


def load(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else default_path()
    if not p.exists() or p.stat().st_size == 0:
        return empty()
    df = pd.read_csv(p, encoding="utf-8-sig", dtype={"clip_id": str, "person_gt_id": str, "camera_id": str, "id": str, "note": str, "labeler": str, "ts": str})
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    for c in ("person_gt_id", "camera_id", "note", "labeler"):
        df[c] = df[c].fillna("").astype(str).replace("nan", "")
    df["id"] = df["id"].where(df["id"].notna() & (df["id"] != ""), [uuid.uuid4().hex[:8] for _ in range(len(df))])
    return df[COLUMNS].reset_index(drop=True)


def save(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Атомарная запись (временный файл + замена): прерванная запись не портит файл разметки."""
    p = Path(path) if path else default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    os.close(fd)
    try:
        df[COLUMNS].to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def add(clip_id: str, start: float, end: float, label: str, *, person: str | int = "", peak: float | None = None, box: Iterable[float] | None = None,
        note: str = "", camera_id: str = "", labeler: str = "", path: str | Path | None = None) -> str:
    """Добавляет интервал и возвращает его id. Ошибка `ValueError` — при неверной метке/времени."""
    label = str(label).upper().strip()
    if label not in LABEL_SET:
        raise ValueError(f"label ∈ {LABEL_SET}, получено «{label}»")
    start, end = float(start), float(end)
    if not (np.isfinite(start) and np.isfinite(end)) or end <= start or start < 0:
        raise ValueError(f"интервал {start}–{end} с некорректен (нужно 0 ≤ начало < конец)")
    peak = (start + end) / 2 if peak is None else float(peak)
    if not start <= peak <= end:
        peak = min(max(peak, start), end)
    b = [float(v) for v in box] if box is not None else [np.nan] * 4
    if len(b) != 4:
        raise ValueError("box = (x1, y1, x2, y2)")
    if np.isfinite(b).all() and not (b[0] < b[2] and b[1] < b[3]):
        raise ValueError("рамка: нужно x1 < x2 и y1 < y2")
    rid = uuid.uuid4().hex[:8]
    row = dict(id=rid, clip_id=str(clip_id), camera_id=camera_id, person_gt_id=str(person) if person != "" else "", start_sec=round(start, 3), end_sec=round(end, 3),
               label=label, peak_sec=round(peak, 3), x1=b[0], y1=b[1], x2=b[2], y2=b[3], note=note, labeler=labeler, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    df = load(path)
    save(pd.concat([df, pd.DataFrame([row])], ignore_index=True), path)
    return rid


def update(row_id: str, path: str | Path | None = None, **fields) -> bool:
    df = load(path)
    m = df["id"] == row_id
    if not m.any():
        return False
    for k, v in fields.items():
        if k in COLUMNS and k != "id":
            df.loc[m, k] = str(v).upper() if k == "label" else v
    save(df, path)
    return True


def delete(ids: Iterable[str], path: str | Path | None = None) -> int:
    ids = set(ids)
    df = load(path)
    keep = df[~df["id"].isin(ids)]
    if len(keep) != len(df):
        save(keep, path)
    return len(df) - len(keep)


def for_clip(df: pd.DataFrame, clip_id: str) -> pd.DataFrame:
    return df[df["clip_id"] == clip_id].sort_values(["start_sec", "end_sec"]).reset_index(drop=True)


def mark_clean(clip_id: str, duration: float, *, camera_id: str = "", labeler: str = "", path: str | Path | None = None) -> str:
    """«Весь клип без курения»: NEGATIVE на всю длительность. Прежние NEGATIVE-строки на весь клип не дублируются."""
    df = load(path)
    old = df[(df.clip_id == clip_id) & (df.label == "NEGATIVE") & (df.start_sec <= 0.05) & (df.end_sec >= duration - 0.5)]
    if len(old):
        return str(old["id"].iloc[0])
    return add(clip_id, 0.0, max(float(duration), 0.1), "NEGATIVE", note="весь клип без курения", camera_id=camera_id, labeler=labeler, path=path)


def clip_status(df: pd.DataFrame, clip_ids: Iterable[str]) -> pd.DataFrame:
    """По клипам: число интервалов каждой метки и признак «просмотрен» (есть хотя бы одна строка)."""
    rows = []
    for c in clip_ids:
        g = df[df["clip_id"] == c]
        rows.append(dict(clip_id=c, positive=int((g.label == "POSITIVE").sum()), negative=int((g.label == "NEGATIVE").sum()), ignore=int((g.label == "IGNORE").sum()),
                         reviewed=bool(len(g))))
    return pd.DataFrame(rows, columns=["clip_id", "positive", "negative", "ignore", "reviewed"])


def validate(df: pd.DataFrame) -> list[str]:
    """Предупреждения о разметке, которая тихо ухудшит оценку (человек читает их до запуска)."""
    out = []
    bad = df[~df.label.isin(LABEL_SET)]
    if len(bad):
        out.append(f"{len(bad)} строк с неизвестной меткой: {sorted(bad.label.unique())}")
    inv = df[(df.end_sec <= df.start_sec) | df.start_sec.isna() | df.end_sec.isna()]
    if len(inv):
        out.append(f"{len(inv)} интервалов с некорректным временем")
    pos = df[df.label == "POSITIVE"]
    nobox = pos[~np.isfinite(pos[["x1", "y1", "x2", "y2"]].to_numpy(float)).all(1)]
    if len(nobox):
        out.append(f"{len(nobox)} POSITIVE без рамки: сопоставление идёт по рамке человека, такие эталоны нельзя засчитать (всегда FN)")
    for (clip, person), g in pos.groupby(["clip_id", "person_gt_id"]):
        if not person:
            continue
        g = g.sort_values("start_sec")
        if (g.start_sec.to_numpy()[1:] < g.end_sec.to_numpy()[:-1]).any():
            out.append(f"{clip}, человек {person}: пересекающиеся POSITIVE-интервалы (одно событие нужно размечать одним интервалом)")
    return out


def to_eval_gts(df: pd.DataFrame, clips: set[str] | None = None, box_at: Callable[[str, str, float], tuple | None] | None = None) -> list[GT]:
    """Строки разметки → `evaluate.GT`. Рамка статичная (x1..y2 в peak); `box_at(clip_id, person, t)` может дать траекторию вместо неё."""
    gts = []
    for r in df.itertuples():
        if clips is not None and r.clip_id not in clips:
            continue
        if r.label not in LABEL_SET or not (np.isfinite(r.start_sec) and np.isfinite(r.end_sec)):
            continue
        box = tuple(float(v) for v in (r.x1, r.y1, r.x2, r.y2)) if np.isfinite([r.x1, r.y1, r.x2, r.y2]).all() else None
        fn = (lambda t, c=r.clip_id, p=r.person_gt_id: box_at(c, p, t)) if box_at and r.person_gt_id else None
        gts.append(GT(r.clip_id, str(r.person_gt_id), float(r.start_sec), float(r.end_sec), r.label, box=box, box_at=fn))
    return gts


def export_organizer(df: pd.DataFrame, clips: set[str] | None = None) -> pd.DataFrame:
    d = df if clips is None else df[df.clip_id.isin(clips)]
    d = d.assign(camera_id=d.camera_id.replace("", np.nan).fillna("cam_local"))
    return d[ORGANIZER_COLUMNS].sort_values(["clip_id", "start_sec"]).reset_index(drop=True)


def import_rows(new: pd.DataFrame, path: str | Path | None = None, replace_clips: bool = False) -> int:
    """Добавляет строки из чужого CSV (минимум: clip_id, start_sec, end_sec, label). `replace_clips` — сначала удалить прежние строки этих клипов."""
    need = {"clip_id", "start_sec", "end_sec", "label"}
    if not need <= set(new.columns):
        raise ValueError(f"в CSV нет колонок: {sorted(need - set(new.columns))}")
    n = new.copy()
    for c in COLUMNS:
        if c not in n:
            n[c] = np.nan
    n["label"] = n["label"].astype(str).str.upper().str.strip()
    n = n[n.label.isin(LABEL_SET)]
    n["id"] = [uuid.uuid4().hex[:8] for _ in range(len(n))]
    n["peak_sec"] = n["peak_sec"].fillna((n.start_sec + n.end_sec) / 2)
    n["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    df = load(path)
    if replace_clips:
        df = df[~df.clip_id.isin(set(n.clip_id))]
    save(pd.concat([df, n[COLUMNS]], ignore_index=True), path)
    return len(n)


def fingerprint(df: pd.DataFrame, clips: set[str] | None = None) -> str:
    """Стабильный отпечаток разметки (для журнала экспериментов: по какому эталону получена цифра)."""
    d = df if clips is None else df[df.clip_id.isin(clips)]
    d = d.sort_values(["clip_id", "start_sec", "end_sec", "label"])[["clip_id", "person_gt_id", "start_sec", "end_sec", "label", "x1", "y1", "x2", "y2"]]
    return hashlib.sha1(d.round(3).to_csv(index=False).encode("utf-8")).hexdigest()[:10]
