"""Метки жестов (кандидаты: затяжка, питьё, телефон, касание лица, еда, нет жеста, неясно) — чтение, проверка и запись с версией.

Файл — `paths.GESTURES_CSV` (по умолчанию docs/review/gesture_gt.csv). Из него строятся таблица обучения классификатора (`oof_eval.attach_labels`) и эталон событий
(`sd gt-from-gestures`). Раньше правка шла из страницы интерфейса с копией в outputs/backups; теперь запись проходит проверку и сохраняет прежнее содержимое
версией (labels/history), откат — на вкладке «Данные → Версии разметки».
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import labels_history as LH
from . import start_eval as SE

COLS = ["video", "tid", "peak_t", "start", "label", "start_quality", "note", "source", "reviewer"]
LABELS = list(SE.GESTURE_LABELS) + ["unsure", "no_gesture"]
LABEL_RU = {"smoke": "затяжка", "drink": "питьё", "phone": "телефон", "eat": "еда", "touch_face": "касание лица", "other_neg": "другой жест", "unsure": "неясно",
            "no_gesture": "нет жеста"}


def path() -> Path:
    return Path(SE.GT_CSV)


def load() -> pd.DataFrame:
    p = path()
    df = pd.read_csv(p, encoding="utf-8-sig") if p.exists() else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df:
            df[c] = None
    return df[COLS]


def validate(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(уровень, текст): ошибки не дают записать файл, предупреждения — показываются."""
    out: list[tuple[str, str]] = []
    bad = df[~df.label.isin(LABELS)]
    if len(bad):
        out.append(("error", f"{len(bad)} строк с неизвестной меткой ({sorted(map(str, bad.label.unique()))[:5]}); допустимо: {', '.join(LABELS)}"))
    nov = df[df.video.isna() | (df.video.astype(str).str.strip() == "")]
    if len(nov):
        out.append(("error", f"{len(nov)} строк без видео"))
    t = pd.to_numeric(df.peak_t, errors="coerce")
    if t.isna().any() or (t < 0).any():
        out.append(("error", f"{int(t.isna().sum() + (t < 0).sum())} строк без момента пика (peak_t) или с отрицательным"))
    s = pd.to_numeric(df.start, errors="coerce")
    late = (s > t + 1e-6) & s.notna() & t.notna()
    if late.any():
        out.append(("warning", f"{int(late.sum())} строк: начало позже пика"))
    d = df.assign(_t=t, _tid=df.tid.astype(str)).dropna(subset=["_t"]).sort_values(["video", "_tid", "_t"])
    close = d.groupby(["video", "_tid"])._t.diff().abs() < 0.3
    if close.any():
        out.append(("warning", f"{int(close.sum())} пар меток одного человека ближе 0.3 с — возможен дубль (одна затяжка размечена дважды)"))
    return out


def _atomic_write(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        for attempt in range(8):
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save(df: pd.DataFrame, author: str = "", note: str = "") -> dict | None:
    """Проверить и записать. Ошибки проверки — ValueError (файл не трогается). Прежнее содержимое сохраняется версией; возвращает запись о ней (или None, если файла не было)."""
    errs = [t for lvl, t in validate(df) if lvl == "error"]
    if errs:
        raise ValueError("; ".join(errs))
    snap = LH.snapshot("gestures", author, note or "перед правкой таблицы жестов", "table-save", force=False)
    out = df[COLS].copy()
    out["peak_t"] = pd.to_numeric(out.peak_t, errors="coerce")
    out = out.sort_values(["video", "peak_t"], kind="stable")
    _atomic_write(out, path())
    return snap


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """По видео: число окон каждой метки."""
    if df.empty:
        return pd.DataFrame()
    t = df.pivot_table(index="video", columns="label", values="peak_t", aggfunc="count", fill_value=0)
    t["всего"] = t.sum(axis=1)
    return t.rename(columns=LABEL_RU).reset_index().rename(columns={"video": "клип"})


def near(df: pd.DataFrame, video: str, t: float, tol: float = 1.5) -> pd.DataFrame:
    """Метки жестов рядом с моментом t (для перехода «ошибка → разметка»)."""
    tt = pd.to_numeric(df.peak_t, errors="coerce")
    return df[(df.video == video) & ((tt - t).abs() <= tol)] if np.isfinite(t) else df.iloc[0:0]
