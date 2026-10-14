"""Общее для вкладки «Лаборатория»: кэшируемые загрузчики таблицы циклов, треков и превью цикла, выбор источника, метки жестов из интерфейса."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sd import classifier as CL
from sd import gestures as GS
from sd import oof_eval as O
from sd import start_eval as SE
from sd.ui import common as UC


def gestures_stamp() -> float:
    return UC.file_mtime(SE.GT_CSV)


@st.cache_data(show_spinner="Собираем таблицу циклов…", max_entries=6)
def table_for(run: str, stamp: float, enriched: bool = True) -> pd.DataFrame:
    """Таблица циклов прогона с метками жестов и признаками enrich; `stamp` — время изменения меток (после разметки таблица пересобирается)."""
    return CL.load_table(enriched=enriched, run=run)


@st.cache_data(show_spinner=False, max_entries=6)
def clip_dirs(run: str) -> dict[str, str]:
    """{видео: каталог запуска позы} для прогона."""
    return {k: str(v.run_dir) for k, v in O.collect_from_run(run).items()}


@st.cache_resource(show_spinner=False, max_entries=8)
def tracks_for(rd: str):
    from sd.tracks import Tracks

    return Tracks.load(Path(rd) / "pose")


def pick_run(label: str, key: str, help: str | None = None) -> str | None:
    """Выбор прогона оценки, из которого можно собрать таблицу циклов (есть запуски по полным клипам)."""
    rows = O.runs_with_tables()
    if not rows:
        st.info("Нет прогонов с готовыми запусками по полным клипам: выполните оценку («Валидация → Запуск»).")
        return None
    ids = [r["id"] for r in rows]
    lab = {r["id"]: f"{r['created']} · {r['name']} · {r['clips']} клипов · {r.get('fps') or '—'} к/с · {r.get('cycle_model') or '—'}" for r in rows}
    return st.selectbox(label, ids, key=key, format_func=lambda i: lab[i], help=help)


def preview(run: str, row: pd.Series, size: int = 170):
    """Полоса кропов человека по фазам цикла (RGB) или None."""
    from sd import lab

    try:
        rd = clip_dirs(run).get(row["video"])
        if not rd:
            return None
        tr = tracks_for(rd)
        video = tr.meta.get("video")
        cyc = pd.Series(dict(tid=int(row["tid"]), start=float(row["start"]), mouth_in=float(row.get("mouth_in", row["start"])), peak_t=float(row["peak_t"]),
                             mouth_out=float(row.get("mouth_out", row["peak_t"])), end=float(row.get("end", row["peak_t"] + 1.0))))
        for c in ("mouth_in", "mouth_out", "end"):
            if not np.isfinite(cyc[c]):
                cyc[c] = cyc["peak_t"]
        return lab.cycle_strip(video, tr, cyc, size)
    except Exception:       # noqa: BLE001 — превью вспомогательное: нет видео/кэша — просто без картинки
        return None


def coverage(tab: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Заполненность признаков: колонка, доля заполненных циклов."""
    return pd.DataFrame([dict(признак=c, заполнено=round(float(tab[c].notna().mean()), 3) if c in tab else 0.0) for c in feats]).sort_values("заполнено", ignore_index=True)


def add_gesture(row: pd.Series, label: str, note: str = "") -> str:
    """Записывает метку жеста для цикла (video, tid, peak_t, start) в таблицу жестов; прежняя версия сохраняется (labels/history). Возвращает сообщение."""
    df = GS.load()
    new = dict(video=row["video"], tid=int(row["tid"]), peak_t=round(float(row["peak_t"]), 3), start=round(float(row["start"]), 3), label=label, start_quality=None,
               note=note or "активное обучение", source="active", reviewer=UC.who() or "lab")
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    snap = GS.save(df, UC.who(), f"лаборатория: метка {label} ({row['video']} {row['peak_t']:.1f} с)")
    return f"записано: {GS.LABEL_RU.get(label, label)} · {row['video']} · {row['peak_t']:.1f} с" + (" (версия сохранена)" if snap else "")


def attempts() -> list[dict]:
    """Попытки лаборатории в этой сессии (спецификация, метрика) — для сравнения."""
    return st.session_state.setdefault("lab_attempts", [])


def short_time() -> str:
    return time.strftime("%H:%M:%S")
