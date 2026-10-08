"""Страница «Анализ»: править метки жестов, смотреть оценки циклов out-of-fold и подбирать порог цикла, управлять кэшем прогонов.

Разделы:
  Метки жестов   — таблица `docs/review/gesture_gt.csv` (кандидаты-жесты: затяжка, питьё, …): любую строку можно исправить, удалить или добавить; перед записью — копия файла.
  Оценка циклов  — оценки цикла моделью, не видевшей его видео; AUC по классам жестов; ошибки (затяжки с низкой оценкой, ложные с высокой); таблица «порог цикла → F1 события».
  Кэш и прогоны  — кэш событий `outputs/eval_cache`, журнал `outputs/experiments`, готовые запуски `outputs/recognize`: размер, удаление выбранного (повторный прогон пересчитает).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sd import start_eval as SE
from sd.paths import OUTPUTS, ROOT

GT_COLS = ["video", "tid", "peak_t", "start", "label", "start_quality", "note", "source", "reviewer"]
LABELS = list(SE.GESTURE_LABELS) + ["unsure", "no_gesture"]
SCORES = OUTPUTS / "analysis" / "oof_scores.parquet"


def _backup(p: Path) -> Path | None:
    if not p.exists():
        return None
    d = OUTPUTS / "backups"
    d.mkdir(parents=True, exist_ok=True)
    b = d / f"{p.stem}_{time.strftime('%Y%m%d_%H%M%S')}{p.suffix}"
    shutil.copy2(p, b)
    return b


def _save_gestures(df: pd.DataFrame) -> Path | None:
    p = SE.GT_CSV
    b = _backup(p)
    tmp = p.with_suffix(".tmp")
    df[GT_COLS].to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(p)
    return b


def _gestures_section() -> None:
    p = SE.GT_CSV
    df = pd.read_csv(p, encoding="utf-8-sig") if p.exists() else pd.DataFrame(columns=GT_COLS)
    for c in GT_COLS:
        if c not in df:
            df[c] = None
    c1, c2, c3 = st.columns([3, 3, 2])
    vids = sorted(df.video.dropna().unique())
    pick_v = c1.multiselect("Видео", vids, key="an_g_v")
    pick_l = c2.multiselect("Метка", LABELS, key="an_g_l")
    only_low = c3.checkbox("только «неясно»", key="an_g_u")
    view = df
    if pick_v:
        view = view[view.video.isin(pick_v)]
    if pick_l:
        view = view[view.label.isin(pick_l)]
    if only_low:
        view = view[view.label == "unsure"]
    st.caption(f"{p.relative_to(ROOT)} · всего строк {len(df)}, показано {len(view)}. Метки поставлены по монтажам кадров (мои суждения, не эталон): исправляйте то, с чем не согласны.")
    edited = st.data_editor(
        view[GT_COLS], num_rows="dynamic", hide_index=False, use_container_width=True, key="an_g_editor",
        column_config={"label": st.column_config.SelectboxColumn("метка", options=LABELS, required=True),
                       "start_quality": st.column_config.SelectboxColumn("старт", options=["ok", "late", "early"]),
                       "peak_t": st.column_config.NumberColumn("peak_t, с", format="%.2f"), "start": st.column_config.NumberColumn("старт, с", format="%.2f"),
                       "tid": st.column_config.NumberColumn("трек", step=1)})
    if st.button("Сохранить изменения", type="primary", disabled=edited.equals(view[GT_COLS])):
        rest = df.drop(index=view.index)
        out = pd.concat([rest, edited], ignore_index=True)
        bad = out[~out.label.isin(LABELS) | out.video.isna()]
        if len(bad):
            st.error(f"строк с неверной меткой или без видео: {len(bad)}")
        else:
            b = _save_gestures(out.sort_values(["video", "peak_t"], kind="stable"))
            st.success(f"сохранено ({len(out)} строк); копия прежнего файла: {b.relative_to(ROOT) if b else '—'}")
            st.rerun()
    st.caption("Изменили метки — пересчитайте раздел «Оценка циклов» и `sd.cmd start-eval`; эталон событий (страница «Разметка») строится отдельно: `sd.cmd gt-from-gestures`.")


def _cycles_section() -> None:
    from sd import oof_eval as O

    st.caption("Оценка каждого цикла делается моделью, обученной на ДРУГИХ видео (фолды по видео): это честная оценка классификатора, в отличие от пакета, обученного на тех же клипах.")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    sname = c1.selectbox("Набор признаков", ["fast", "kin+pose", "kin", "cheap", "fast+vlm", "cheap+vlm"], key="an_set", help="`cheap`/`vlm` требуют `sd.cmd enrich`")
    enriched = c2.checkbox("с предметом и VLM (`sd.cmd enrich`)", value=sname in ("cheap", "fast+vlm", "cheap+vlm"), key="an_enr")
    repeats = int(c3.number_input("Перемешиваний", 1, 10, 3, key="an_rep"))
    if c4.button("Пересчитать", type="primary"):
        with st.spinner("оцениваем циклы…"):
            clips = O.collect()
            gest = pd.read_csv(SE.GT_CSV, encoding="utf-8-sig")
            tab = O.attach_labels(clips, gest)
            if enriched:
                from sd import enrich as EN

                tab = EN.attach(tab, list(clips))
            tab["score"] = O.oof_scores(tab, sname, repeats=repeats)
            tab.to_parquet(SCORES, index=False)
            st.session_state["an_meta"] = dict(set=sname, enriched=enriched, repeats=repeats, at=time.strftime("%H:%M:%S"))
    if not SCORES.exists():
        st.info("Нажмите «Пересчитать» (на 30 клипах — около минуты).")
        return
    tab = pd.read_parquet(SCORES)
    meta = st.session_state.get("an_meta")
    if meta:
        st.caption(f"последний расчёт: набор {meta['set']}, предмет/VLM {'да' if meta['enriched'] else 'нет'}, перемешиваний {meta['repeats']}, {meta['at']}")
    rep = O.auc_report(tab, tab.score.to_numpy(), n_boot=200)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC циклов (oof)", f"{rep['auc']:.3f}", help=f"95% интервал по видео: {rep['lo']:.2f}–{rep['hi']:.2f}")
    m2.metric("размечено", rep["n"], help=f"затяжек {rep['positives']}, видео {rep['clips']}")
    m3.metric("циклов всего", len(tab))
    m4.metric("без метки", int(tab.y.isna().sum()))
    if rep["per_class"]:
        st.dataframe(pd.DataFrame([dict(класс=k, циклов=v["n"], AUC=round(v["auc"], 3)) for k, v in rep["per_class"].items()]), hide_index=True)
    lab = tab[tab.y.notna()]
    c1, c2 = st.columns(2)
    c1.markdown("**Затяжки с низкой оценкой** (пропуски)")
    c1.dataframe(lab[lab.y == 1].nsmallest(12, "score")[["video", "tid", "peak_t", "score"]].round(2), hide_index=True)
    c2.markdown("**Не затяжки с высокой оценкой** (ложные)")
    c2.dataframe(lab[lab.y == 0].nlargest(12, "score")[["video", "tid", "peak_t", "label", "score"]].round(2), hide_index=True)
    st.markdown("**Порог цикла → событие по регламенту**")
    st.caption("Для каждого порога цикла `events.cycle_th` — лучший F1 по порогу события. Подбор на этих же клипах оптимистичен; окончательный порог фиксируется до скрытого набора.")
    if st.button("Посчитать таблицу порогов"):
        from sd.paths import list_videos, video_id
        from sd.video_io import probe

        clips = O.collect()
        vids = {video_id(v): v for v in list_videos()}
        dur = {k: float(probe(vids[k]).duration) for k in clips}
        with st.spinner("собираем события…"):
            st.session_state["an_grid"] = O.grid(clips, tab.drop(columns=["score"]), tab.score.to_numpy(), dur)
    g = st.session_state.get("an_grid")
    if g is not None:
        st.dataframe(g.round(3), hide_index=True)


def _size(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0


def _cache_section() -> None:
    st.caption("Все прогоны сохраняются на диск. Кэш — это результат, а не настройка: удалите запись, чтобы пересчитать; остальное останется.")
    kinds = {"eval_cache": ("Кэш событий оценки (`outputs/eval_cache`)", OUTPUTS / "eval_cache"), "experiments": ("Журнал оценок (`outputs/experiments`)", OUTPUTS / "experiments"),
             "recognize": ("Запуски распознавания (`outputs/recognize`)", OUTPUTS / "recognize"), "enrich": ("Предмет и VLM по циклам (`outputs/enrich`)", OUTPUTS / "enrich")}
    kind = st.radio("Что смотреть", list(kinds), format_func=lambda k: kinds[k][0], horizontal=True, key="an_cache_kind")
    root = kinds[kind][1]
    if not root.exists():
        st.info("пусто")
        return
    rows = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            rows.append(dict(имя=d.name, МБ=round(_size(d), 1), изменено=time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))))
    df = pd.DataFrame(rows)
    st.write(f"записей {len(df)}, всего {df['МБ'].sum():.0f} МБ" if len(df) else "пусто")
    if df.empty:
        return
    sel = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="multi-row", key=f"an_cache_{kind}")
    idx = sel["selection"]["rows"] if sel else []
    if idx and st.button(f"Удалить выбранные ({len(idx)})"):
        for i in idx:
            shutil.rmtree(root / df.iloc[i]["имя"], ignore_errors=True)
        st.rerun()


def render() -> None:
    sec = st.radio("Раздел", ["Метки жестов", "Оценка циклов", "Кэш и прогоны"], horizontal=True, label_visibility="collapsed", key="an_sec")
    {"Метки жестов": _gestures_section, "Оценка циклов": _cycles_section, "Кэш и прогоны": _cache_section}[sec]()
