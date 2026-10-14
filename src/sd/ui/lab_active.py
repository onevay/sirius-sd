"""Лаборатория → «Активное обучение»: очередь циклов на разметку по стратегиям, разметка прямо здесь, перепроверка подозрительных меток и симуляция стратегий на размеченных данных."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sd import active as AC
from sd import catalog as CAT
from sd import classifier as CL
from sd import gestures as GS
from sd.paths import repo_path
from sd.ui import common as UC
from sd.ui import lab_common as LC

LABEL_BTNS = [("smoke", "💨 затяжка"), ("drink", "🥤 питьё"), ("phone", "📱 телефон"), ("eat", "🍔 еда"), ("touch_face", "🖐 лицо"), ("other_neg", "другой жест"),
              ("no_gesture", "нет жеста"), ("unsure", "❓ неясно")]


def _spec(key: str) -> dict | None:
    bundles = [c.id for c in CAT.cycle_bundles()]
    names = CL.list_specs()
    opts = [f"пакет: {b}" for b in bundles] + [f"спецификация: {n}" for n in names]
    if not opts:
        st.info("Нет ни пакетов, ни спецификаций классификатора.")
        return None
    default = next((i for i, o in enumerate(opts) if o.endswith("cycle_mvp")), 0)
    sel = st.selectbox("Модель, которая выбирает циклы", opts, index=default, key=key)
    return AC.spec_from_bundle(repo_path(sel.split(": ", 1)[1])) if sel.startswith("пакет") else CL.load_spec(sel.split(": ", 1)[1])


def _queue_tab() -> None:
    st.markdown("Модель обучается на размеченных циклах, оценивает **неразмеченные** и предлагает, какие разметить первыми. "
                "Метка записывается в таблицу жестов (с сохранением версии) и сразу учитывается при следующем обучении.")
    c = st.columns(2)
    with c[0]:
        pool_run = LC.pick_run("Откуда брать циклы для разметки (прогон оценки)", "la_pool", help="например, прогон по новой папке с видео")
    with c[1]:
        train_run = LC.pick_run("На чём учить выбирающую модель", "la_train", help="прогон с размеченными циклами; может совпадать с предыдущим")
    spec = _spec("la_model")
    if not pool_run or not train_run or spec is None:
        return
    stamp = LC.gestures_stamp()
    pool_tab = LC.table_for(pool_run, stamp)
    train_tab = pool_tab if train_run == pool_run else LC.table_for(train_run, stamp)
    train_lab = train_tab[train_tab.y.notna()]
    unl = pool_tab[pool_tab.label.isna()].reset_index(drop=True)
    feats = CL.spec_features(spec)
    miss = [f for f in feats if f not in pool_tab.columns]
    m = st.columns(4)
    m[0].metric("Циклов в пуле", len(pool_tab))
    m[1].metric("Без метки", len(unl))
    m[2].metric("Размечено для обучения", len(train_lab))
    m[3].metric("Размечено в этой сессии", st.session_state.get("la_count", 0))
    if miss:
        st.error(f"в таблице пула нет признаков модели: {', '.join(miss)}. Посчитайте «Задачи → Признаки» для прогона {pool_run}.")
        return
    cov = LC.coverage(unl, feats) if len(unl) else pd.DataFrame()
    if len(cov) and (cov.заполнено < 0.6).any():
        st.warning("У неразмеченных циклов признаки заполнены не полностью (" + ", ".join(cov[cov.заполнено < 0.6].признак) + "): оценки приблизительные. Посчитайте признаки предмета/фото для прогона.")
        UC.task_button("enrich", dict(run=pool_run, objects="smoking_yolo11m_beehzod", photo_bundle="models/photo/photo_v2", vlm="", clips=""), "Посчитать признаки пула (фоном)", key="la_enrich", primary=False)
    mode = st.radio("Что показывать", ["Новые циклы (без метки)", "Перепроверка меток (подозрительные)"], horizontal=True, key="la_mode")
    if mode.startswith("Перепроверка"):
        _recheck(pool_tab, spec, pool_run)
        return
    if unl.empty:
        st.success("В этом прогоне все циклы уже размечены. Для активного обучения нужны новые данные: выполните оценку по папке с неразмеченными видео («Валидация → Запуск»), затем выберите её прогон здесь.")
        return
    if len(train_lab) < 30 or train_lab.y.nunique() < 2:
        st.error("Для выбирающей модели нужно ≥ 30 размеченных циклов обоих классов.")
        return
    cc = st.columns([3, 2, 2], vertical_alignment="bottom")
    strat = cc[0].selectbox("Стратегия", list(AC.STRATEGIES), format_func=lambda k: f"{k} — {AC.STRATEGIES[k]}", index=1, key="la_strat")
    k = cc[1].slider("Размер очереди", 3, 40, 12, key="la_k")
    th = cc[2].slider("Порог цикла", 0.1, 0.9, 0.4, 0.05, key="la_th", help="для стратегии «около порога» (events.cycle_th профиля)")
    cm = AC.committee(train_lab, unl, spec)
    pos = AC.priority(cm, strat, threshold=th, space=AC._feature_space(unl, feats), k=int(k), rng=np.random.default_rng(int(st.session_state.get("la_count", 0))))
    q = unl.iloc[pos].reset_index(drop=True)
    q = q.assign(оценка=cm.iloc[pos]["p"].to_numpy(), разброс=cm.iloc[pos]["std"].to_numpy())
    _label_panel(q, pool_run)


def _label_panel(q: pd.DataFrame, run: str) -> None:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Очередь**")
        show = q[["video", "peak_t", "оценка", "разброс", "hold"]].rename(columns=dict(video="клип", peak_t="пик, с", hold="пауза")).round(2)
        st.dataframe(show, hide_index=True, width="stretch", key="la_tbl", on_select="rerun", selection_mode="single-row", height=420)
    sel = st.session_state.get("la_tbl", {}).get("selection", {}).get("rows", [])
    i = sel[0] if sel else 0
    row = q.iloc[min(i, len(q) - 1)]
    with right:
        st.markdown(f"**{row.video}** · пик {row.peak_t:.1f} с · ID {int(row.tid)} · оценка модели {row['оценка']:.2f} (разброс {row['разброс']:.2f})")
        img = LC.preview(run, row)
        if img is not None:
            st.image(img, width="stretch")
        else:
            st.caption("кадры недоступны (нет видео или кэша позы)")
        near = GS.near(GS.load(), row.video, float(row.peak_t), 2.0)
        if len(near):
            st.caption("Рядом уже есть метки: " + ", ".join(f"{GS.LABEL_RU.get(l, l)} ({t:.1f} с)" for l, t in zip(near.label, near.peak_t)))
        _buttons(row, "q")


def _buttons(row: pd.Series, key: str) -> None:
    cols = st.columns(4)
    for n, (lab, text) in enumerate(LABEL_BTNS):
        if cols[n % 4].button(text, key=f"la_{key}_{lab}", width="stretch"):
            try:
                msg = LC.add_gesture(row, lab)
                st.session_state["la_count"] = st.session_state.get("la_count", 0) + 1
                st.toast(msg, icon="✅")
                st.session_state.pop("la_tbl", None)
                st.rerun()
            except Exception as e:      # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")
    st.caption("Метка пишется в `labels/review/gesture_gt.csv`; перед записью сохраняется версия (откат — «Данные → Версии разметки»). «Пропустить» — просто выберите другую строку.")


def _recheck(tab: pd.DataFrame, spec: dict, run: str) -> None:
    n = st.slider("Сколько показать", 5, 40, 15, key="la_sus_n")
    if tab[tab.y.notna()].y.nunique() < 2:
        st.info("Нет размеченных циклов.")
        return
    sus = AC.suspects(tab, spec, int(n))
    st.caption("Размеченные циклы, где out-of-fold оценка сильно расходится с меткой. Это либо ошибка модели, либо ошибка разметки — посмотрите кадры и при необходимости исправьте метку (новая запись имеет приоритет рядом по времени).")
    show = sus[["video", "peak_t", "label", "y", "oof", "расхождение"]].rename(columns=dict(video="клип", peak_t="пик, с", label="метка")).round(2)
    c = st.columns([2, 3], gap="large")
    with c[0]:
        st.dataframe(show, hide_index=True, width="stretch", key="la_sus", on_select="rerun", selection_mode="single-row", height=420)
    sel = st.session_state.get("la_sus", {}).get("selection", {}).get("rows", [])
    row = sus.iloc[sel[0] if sel else 0]
    with c[1]:
        st.markdown(f"**{row.video}** · пик {row.peak_t:.1f} с · метка **{GS.LABEL_RU.get(row.label, row.label)}** · модель говорит {row.oof:.2f}")
        img = LC.preview(run, row)
        if img is not None:
            st.image(img, width="stretch")
        _buttons(row, "sus")


def _simulate_tab() -> None:
    st.markdown("Метки скрываются и «покупаются» по очереди разными стратегиями; качество измеряется на отложенных видео. Так видно, какая стратегия и какой размер пакета "
                "дают лучшую кривую обучения на ваших данных.")
    run = LC.pick_run("Размеченные данные (прогон оценки)", "las_run")
    spec = _spec("las_model")
    if not run or spec is None:
        return
    tab = LC.table_for(run, LC.gestures_stamp())
    lt = tab[tab.y.notna()]
    if len(lt) < 60 or lt.y.nunique() < 2:
        st.error("Для симуляции нужно ≥ 60 размеченных циклов.")
        return
    c = st.columns(5)
    strategies = st.multiselect("Стратегии", list(AC.STRATEGIES), default=["random", "uncertainty", "committee", "diverse"], key="las_str")
    init = c[0].slider("Стартовых меток", 8, 40, 16, key="las_init")
    batch = c[1].slider("Размер пакета", 2, 20, 8, key="las_batch")
    rounds = c[2].slider("Раундов", 2, 12, 6, key="las_rounds")
    folds = c[3].slider("Отложенных частей", 2, 6, 4, key="las_folds")
    scene = c[4].toggle("По сценам", value=True, key="las_scene")
    if st.button("Запустить симуляцию", type="primary", key="las_go", disabled=not strategies):
        bar = st.progress(0.0, text="старт…")
        res = AC.simulate(tab, spec, strategies, init=int(init), batch=int(batch), rounds=int(rounds), folds=int(folds), by_scene=scene,
                          progress=lambda i, n, msg: bar.progress(min(i / n, 1.0), text=msg))
        bar.empty()
        st.session_state["las_res"] = dict(res=res, run=run)
    r = st.session_state.get("las_res")
    if not r or r["run"] != run:
        return
    res = r["res"]
    if res.empty:
        st.warning("Не получилось: в отложенных частях нет обоих классов.")
        return
    import plotly.express as px

    fig = px.line(res, x="labels", y="auc", color="strategy", markers=True, error_y="std", height=360, labels=dict(labels="размечено циклов", auc="AUC на отложенных", strategy="стратегия"))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch", key="las_fig")
    last = res[res["round"] == res["round"].max()].sort_values("auc", ascending=False)
    area = res.groupby("strategy").auc.mean().sort_values(ascending=False)
    st.dataframe(last[["strategy", "labels", "auc", "std", "n"]].round(3), hide_index=True, width="stretch")
    best, base = area.index[0], area.get("random", np.nan)
    st.info(f"По площади под кривой лучшая — **{best}** ({area.iloc[0]:.3f}; случайный отбор {base:.3f}). " + (
        "Разница меньше разброса между частями — различия стратегий на этих данных не доказаны." if np.isfinite(base) and (area.iloc[0] - base) < float(res["std"].mean(skipna=True) or 0.05)
        else "Выигрыш больше разброса — стратегия стоит применения."))


def render() -> None:
    t1, t2 = st.tabs(["Очередь на разметку", "Симуляция стратегий"])
    with t1:
        _queue_tab()
    with t2:
        _simulate_tab()
