"""Вкладка «Лаборатория»: каждая часть конвейера отдельно, обучение моделей и активное обучение на реальных данных.

Разделы:
  Этапы на клипе      — поза → ряд d_t → циклы → оценка цикла → события на одном клипе по кэшу позы: меняете ручки этапа и сразу видите результат, сравнение с исходной настройкой
  Временные ряды      — модели по форме рядов (статистики, случайные свёртки, 1D-сеть): обучение, честная оценка, смесь с табличным классификатором
  Классификатор       — быстрый подбор признаков и членов ансамбля с кросс-валидацией по видео/сценам: ROC, PR, калибровка, ошибки, попытки, сохранение спецификации, обучение пакета
  Дообучение          — переобучение пакета на новых данных с повышенным весом новых и трудных циклов; честное сравнение «с весами / без»
  Активное обучение   — очередь циклов на разметку по стратегиям, разметка прямо здесь, симуляция стратегий на размеченных данных
Ничего из этого не пишет в журнал оценок и эталон событий; метки жестов (раздел «Активное обучение») сохраняются версией. Описание — docs/DEV_STUDIO.md, раздел 8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sd import lab
from sd import profiles as PR
from sd.ui import common as UC
from sd.ui import lab_active, lab_clf, lab_finetune, lab_ts

SECTIONS = ["Этапы на клипе", "Временные ряды", "Классификатор", "Дообучение", "Активное обучение"]
STAGE_TITLES = {"series": "Ряд d_t", "cycles": "Автомат циклов", "events": "Правила события"}


# ---------------------------------------------------------------------------------------------- выбор клипа
@st.cache_data(ttl=30, show_spinner=False)
def _clips() -> pd.DataFrame:
    return lab.list_clips()


@st.cache_resource(show_spinner=False, max_entries=6)
def _tracks(rd: str):
    return lab.load_clip(rd)[0]


def _pick_clip() -> tuple[pd.Series | None, str]:
    df = _clips()
    if df.empty:
        st.info("Кэша позы нет: выполните оценку («Валидация → Запуск») — тогда здесь появятся клипы.")
        return None, ""
    from sd import gt as GT

    gtv = set(GT.load().clip_id) if len(GT.load()) else set()
    c = st.columns([3, 4, 2, 2], vertical_alignment="bottom")
    only = c[3].toggle("только размеченные", value=True, key="lab_only_gt", help="клипы, для которых есть эталон событий (нужен для проверки результата)")
    d = df[df.video_id.isin(gtv)] if only else df
    d = d[~d.video_id.str.startswith("hmdb51")] if only else d
    vids = sorted(d.video_id.unique())
    if not vids:
        st.info("Нет клипов под фильтр.")
        return None, ""
    vid = c[0].selectbox("Клип", vids, key="lab_clip")
    runs = d[d.video_id == vid].sort_values("run_id", ascending=False)
    full = runs[runs.run_id.str.startswith("0-ends")]
    runs = pd.concat([full, runs.drop(full.index)])
    lab_ = {r.run_id: f"{r.start:g}–{r.end:g} с · {r.fps:g} к/с · {r.pose} · {r.created}" for r in runs.itertuples()}
    rid = c[1].selectbox("Запуск позы (из кэша)", runs.run_id.tolist(), key="lab_run", format_func=lambda i: lab_[i])
    profs = PR.list_profiles()
    prof = c[2].selectbox("Профиль", profs, index=profs.index("mvp") if "mvp" in profs else 0, key="lab_profile") if profs else ""
    return runs[runs.run_id == rid].iloc[0], prof


# ---------------------------------------------------------------------------------------------- ручки
def _knobs(cfg_base) -> dict:
    """Ползунки параметров этапов; возвращает только изменённые значения {ключ: значение}."""
    over: dict = {}
    by_stage: dict[str, list[str]] = {}
    for k, (stg, *_rest) in lab.STAGE_KNOBS.items():
        by_stage.setdefault(stg, []).append(k)
    for stg, keys in by_stage.items():
        with st.expander(STAGE_TITLES[stg], expanded=stg == "cycles"):
            for k in keys:
                _, label, lo, hi, step = lab.STAGE_KNOBS[k]
                base = lab.knob_value(cfg_base, k)
                v = st.slider(label, float(lo), float(hi), float(min(max(base, lo), hi)), float(step), key=f"lab_k_{k}", help=f"{k}; в профиле: {base:g}")
                if abs(v - base) > 1e-9:
                    over[k] = type(base)(v) if isinstance(base, int) else float(v)
    if over:
        st.caption(f"изменено: {len(over)} — " + ", ".join(f"`{k}`" for k in over))
        if st.button("Сбросить ручки", key="lab_reset"):
            for k in lab.STAGE_KNOBS:
                st.session_state.pop(f"lab_k_{k}", None)
            st.rerun()
    return over


# ---------------------------------------------------------------------------------------------- вкладки этапов
def _pose_tab(tr, rd, cfg, video, vid) -> None:
    s = tr.summary(float(cfg["video"]["min_person_height_px"]))
    c = st.columns(4)
    c[0].metric("Людей (треков)", len(s))
    c[1].metric("Кадров обработано", int(len(tr.frame_t)))
    c[2].metric("Модель позы", str(tr.meta.get("pose", {}).get("weights") or tr.meta.get("pose", {}).get("model") or "—")[:24])
    fps = tr.meta.get("fps_steady")
    c[3].metric("Скорость позы, к/с", f"{fps:.1f}" if fps else "—")
    st.dataframe(s.rename(columns=dict(tid="ID", t0="с", t1="по", n_det="детекций", h_med="рост медиана, px", h_min="мин", h_max="макс", coverage="покрытие", ignore_small="IGNORE (мелкий)"))
                 .round(2), hide_index=True, width="stretch")
    dur = float(tr.frame_t.t.max()) if len(tr.frame_t) else 1.0
    t = st.slider("Момент, с", 0.0, dur, min(dur / 2, 10.0), 0.1, key="lab_pose_t")
    if video:
        try:
            img, tt = lab.pose_frame(video, tr, cfg, t)
            st.image(img, caption=f"кадр {tt:.2f} с: рамки, ID, ключевые точки (цвет = уверенность)", width="stretch")
        except Exception as e:      # noqa: BLE001
            st.warning(f"кадр не показан: {e}")
    st.caption("Позу пересчитывает «Конструктор → Оценить» (вход, модель, трекер); здесь используется кэш — этапы ниже пересчитываются за секунды.")


def _series_tab(fr, fr_base, cfg, vid) -> None:
    ser, cyc, rej = fr["series"], fr["cycles"], fr["rejected"]
    if ser.empty or "tid" not in ser:
        st.info("В запуске нет людей.")
        return
    cnt = ser.groupby("tid").size()
    ncyc = cyc.groupby("tid").size() if len(cyc) else pd.Series(dtype=int)
    tids = [int(t) for t in cnt.sort_values(ascending=False).index]
    tid = st.selectbox("Человек (ID трека)", tids, key="lab_tid", format_func=lambda t: f"ID {t} · {cnt[t]} кадров · циклов {int(ncyc.get(t, 0))}")
    from sd import gestures as GS

    st.plotly_chart(lab.dt_figure(ser, tid, cyc, rej, cfg, GS.load(), vid), width="stretch", key="lab_dt")
    a, b = len(fr_base["cycles"]), len(cyc)
    c = st.columns(3)
    c[0].metric("Циклов найдено", b, delta=b - a if a != b else None, help="по сравнению с исходной настройкой профиля")
    c[1].metric("Отброшено жестов", len(rej), delta=len(rej) - len(fr_base["rejected"]) if len(rej) != len(fr_base["rejected"]) else None)
    c[2].metric("Кадров с рукой у рта", int((ser.d < float(cfg["cycles"]["th_in"])).sum()))
    st.caption("Зелёные полосы — найденные циклы, серые — отброшенные (причина подписана), ромбы — метки жестов. Пороги автомата — пунктир; двигайте их слева.")


def _cycles_tab(fr, fr_base, tr, vid, video) -> None:
    from sd import gestures as GS

    cyc, rej = fr["cycles"], fr["rejected"]
    c = st.columns([3, 2])
    with c[0]:
        if cyc.empty:
            st.info("Циклов нет при этих порогах.")
        else:
            g = GS.load()
            show = cyc.copy()
            show["рука"] = show.hand.map({0: "левая", 1: "правая"})
            show["метка жеста"] = [", ".join(GS.near(g, vid, float(p), 1.0).label.map(lambda x: GS.LABEL_RU.get(x, x))) or "—" for p in show.peak_t]
            show["пауза"] = show.hold.round(2)
            st.dataframe(show[["tid", "start", "peak_t", "end", "пауза", "d_min", "рука", "метка жеста"]].round(2), hide_index=True, width="stretch", key="lab_cyc_tbl",
                         on_select="rerun", selection_mode="single-row")
    with c[1]:
        st.markdown("**Почему жесты отброшены**")
        if rej.empty:
            st.caption("нет")
        else:
            st.dataframe(rej.groupby("reason").size().rename("штук").reset_index().rename(columns={"reason": "причина"}), hide_index=True, width="stretch")
    sel = st.session_state.get("lab_cyc_tbl", {}).get("selection", {}).get("rows", []) if cyc is not None and len(cyc) else []
    if sel and video:
        r = cyc.iloc[sel[0]]
        img = lab.cycle_strip(video, tr, r)
        if img is not None:
            st.image(img, caption=f"цикл ID {int(r.tid)}: до · старт · рот вх. · пик · рот вых. · после", width="stretch")
    else:
        st.caption("Выберите цикл в таблице — покажем кадры по фазам.")
    a, b = {(int(t), round(float(s), 1)) for t, s in zip(fr_base["cycles"].tid, fr_base["cycles"].start)}, {(int(t), round(float(s), 1)) for t, s in zip(cyc.tid, cyc.start)}
    if a != b:
        st.info(f"Относительно исходной настройки: появилось {len(b - a)}, пропало {len(a - b)} циклов.")


def _score_key(rid, prof, bundle, light, over) -> str:
    return f"{rid}|{prof}|{bundle}|{light}|{sorted(over.items())}"


def _scoring(rd, cfg, cfg_base, prof, vid, over, tr, ser, cyc, rej) -> dict | None:
    """Оценка циклов классификатором (дорогая часть: признаки предмета/фото считаются по новым циклам) и кэш результата в сессии."""
    from sd import catalog as CAT

    bundles = [c.id for c in CAT.cycle_bundles()]
    pro = PR.load_named(prof)
    cur = pro.opts().get("cycle_bundle")
    c = st.columns([4, 3, 2], vertical_alignment="bottom")
    opts = ["как в профиле"] + bundles
    sel = c[0].selectbox("Классификатор цикла", opts, key="lab_bundle", help=f"в профиле: {cur}")
    bundle = None if sel == "как в профиле" else sel
    light = c[1].toggle("без предмета/фото", value=False, key="lab_light", help="быстро, но пакетам с признаками предмета и фото не хватит признаков — оценки приблизительные")
    n = len(cyc)
    run = c[2].button("Посчитать", type="primary", key="lab_score_btn", help=f"оценка циклов и события; циклов: {n}; признаки предмета ≈ 2 с на новый цикл")
    store = st.session_state.setdefault("lab_scored", {})
    key = _score_key(rd.name, prof, bundle, light, over)
    if run:
        try:
            with st.spinner("Считаем признаки циклов и оценку классификатора…"):
                res_new = lab.recognize_clip(rd, cfg, prof, bundle=bundle, light=light)
                store[key] = res_new
                bkey = _score_key(rd.name, prof, bundle, light, {})
                if bkey not in store:
                    store[bkey] = lab.recognize_clip(rd, cfg_base, prof, bundle=bundle, light=light) if over else res_new
        except Exception as e:      # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")
    return store.get(key)


def _score_tab(res, cyc, vid) -> None:
    if res is None:
        st.info("Нажмите «Посчитать» — результат запоминается на время сессии, ручки событий потом работают мгновенно.")
        return
    sc = res["scored"]
    if sc is None or sc.empty or "score" not in sc:
        st.warning("Оценок нет (циклов нет).")
        return
    for w in res["meta"].get("warnings", []):
        st.warning(w)
    from sd import gestures as GS
    from sd import oof_eval as O

    g = GS.load()
    lab_ = O.match_gestures(sc[["video", "tid", "peak_t"]].assign(video=vid), g)
    show = sc.assign(метка=[GS.LABEL_RU.get(x, "—") if x else "—" for x in lab_.label])
    cols = [c for c in ("tid", "start", "peak_t", "hold", "score", "score_cheap", "obj_any_max_conf", "photo_zsd_max", "метка") if c in show.columns]
    st.dataframe(show[cols].round(3), hide_index=True, width="stretch")
    import plotly.express as px

    fig = px.bar(show, x="peak_t", y="score", color="метка", hover_data=["tid"], height=260)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch", key="lab_score_bar")
    y = lab_.label.map(lambda x: 1.0 if x == "smoke" else (0.0 if x in O.NEG_LABELS else np.nan))
    ok = y.notna().to_numpy()
    if ok.sum() >= 4 and len(np.unique(y[ok])) == 2:
        from sd.feature_auc import _auc

        st.metric("AUC оценки на размеченных циклах клипа", f"{_auc(y[ok].to_numpy(int), sc.score.to_numpy(float)[ok]):.3f}", help=f"циклов с меткой: {int(ok.sum())}; на одном клипе число шумное")


def _events_tab(res, res_base, tr, fr, cfg, vid, dur) -> None:
    from sd import stages

    if res is None:
        st.info("Сначала посчитайте оценку циклов (вкладка «Оценка цикла»).")
        return
    sc = res["scored"]
    scores = {(int(r.tid), round(float(r.start), 3)): float(r.score) for r in sc.itertuples() if hasattr(r, "score") and np.isfinite(r.score)} if len(sc) and "score" in sc else {}
    ev, _ = stages.assemble_events(tr, fr["series"], fr["cycles"], cfg, "lab", vid, cycle_scores=scores or None)
    th = float(cfg["events"]["confidence_threshold"])
    kept = ev[ev.confidence >= th] if len(ev) else ev
    ce = lab.clip_eval(kept, vid, dur, th)
    m = ce["metrics"]
    c = st.columns(5)
    c[0].metric("Событий системы", len(kept))
    if ce["has_gt"]:
        c[1].metric("F1 клипа", f"{m['f1']:.2f}")
        c[2].metric("TP / FP / FN", f"{m['tp']} / {m['fp']} / {m['fn']}")
        c[3].metric("Точность", f"{m['precision']:.2f}")
        c[4].metric("Полнота", f"{m['recall']:.2f}")
    else:
        st.caption("Для клипа нет эталона событий — метрики не считаются.")
    gt = ce["gt"]
    rows = [("система", ce["events"].rename(columns={}), "#dc2626")] if len(ce["events"]) else [("система", pd.DataFrame(columns=["start", "end"]), "#dc2626")]
    if len(gt):
        pos = gt[gt.label == "POSITIVE"].rename(columns={"start_sec": "start", "end_sec": "end"})
        ign = gt[gt.label != "POSITIVE"].rename(columns={"start_sec": "start", "end_sec": "end"})
        rows = [("эталон: курение", pos, "#16a34a"), ("эталон: игнор/отрицательные", ign, "#9ca3af"), *rows]
    st.plotly_chart(lab.events_timeline(dur, rows), width="stretch", key="lab_ev_tl")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**События системы**")
        st.dataframe(ce["events"].round(2), hide_index=True, width="stretch")
    with t2:
        st.markdown("**Пропущено (эталон)**")
        st.dataframe(ce["missed"].round(2), hide_index=True, width="stretch")
    if res_base is not None and res_base is not res:
        scb = res_base["scored"]
        sb = {(int(r.tid), round(float(r.start), 3)): float(r.score) for r in scb.itertuples() if hasattr(r, "score") and np.isfinite(r.score)} if len(scb) and "score" in scb else {}
        cfg_b = lab.build_cfg(lab_rd(), st.session_state.get("lab_profile"))
        evb, _ = stages.assemble_events(tr, lab.stage_frames(tr, lab_rd(), cfg_b)["series"], lab.stage_frames(tr, lab_rd(), cfg_b)["cycles"], cfg_b, "lab", vid, cycle_scores=sb or None)
        evb = evb[evb.confidence >= float(cfg_b["events"]["confidence_threshold"])] if len(evb) else evb
        d = lab.diff_events(evb, kept)
        if len(d) and (d.статус != "осталось").any():
            st.markdown("**Что изменилось относительно исходной настройки**")
            st.dataframe(d.round(2), hide_index=True, width="stretch")
        else:
            st.caption("События совпадают с исходной настройкой.")


def lab_rd():
    from pathlib import Path

    return Path(st.session_state["lab_rd"])


def _stages() -> None:
    row, prof = _pick_clip()
    if row is None or not prof:
        return
    rd = row.path
    st.session_state["lab_rd"] = rd
    tr = _tracks(rd)
    video = tr.meta.get("video")
    from pathlib import Path

    rdp = Path(rd)
    cfg_base = lab.build_cfg(rdp, prof)
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.markdown("**Ручки этапов**")
        over = _knobs(cfg_base)
    cfg = lab.build_cfg(rdp, prof, over)
    try:
        fr = lab.stage_frames(tr, rdp, cfg)
        fr_base = lab.stage_frames(tr, rdp, cfg_base) if over else fr
    except Exception as e:      # noqa: BLE001
        right.error(f"{type(e).__name__}: {e}")
        return
    dur = float(row.end) if row.end else float(tr.frame_t.t.max())
    with right:
        t_pose, t_ser, t_cyc, t_score, t_ev = st.tabs(["1 · Поза и трек", "2 · Ряд d_t", "3 · Циклы", "4 · Оценка цикла", "5 · События"])
        with t_pose:
            _pose_tab(tr, rdp, cfg, video, row.video_id)
        with t_ser:
            _series_tab(fr, fr_base, cfg, row.video_id)
        with t_cyc:
            _cycles_tab(fr, fr_base, tr, row.video_id, video)
        with t_score:
            res = _scoring(rdp, cfg, cfg_base, prof, row.video_id, over, tr, fr["series"], fr["cycles"], fr["rejected"])
            _score_tab(res, fr["cycles"], row.video_id)
        with t_ev:
            bundle = st.session_state.get("lab_bundle")
            res_ = st.session_state.get("lab_scored", {}).get(_score_key(rdp.name, prof, None if bundle in (None, "как в профиле") else bundle, st.session_state.get("lab_light", False), over))
            res_b = st.session_state.get("lab_scored", {}).get(_score_key(rdp.name, prof, None if bundle in (None, "как в профиле") else bundle, st.session_state.get("lab_light", False), {}))
            _events_tab(res_, res_b, tr, fr, cfg, row.video_id, dur)


# ---------------------------------------------------------------------------------------------- страница
def render() -> None:
    UC.style()
    UC.header("Лаборатория", "Каждая часть конвейера отдельно: настройка, обучение моделей, дообучение и активное обучение — с проверкой на реальных данных.")
    sec = UC.sections(SECTIONS, "lab_section")
    if sec == "Этапы на клипе":
        _stages()
    elif sec == "Временные ряды":
        lab_ts.render()
    elif sec == "Классификатор":
        lab_clf.render()
    elif sec == "Дообучение":
        lab_finetune.render()
    else:
        lab_active.render()
