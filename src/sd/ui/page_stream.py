"""Страница «Поток» (для разработчика): имитация реального времени по видеофайлу. Файл идёт через тот же движок, что и камеры; тревоги появляются в момент срабатывания,
с эталоном считаются TP/FP/FN, F1 и задержка тревоги, всегда — пропускная способность («справится ли устройство»)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sd import catalog as CAT
from sd import gt as GT
from sd import profiles as PR
from sd import runner as RN
from sd import solver as SV
from sd.paths import list_videos, video_id


def _profile() -> tuple[PR.Profile | None, str | None]:
    inst = [s["name"] for s in SV.installed_solvers()]
    names = PR.list_profiles()
    c = st.columns([2, 2, 2])
    src = c[0].radio("Источник решения", ["Профиль", "Установленный решатель"], horizontal=True, key="st_src", disabled=not inst)
    if src == "Установленный решатель" and inst:
        name = c[1].selectbox("Решатель", inst, key="st_solver")
        prof = SV.resolve_profile(name)
    elif names:
        prof = PR.load(c[1].selectbox("Профиль", names, key="st_prof"))
    else:
        st.info("Нет ни профилей, ни установленных решателей: соберите профиль на странице «Оценка» или установите решатель на странице «Модели».")
        return None, None
    cls = c[2].selectbox("Классификатор цикла", ["как в профиле", *[b.id for b in CAT.cycle_bundles()]], key="st_cls")
    prof = RN.with_classifier(prof, None if cls == "как в профиле" else cls)
    return prof, cls


def render() -> None:
    prof, _ = _profile()
    vids = list_videos()
    if not vids:
        st.info("Видео не найдены (каталог данных: SD_DATA).")
        return
    c = st.columns([4, 1, 1, 1])
    vid = c[0].selectbox("Видео", vids, format_func=lambda p: Path(p).name, key="st_vid")
    start = c[1].number_input("С, с", 0.0, value=0.0, step=10.0, key="st_s")
    dur = c[2].number_input("Длина, с (0 — весь)", 0.0, value=60.0, step=10.0, key="st_d")
    speed = c[3].selectbox("Темп", [0.0, 1.0, 2.0, 4.0], format_func=lambda v: "максимально быстро" if v == 0 else f"×{v:g}", key="st_sp")
    o = st.columns(3)
    use_gt = o[0].checkbox("Сверять с эталоном", True, help="labels/events_gt.csv; без разметки клипа метрики не считаются")
    render_v = o[1].checkbox("Видео с рамками и баннером тревоги", False)
    go = o[2].button("Запустить", type="primary", disabled=prof is None)
    if prof is not None:
        errs = SV.errors(prof, "replay")
        for e in errs:
            st.error(e.text)
        go = go and not errs
        st.caption(" · ".join(f"{k}: {v}" for k, v in prof.describe().items() if k in ("pose", "imgsz", "fps", "cycle_model", "photo", "objects", "fusion", "threshold")))
    if go and prof is not None:
        from sd.realtime import replay as RP

        box, bar, rows = st.container(), st.progress(0.0), []
        end = float(start + dur) if dur > 0 else None

        def on_alert(u) -> None:
            if u.kind == "open":
                rows.append(dict(поток_с=round(u.t_now, 1), начало_с=round(u.start, 1), ID=u.tid, причина=u.explain, уверенность=round(u.confidence, 2), задержка_с=round(u.t_now - u.start, 1)))
                box.dataframe(pd.DataFrame(rows), hide_index=True)

        def progress(t: float, tend: float) -> None:
            bar.progress(min(1.0, (t - start) / max(tend - start, 1e-6)) if end else 0.0)

        try:
            with st.spinner("Идёт поток…"):
                st.session_state["st_res"] = RP.replay_video(vid, prof, start=float(start), end=end, speed=float(speed), gt=GT.load() if use_gt else None, render=render_v,
                                                              on_alert=on_alert, progress=progress)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
        bar.progress(1.0)
    res = st.session_state.get("st_res")
    if res is None:
        return
    s = res.stats
    (st.success if (s.get("rt_factor") or 0) >= 1 else st.warning)(s["verdict"])
    k = st.columns(5)
    k[0].metric("Тревог", s["alerts"])
    k[1].metric("к/с обработки", s["fps_proc"], help=f"нужно {s['process_fps_target']:g}")
    k[2].metric("× реального времени", s["rt_factor"])
    k[3].metric("мс/кадр p50 / p95", f"{s['ms_p50']:.0f} / {s['ms_p95']:.0f}")
    k[4].metric("Циклов", s["cycles"])
    if res.metrics:
        m = res.metrics
        k = st.columns(5)
        k[0].metric("F1", f"{m['f1']:.3f}")
        k[1].metric("Precision / Recall", f"{m['precision']:.2f} / {m['recall']:.2f}")
        k[2].metric("TP / FP / FN", f"{m['tp']} / {m['fp']} / {m['fn']}")
        k[3].metric("Задержка тревоги, с (медиана)", "—" if m["alert_delay_median"] is None else f"{m['alert_delay_median']:.1f}")
        k[4].metric("Событий в эталоне", m["n_gt"])
    for n in res.notes:
        st.caption(n)
    if len(res.alerts):
        st.dataframe(res.alerts.drop(columns=["key"]), hide_index=True)
    if res.overlay and Path(res.overlay).exists():
        st.video(res.overlay)
    if res.out_dir:
        st.caption(f"Результаты: `{res.out_dir}`")
