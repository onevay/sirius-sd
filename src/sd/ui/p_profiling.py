"""Вкладка «Профилирование»: на что способно железо — характеристики машины, фактическая скорость позы по кэшу (все конфигурации и клипы), лестница вычислений
(сколько затяжек находит автомат при разных моделях/размерах входа и при имитации дальней камеры) и оценка числа камер."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from sd.paths import OUTPUTS
from sd.ui import common as UC

SECTIONS = ["Скорость по кэшу", "Лестница вычислений", "Машина"]


@st.cache_data(show_spinner="читаем замеры позы из кэша…", ttl=120)
def _pose_speed(stamp: float) -> pd.DataFrame:
    rows = []
    for f in (OUTPUTS / "runs").glob("*/*/pose/meta.json"):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:       # noqa: BLE001
            continue
        run = f.parent.parent.name
        cfg = run.split("_", 1)[1] if "_" in run else run
        vi, p = m.get("video_info") or {}, m.get("pose") or {}
        rows.append(dict(конфигурация=cfg, клип=f.parent.parent.parent.name, ширина=vi.get("width"), высота=vi.get("height"), модель=p.get("weights"), imgsz=p.get("imgsz"),
                         рантайм=f"{p.get('runtime')}/{p.get('device')}", уточнение=p.get("refine_method") or ("да" if p.get("refine") else "нет"), к_с_нужно=m.get("proc_fps"),
                         к_с_факт=m.get("fps_steady"), мс_кадр=((m.get("ms") or {}).get("total") or {}).get("median"), кадров=m.get("frames_processed")))
    return pd.DataFrame(rows)


def _speed() -> None:
    root = OUTPUTS / "runs"
    df = _pose_speed(UC.file_mtime(root))
    if df.empty:
        st.info("Кэша позы нет: запустите оценку («Валидация»).")
        return
    df["успевает"] = df.к_с_факт / df.к_с_нужно
    g = df.groupby("конфигурация").agg(клипов=("клип", "nunique"), модель=("модель", "first"), imgsz=("imgsz", "first"), рантайм=("рантайм", "first"),
                                       к_с_нужно=("к_с_нужно", "median"), к_с_факт=("к_с_факт", "median"), мс_кадр=("мс_кадр", "median"),
                                       успевает_медиана=("успевает", "median"), успевает_худший=("успевает", "min")).reset_index()
    g["камер_1080p"] = (g.успевает_медиана).round(2)
    st.markdown("**Поза по конфигурациям** (замеры при расчёте кэша на этой машине; ≥ 1 — поза успевает за частотой обработки одной камеры)")
    st.dataframe(g.sort_values("клипов", ascending=False), hide_index=True,
                 column_config={"успевает_медиана": st.column_config.ProgressColumn("успевает (медиана)", min_value=0, max_value=2, format="%.2f×"),
                                "камер_1080p": st.column_config.NumberColumn("≈ камер на машину", help="грубо: во сколько раз поза быстрее нужной частоты; без признаков предмета и фото")})
    st.caption("Оценка числа камер — верхняя граница по позе; признаки предмета и фото добавляют ≈ 2.5 с на цикл. Полная скорость конвейера по клипам — «Валидация → Анализ кэша».")
    with st.expander("Все замеры по клипам"):
        st.dataframe(df, hide_index=True)


def _ladder() -> None:
    from sd import ladder as LD

    files = sorted((OUTPUTS / "analysis").glob("ladder*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        f = st.selectbox("Результат", files, format_func=lambda p: f"{p.name} · {UC.ago(p.stat().st_mtime)}", key="ld_file")
        raw = pd.read_csv(f, encoding="utf-8-sig")
        raw["error"] = raw["error"].fillna("")
        s = LD.summarize(raw)
        st.dataframe(s, hide_index=True, column_config={"recall": st.column_config.ProgressColumn("найдено затяжек", min_value=0, max_value=1, format="%.0f%%")})
        st.caption("recall — доля размеченных затяжек, на которые пришёлся цикл автомата (потолок полноты всей системы); extra — циклы без затяжки (работа классификатору); "
                   "файлы *zoom*/*shrunk* — имитация дальней камеры.")
        with st.expander("По клипам"):
            st.dataframe(raw, hide_index=True)
    else:
        st.info("Лестницу ещё не запускали.")
    st.markdown("##### Запустить лестницу (фон)")
    c = st.columns([3, 3, 1])
    cfgs = c[0].multiselect("Конфигурации", ["n960", "n1280", "n1600", "s960", "s1600", "m1280", "x1280", "gpu_s1600", "gpu_m1600"], default=["n960", "n1280"], key="ld_cfg")
    clips = c[1].text_input("Клипы (video_id[:a-b][@масштаб] через запятую; пусто — по умолчанию)", key="ld_clips")
    scale = c[2].number_input("Масштаб", 0.2, 1.0, 1.0, 0.1, key="ld_scale", help="< 1 — люди мельче (имитация дальней камеры)")
    items = [("warning", "каждая конфигурация считает позу на клипах заново (если нет в кэше) — от минут до часов; на 8 ГБ ОЗУ не запускайте параллельно с оценкой")] if cfgs else [("error", "выберите конфигурации")]
    if any(c_.startswith(("x", "m", "gpu")) for c_ in cfgs):
        items.append(("warning", "m/x/gpu-конфигурации на этом ноутбуке очень медленные или требуют CUDA"))
    ok = UC.checklist(items)
    UC.task_button("ladder", dict(configs=",".join(cfgs), clips=clips.strip(), scale=float(scale), tag=f"_ui_{int(scale * 100)}"), "Запустить лестницу", key="ld_go", disabled=not ok)
    UC.task_panel(("ladder",), key="ld_tp", limit=2, title=None)


def _machine() -> None:
    if st.button("Собрать характеристики машины", key="hw_go", help="процессор, ОЗУ, графика, OpenVINO/CUDA, ffmpeg (несколько секунд)"):
        from sd import hw

        with st.spinner("опрос устройств…"):
            st.session_state["hw_info"] = hw.collect(bench=False)
    info = st.session_state.get("hw_info")
    if info:
        from sd import hw

        st.code(hw.render_report(info), language=None)
    from sd import experiments as XP

    st.caption("Кратко: " + " · ".join(f"{k}: {v}" for k, v in XP.host_info().items()))
    UC.task_button("doctor", {}, "Полная проверка окружения (фон)", key="hw_doc", primary=False)
    UC.task_panel(("doctor",), key="hw_tp", limit=1, title=None)


def render() -> None:
    UC.style()
    UC.header("Профилирование", "На что способно железо: скорость позы по кэшу, лестница вычислений, характеристики машины.")
    sec = UC.sections(SECTIONS, "prof_section")
    {"Скорость по кэшу": _speed, "Лестница вычислений": _ladder, "Машина": _machine}[sec]()
