"""Вкладки «Распознавание» (полный цикл одной кнопкой) и «Плеер» (просмотр результатов с таймлайном и выходами классификаторов)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sd import model_tools as MT
from sd import ollama_ctl as O
from sd import pipeline as P
from sd import stages
from sd.models import load_registry
from sd.paths import MODELS, OUTPUTS, ROOT
from sd.ui import nav

NONE = "— нет —"


def _bar(label: str):
    b = st.progress(0.0, text=label)
    last = {"t": 0.0}

    def cb(stage: str, i: int, n: int) -> None:
        if time.perf_counter() - last["t"] > 0.3 or i >= n:
            b.progress(min(i / max(n, 1), 1.0), text=f"{stage}: {i}/{n}")
            last["t"] = time.perf_counter()

    return b, cb


def _names(kind: str) -> list[str]:
    return [b["name"] for b in MT.list_bundles() if b["kind"] == kind]


def tab_recognize(vid, cfg, t0, t1, rd):
    cyc_b, photo_b = _names("cycle"), _names("photo")
    c1, c2, c3 = st.columns(3)
    cycle = c1.selectbox("Классификатор цикла (без VLM)", [NONE] + cyc_b, help="пакет из models/cycle; без него оценка — эвристика по длительности паузы")
    full = c2.selectbox("Классификатор цикла с VLM-признаком", [NONE] + cyc_b, help="для режима VLM «серая зона» / «все циклы»")
    photo = c3.selectbox("Фото-модель (признаки photo_*)", [NONE] + photo_b)
    reg = load_registry()
    dets = [s.id for s in reg.values() if s.kind == "detector" and s.local_path and s.local_path.exists()]
    objs = st.multiselect("Детекторы предмета на кропах кисть–рот", dets, help="community-веса проходят сканирование перед загрузкой")
    c4, c5, c6 = st.columns(3)
    vlm_mode = c4.radio("VLM", ["выкл", "серая зона", "все циклы"], horizontal=True, help="≈ 9 с на цикл на встроенной графике; «серая зона» — только циклы с оценкой дешёвого пакета в интервале")
    grey = c5.slider("Серая зона оценки", 0.0, 1.0, (0.3, 0.8), 0.05, disabled=vlm_mode != "серая зона")
    models_up = [m["name"] for m in O.status().get("models", [])] if vlm_mode != "выкл" and O.is_up() else []
    vlm_model = c6.selectbox("Модель Ollama", models_up or [NONE], disabled=vlm_mode == "выкл")
    if vlm_mode != "выкл" and not models_up:
        st.warning("Сервер Ollama не запущен или моделей нет: `sd.cmd ollama up --igpu`, `sd.cmd ollama pull qwen3.5:2b-q4_K_M` (вкладка 7 · VLM умеет запускать сервер).")
    render = st.checkbox("Видео с разметкой (overlay.mp4) для плеера", value=True)
    opts = P.Options(cycle_bundle=f"models/cycle/{cycle}" if cycle != NONE else None, cycle_bundle_full=f"models/cycle/{full}" if full != NONE else None,
                     photo_bundle=f"models/photo/{photo}" if photo != NONE else None, objects=tuple(objs),
                     vlm_model=vlm_model if vlm_mode != "выкл" and vlm_model != NONE else None,
                     vlm_mode={"выкл": "off", "серая зона": "grey", "все циклы": "all"}[vlm_mode], grey=tuple(grey), render=render)
    if st.button("Распознать окно", type="primary"):
        b, cb = _bar("распознавание")
        with st.spinner("считаем (CPU/iGPU)…"):
            res = P.recognize(vid, t0, t1, cfg, opts, progress=cb)
        b.empty()
        st.session_state["rec_last"] = str(res.out_dir)
    last = st.session_state.get("rec_last")
    if not last:
        st.caption("Поза берётся из кэша, если окно и параметры уже считались.")
        return
    d = Path(last)
    meta = json.loads((d / "result.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(d / "events.csv")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("циклов", meta["counts"]["cycles"])
    m2.metric("событий", meta["counts"]["events"])
    m3.metric("вызовов VLM", meta["counts"]["vlm_calls"])
    m4.metric("время, с", meta["timing_sec"]["total"])
    for w in meta["warnings"]:
        st.warning(w)
    st.caption("время этапов, с: " + ", ".join(f"{k} {v}" for k, v in meta["timing_sec"].items()))
    st.subheader("События")
    st.dataframe(ev, hide_index=True)
    f = d / "cycles_scored.csv"
    if f.exists():
        cy = pd.read_csv(f)
        cols = [c for c in ("tid", "start", "end", "hold", "score", "score_cheap", "photo_p_mean", "photo_p_max", "vlm_yesno", "vlm_called", "obj_any_max_conf") if c in cy.columns]
        st.subheader("Циклы и оценки")
        st.dataframe(cy[cols].round(3), hide_index=True)
        st.download_button("Скачать cycles_scored.csv", f.read_bytes(), file_name="cycles_scored.csv")
    st.download_button("Скачать events.json", (d / "events.json").read_bytes(), file_name="events.json")
    if st.button("Открыть в просмотре"):
        nav.go("view", view_src="Отдельный прогон", player_dir=str(d))
