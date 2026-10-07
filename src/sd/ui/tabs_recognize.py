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
from sd.ui.player import build_data, player_html

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
    st.markdown(
        "**Полный цикл распознавания по выбранному окну** (интервал и пороги — слева): поза+трекинг → циклы → признаки → оценка цикла выбранными моделями → события по регламенту. "
        "Результаты — в `outputs/recognize/…`; смотреть их удобно на вкладке «Плеер». Те же настройки есть в CLI: `sd.cmd recognize <видео> --start … --end … --cycle-bundle …`.")
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
    if st.button("▶ Распознать окно", type="primary"):
        b, cb = _bar("распознавание")
        with st.spinner("считаем (CPU/iGPU)…"):
            res = P.recognize(vid, t0, t1, cfg, opts, progress=cb)
        b.empty()
        st.session_state["rec_last"] = str(res.out_dir)
    last = st.session_state.get("rec_last")
    if not last:
        st.info("Нажмите кнопку. Поза берётся из кэша, если для этого окна и параметров уже считалась.")
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
    # состояние виджета «Этап» можно менять только в callback (до перерисовки), иначе Streamlit выбросит исключение
    st.button("Открыть в плеере", on_click=lambda: st.session_state.update(stage="Плеер", player_dir=str(d)))


def _result_data(d: Path) -> tuple[Path | None, dict, dict]:
    meta = json.loads((d / "result.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(d / "events.csv") if (d / "events.csv").exists() else pd.DataFrame()
    cy = pd.read_parquet(d / "cycles_scored.parquet") if (d / "cycles_scored.parquet").exists() else pd.DataFrame()
    info = meta["window"]
    dur = (info[1] - info[0]) if info[1] is not None else 0.0
    video = d / "overlay.mp4" if (d / "overlay.mp4").exists() else None
    return video, build_data(cy, ev, dur, 15.0, float(info[0] or 0.0)), meta


def tab_player(vid, cfg, t0, t1, rd):
    st.markdown("**Плеер**: видео с точками/состояниями/оценками + таймлайн циклов и событий + панель выходов моделей. Клавиши: Пробел, ←/→ (кадр), `,` `.` (цикл), `L` (зацикливание), `[` `]` (скорость).")
    src = st.radio("Источник", ["результат распознавания", "видео этапов текущего запуска (без оценок)"], horizontal=True)
    if src.startswith("результат"):
        res = P.list_results()
        if res.empty:
            st.info("Пока нет результатов: запустите вкладку «Распознавание».")
            return
        default = st.session_state.get("player_dir")
        idx = int(res.index[res.dir == default][0]) if default in set(res.dir) else 0
        k = st.selectbox("Результат", res.index, index=idx, format_func=lambda i: f"{res['clip'][i]} · {res.window[i]} · {res.created[i]} · циклов {res.cycles[i]}, событий {res.events[i]}")
        d = Path(res.dir[k])
        video, data, meta = _result_data(d)
        if video is None:
            st.warning("В этом результате нет overlay.mp4 (запуск без видео). Перезапустите распознавание с галочкой «Видео с разметкой».")
        ev_df = pd.read_csv(d / "events.csv") if (d / "events.csv").exists() else pd.DataFrame()
        cy = pd.read_parquet(d / "cycles_scored.parquet") if (d / "cycles_scored.parquet").exists() else pd.DataFrame()
        # время в overlay.mp4 — время исходного видео (кадры идут с метками видео), поэтому длительность плеера = конец окна
        w0, w1 = float(meta["window"][0] or 0.0), meta["window"][1]
        data["offset"] = w0
        data["duration"] = float((w1 - w0) if w1 is not None else (cy["end"].max() + 5 - w0 if len(cy) else 30.0))   # уточняется в браузере по реальной длительности mp4
    else:
        tr_ok = (rd / "pose" / "tracks.parquet").exists()
        vids = sorted((rd / "render").glob("*.mp4")) if (rd / "render").exists() else []
        if not tr_ok or not vids:
            st.info("Нет видео этапов для текущего запуска: на вкладках 1–6 нажмите «🎞 Видео …».")
            return
        video = st.selectbox("Видео", vids, format_func=lambda p: p.name)
        from sd.tracks import Tracks

        tr = Tracks.load(rd / "pose")
        ser = stages.stage_features(tr, cfg, rd)
        cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
        ev_df, _ = stages.stage_events(tr, ser, cyc, cfg, rd, "cam_local", vid.stem)
        cy = cyc.assign(score=float("nan"))
        data = build_data(cy, ev_df, float(t1 - t0), 15.0, float(t0))
        meta = None
    html, warn = player_html(video, data)
    if warn:
        st.warning(warn)
        if video:
            st.video(str(video))
    components.html(html, height=720, scrolling=False)
    if len(cy):
        cols = [c for c in ("tid", "start", "end", "hold", "score", "score_cheap", "photo_p_mean", "vlm_yesno", "obj_any_max_conf") if c in cy.columns]
        st.dataframe(cy[cols].round(3), hide_index=True)
    if len(ev_df):
        st.dataframe(ev_df, hide_index=True)
    if meta:
        with st.expander("Что запускалось (result.json)"):
            st.json(meta)
