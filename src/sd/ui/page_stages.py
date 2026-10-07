"""Страница «Этапы»: разбор пайплайна по шагам (поза → признаки → циклы → предмет → видео-модели → события → VLM), метки циклов, обучение, бенчмарки.

Для отладки и обучения; оценка качества, разметка событий и просмотр результатов — на отдельных страницах (`page_eval`, `page_label`, `page_view`).
Параметры этапов — в боковой панели ЭТОЙ страницы. Тяжёлые шаги запускаются кнопками, результаты кэшируются в `outputs/runs/...`.
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sd import hw as HW
from sd import library as LIB
from sd import models as M
from sd import stages
from sd.calibrate import cfg_current_for_run
from sd.config import load_config
from sd.cycles import STATE_ID
from sd.dataset import LABEL_CLASSES, attach_labels, build_cycle_table, collect_runs, feature_columns, load_labels, save_label
from sd.paths import MODELS, OUTPUTS, ROOT, list_videos, video_id, weak_label
from sd.render import Renderer, render_frame_img, render_video
from sd.tracks import Tracks
from sd.video_io import probe

STATE_NAMES = {v: k for k, v in STATE_ID.items()}


# ------------------------------------------------------------------------------------------ данные
@st.cache_data(show_spinner=False)
def video_table() -> pd.DataFrame:
    rows = []
    for p in LIB.videos_in(LIB.discover_dirs().path):
        i = probe(p)
        rows.append(dict(path=str(p), name=p.name, label=weak_label(p) or "-", res=f"{i.width}x{i.height}", fps=i.fps, codec=i.codec,
                         dur=i.duration, mb=i.size_mb))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=120)
def hw_info(bench: bool = False) -> dict:
    return HW.collect(bench=bench)


@st.cache_resource(show_spinner=False)
def _load_tracks(path: str, mtime: float) -> Tracks:
    return Tracks.load(path)


def cached_tracks(rd: Path) -> Tracks | None:
    p = rd / "pose"
    return _load_tracks(str(p), (p / "tracks.parquet").stat().st_mtime) if (p / "tracks.parquet").exists() else None


def bar(label: str):
    b = st.progress(0.0, text=label)

    def cb(i: int, n: int) -> None:
        b.progress(min(i / max(n, 1), 1.0), text=f"{label}: {i}/{n}")

    return b, cb


# ------------------------------------------------------------------------------------------ боковая панель
def sidebar():
    st.sidebar.subheader("Видео и интервал")
    vt = video_table()
    if vt.empty:
        st.info("Видео не найдены: положите клипы в `data/` или `data_external/`.")
        st.stop()
    idx = st.sidebar.selectbox("Видео", vt.index, format_func=lambda i: f"[{vt.label[i]}] {vt.name[i]} · {vt.dur[i]:.0f} с · {vt.res[i]}")
    vid = Path(vt.path[idx])
    dur = float(vt.dur[idx])
    t0, t1 = st.sidebar.slider("Интервал анализа, с", 0.0, max(dur, 1.0), (0.0, min(dur, 40.0)), step=1.0)
    if t1 - t0 > 120:
        st.sidebar.warning("Интервал > 120 с: на этом CPU это долго")
    base = load_config()   # значения по умолчанию в панели = configs/default.yaml (после калибровки порогов UI подхватит их сам)
    reg = M.load_registry()
    pose_ids = [s.id for s in reg.values() if s.kind == "pose" and ((s.file and (MODELS / s.file).exists()) or s.rtmlib)]
    st.sidebar.subheader("Поза и трекинг")
    weights = st.sidebar.selectbox("Модель позы", pose_ids, index=pose_ids.index(base.pose.weights) if base.pose.weights in pose_ids else 0)
    imgsz = st.sidebar.select_slider("imgsz (длинная сторона)", [480, 640, 800, 960, 1088, 1280], value=int(base.pose.imgsz))
    runtimes = ["torch", "openvino"]
    runtime = st.sidebar.radio("Рантайм", runtimes, index=runtimes.index(base.pose.runtime), horizontal=True)
    devs = ["intel:gpu", "intel:cpu"]
    device = "cpu" if runtime == "torch" else st.sidebar.selectbox("Устройство OpenVINO", devs, index=devs.index(base.pose.device) if base.pose.device in devs else 0)
    trackers = ["botsort", "bytetrack"]
    tracker = st.sidebar.radio("Трекер", trackers, index=trackers.index(base.tracking.tracker), horizontal=True)
    refine = st.sidebar.checkbox("Уточнение ключевых точек по кропу (top-down)", value=bool(base.pose.refine.enabled))
    methods = ["rtmpose-s", "rtmpose-m", "yolo"]
    refine_method = st.sidebar.selectbox("Метод уточнения", methods, index=methods.index(base.pose.refine.method)) if refine else base.pose.refine.method
    pfps = st.sidebar.slider("Частота обработки, к/с", 4, 15, int(base.video.process_fps))
    st.sidebar.subheader("Автомат циклов")
    th_in = st.sidebar.slider("th_in — вход ко рту", 0.10, 1.50, float(base.cycles.th_in), 0.05)
    th_out = st.sidebar.slider("th_out — выход", 0.20, 2.00, float(base.cycles.th_out), 0.05)
    hold = st.sidebar.slider("Пауза у рта, с (min–max)", 0.2, 8.0, (float(base.cycles.hold_min_sec), float(base.cycles.hold_max_sec)), 0.1)
    hand_extend = st.sidebar.slider("Продление кисти вдоль предплечья, k", 0.0, 1.0, float(base.features.hand_extend), 0.1,
                                    help="точка кисти = запястье + k·(запястье − локоть); 0 — просто запястье (как в руководстве)")
    st.sidebar.subheader("Сборка событий")
    cycle_th = st.sidebar.slider("Порог оценки цикла", 0.0, 1.0, float(base.events.cycle_th), 0.05)
    merge_gap = st.sidebar.slider("Склейка циклов, с", 5.0, 30.0, float(base.events.merge_gap_sec), 1.0)
    window = st.sidebar.slider("Окно «два цикла», с", 10.0, 40.0, float(base.events.window_sec), 1.0)
    ov = [f"pose.weights={weights}", f"pose.imgsz={imgsz}", f"pose.runtime={runtime}", f"pose.device={device}", f"tracking.tracker={tracker}",
          f"pose.refine.enabled={str(refine).lower()}", f"pose.refine.method={refine_method}", f"video.process_fps={pfps}", f"cycles.th_in={th_in}",
          f"cycles.th_out={max(th_out, th_in + 0.05)}", f"features.hand_extend={hand_extend}",
          f"cycles.hold_min_sec={hold[0]}", f"cycles.hold_max_sec={hold[1]}", f"events.cycle_th={cycle_th}", f"events.merge_gap_sec={merge_gap}",
          f"events.window_sec={window}"]
    if weights.startswith("rtmlib"):
        ov.append("pose.backend=rtmlib")
    cfg = load_config(overrides=ov)
    rd = stages.run_dir(vid, cfg, t0, t1, create=False)
    cached = (rd / "pose" / "tracks.parquet").exists()
    st.sidebar.caption(f"Каталог запуска: `{rd.relative_to(OUTPUTS)}`  \nПоза в кэше: {'да' if cached else 'нет'}")
    return vid, cfg, t0, t1, rd


# ------------------------------------------------------------------------------------------ вспомогательное
def renderer_for(tr, cfg, rd, upto: str) -> Renderer:
    kw = {}
    if upto in ("features", "cycles", "events"):
        kw["series"] = stages.stage_features(tr, cfg, rd)
    if upto in ("cycles", "events"):
        cyc, rej, st_ = stages.stage_cycles(kw["series"], cfg, rd)
        kw.update(cycles=cyc, states=st_)
    if upto == "events":
        df, _ = stages.stage_events(tr, kw["series"], kw["cycles"], cfg, rd, "cam_local", "clip")
        kw["events"] = df
    return Renderer(tr, cfg, **kw)


def frame_viewer(vid, tr, cfg, rd, upto: str, key: str, t_default: float | None = None):
    ft = tr.frame_t
    step = float(np.median(np.diff(ft.t.values))) if len(ft) > 2 else 0.1
    t = st.slider("Момент, с", float(ft.t.min()), float(ft.t.max()), float(t_default if t_default is not None else ft.t.median()), step=step, key=f"t_{key}")
    r = renderer_for(tr, cfg, rd, upto)
    img, tt = render_frame_img(vid, tr, cfg, t, renderer=r)
    st.image(img[:, :, ::-1], caption=f"t = {tt:.2f} с", width="stretch")


def plot_d(s: pd.DataFrame, cyc: pd.DataFrame, rej: pd.DataFrame, cfg, cursor: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.t, y=s.d_raw, name="d сырое", mode="lines", line=dict(color="lightgray", width=1)))
    fig.add_trace(go.Scatter(x=s.t, y=s.d, name="d сглаженное", mode="lines", line=dict(color="black", width=2)))
    fig.add_hline(y=cfg["cycles"]["th_in"], line_dash="dash", line_color="green", annotation_text="th_in")
    fig.add_hline(y=cfg["cycles"]["th_out"], line_dash="dash", line_color="orange", annotation_text="th_out")
    for c in cyc.itertuples():
        fig.add_vrect(x0=c.start, x1=c.end, fillcolor="rgba(255,0,0,0.10)", line_width=0)
        fig.add_vrect(x0=c.mouth_in, x1=c.mouth_out, fillcolor="rgba(255,0,0,0.30)", line_width=0)
    for r in rej.itertuples():
        fig.add_vrect(x0=r.t0, x1=r.t1, fillcolor="rgba(120,120,120,0.25)", line_width=0, annotation_text=r.reason, annotation_position="top left")
    if cursor is not None:
        fig.add_vline(x=cursor, line_color="deepskyblue")
    fig.update_yaxes(range=[0, 3.0], title="d = запястье–рот, ширин плеч")
    fig.update_xaxes(title="время, с")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    return fig


# ------------------------------------------------------------------------------------------ вкладки
def tab_overview(vid, cfg):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Железо")
        if st.button("Собрать отчёт (с бенчмарком GFLOPS)"):
            st.session_state["hw"] = HW.collect(bench=True)
        info = st.session_state.get("hw") or hw_info()
        st.code(HW.render_report(info), language="text")
        rows = pd.DataFrame([r for r in HW.vlm_budget(info) if r["quant"] in ("Q4", "Q8")])
        st.caption("VLM: помещается ли в память (оценка)")
        st.dataframe(rows[["model", "quant", "weights_gb", "need_gb", "usable_gb", "avail_now_gb", "fits", "fits_now"]], hide_index=True)
    with c2:
        st.subheader("Веса")
        reg = M.load_registry()
        st.dataframe(pd.DataFrame([M.status(s) for s in reg.values()]), hide_index=True)
        st.subheader("Видео датасета")
        st.dataframe(video_table().drop(columns=["path"]), hide_index=True, height=300)


def tab_pose(vid, cfg, t0, t1, rd):
    c1, c2 = st.columns([1, 4])
    force = c1.checkbox("Пересчитать (force)")
    if c1.button("Запустить поза+трекинг", type="primary"):
        b, cb = bar("поза+трекинг")
        with st.spinner("считаем (CPU)…"):
            stages.stage_pose(vid, cfg, t0, t1, force=force, progress=cb)
        b.empty()
        st.cache_resource.clear()
    tr = cached_tracks(rd)
    if tr is None:
        c2.info("Результат ещё не посчитан для этих параметров.")
        return
    m = tr.meta
    c2.markdown(f"**{m['pose']['weights']}** · {m['pose']['runtime']}/{m['pose']['device']} · imgsz {m['pose']['imgsz']} · трекер {m['pose']['tracker']} · "
                f"кадров **{m['frames_processed']}** · **{m.get('fps_steady') or m['fps_wall']} к/с** установившаяся (с декодированием; прогрев {m.get('warmup_sec', '—')} с) · "
                f"мс на шаг (поза+трекер): медиана {m['ms']['total'].get('median')}, p95 {m['ms']['total'].get('p95')}")
    st.markdown(f"Треков до склейки **{m['stitch']['tracks_before']}** → после **{m['stitch']['tracks_after']}** (склеено {len(m['stitch']['merged'])}, коротких удалено {m['stitch']['dropped_short']})")
    s = tr.summary(cfg["video"]["min_person_height_px"])
    st.dataframe(s.round(2), hide_index=True)
    frame_viewer(vid, tr, cfg, rd, "pose", "pose")
    if st.button("Видео с разметкой", key="vid_pose"):
        b, cb = bar("рендер")
        p = render_video(vid, tr, cfg, rd / "render" / "pose.mp4", renderer=renderer_for(tr, cfg, rd, "pose"), progress=cb)
        b.empty()
        st.video(str(p))


def tab_features(vid, cfg, t0, t1, rd):
    tr = cached_tracks(rd)
    if tr is None:
        st.info("Сначала запустите поза+трекинг (вкладка 1).")
        return
    ser = stages.stage_features(tr, cfg, rd)
    tids = tr.tids
    tid = st.selectbox("Трек", tids, key="f_tid")
    s = ser[ser.tid == tid]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("d валиден", f"{s.d.notna().mean():.0%}")
    c2.metric("d минимум", f"{s.d.min():.2f}" if s.d.notna().any() else "—")
    c3.metric("масштаб s, px", f"{s.s.median():.0f}")
    c4.metric("conf запястья/носа", f"{s.wrist_conf.mean():.2f} / {s.nose_conf.mean():.2f}")
    st.plotly_chart(plot_d(s, pd.DataFrame(columns=["start", "end", "mouth_in", "mouth_out"]), pd.DataFrame(columns=["t0", "t1", "reason"]), cfg))
    fig = go.Figure()
    for col, nm in (("wrist_conf", "уверенность запястья"), ("nose_conf", "уверенность носа")):
        fig.add_trace(go.Scatter(x=s.t, y=s[col], name=nm))
    fig.add_trace(go.Scatter(x=s.t, y=s.head_tilt, name="наклон головы (нос−уши)/s", yaxis="y2"))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.05]), yaxis2=dict(overlaying="y", side="right"), legend=dict(orientation="h"))
    st.plotly_chart(fig)
    frame_viewer(vid, tr, cfg, rd, "features", "feat")


def tab_cycles(vid, cfg, t0, t1, rd):
    tr = cached_tracks(rd)
    if tr is None:
        st.info("Сначала запустите поза+трекинг (вкладка 1).")
        return
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, states = stages.stage_cycles(ser, cfg, rd)
    st.markdown(f"**Циклов: {len(cyc)}**, отброшено жестов: {len(rej)} — {rej.reason.value_counts().to_dict() if len(rej) else '{}'}")
    tid = st.selectbox("Трек", tr.tids, key="c_tid")
    s = ser[ser.tid == tid]
    cursor = None
    ct = cyc[cyc.tid == tid].reset_index(drop=True)
    if len(ct):
        k = st.selectbox("Цикл", range(len(ct)), format_func=lambda i: f"#{i}: {ct.start[i]:.1f}–{ct.end[i]:.1f} с, пауза {ct.hold[i]:.1f} с", key="c_idx")
        cursor = float(ct.peak_t[k])
    st.plotly_chart(plot_d(s, ct, rej[rej.tid == tid], cfg, cursor))
    if len(cyc):
        st.dataframe(cyc.round(2), hide_index=True)
    frame_viewer(vid, tr, cfg, rd, "cycles", "cyc", t_default=cursor)
    if st.button("Видео с состояниями", key="vid_cyc"):
        b, cb = bar("рендер")
        p = render_video(vid, tr, cfg, rd / "render" / "cycles.mp4", renderer=renderer_for(tr, cfg, rd, "cycles"), progress=cb)
        b.empty()
        st.video(str(p))


def tab_object(vid, cfg, t0, t1, rd):
    from sd.evidence import cycle_object_evidence, run_evidence

    tr = cached_tracks(rd)
    if tr is None:
        st.info("Сначала запустите поза+трекинг (вкладка 1).")
        return
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, _ = stages.stage_cycles(ser, cfg, rd)
    reg = M.load_registry()
    dets = [s.id for s in reg.values() if s.kind == "detector" and s.local_path and s.local_path.exists()]
    sel = st.multiselect("Детекторы сигареты/вейпа", dets, default=dets[:1])
    conf = st.slider("Порог уверенности", 0.05, 0.9, float(cfg["evidence"]["conf"]), 0.05)
    cfg2 = load_config(overrides=[f"evidence.conf={conf}"])
    out_dir = rd / "evidence"
    if st.button("Искать предмет на кропах", type="primary", disabled=not len(cyc) or not sel):
        b, cb = bar("детектор на кропах")
        with st.spinner("…"):
            det = run_evidence(vid, tr, ser, cyc, cfg2, sel, progress=cb, save_crops=out_dir / "crops")
        b.empty()
        out_dir.mkdir(exist_ok=True)
        det.to_parquet(out_dir / "detections.parquet", index=False)
    f = out_dir / "detections.parquet"
    if not f.exists():
        st.info("Нажмите кнопку — детектор прогонится по кадрам вокруг каждого цикла.")
        return
    det = pd.read_parquet(f)
    ev = cycle_object_evidence(det, cfg2)
    st.dataframe(ev.round(2), hide_index=True)
    st.caption("has_object: предмет найден в ≥3 из 5 соседних кадров")
    imgs = sorted((out_dir / "crops").glob("*.jpg"), key=lambda p: ("_HIT" not in p.name, p.name))[:24]
    cols = st.columns(6)
    for i, p in enumerate(imgs):
        cols[i % 6].image(str(p), caption=p.stem, width="stretch")


def tab_tube(vid, cfg, t0, t1, rd):
    from sd.tube import run_tube

    tr = cached_tracks(rd)
    if tr is None:
        st.info("Сначала запустите поза+трекинг (вкладка 1).")
        return
    ser = stages.stage_features(tr, cfg, rd)
    cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
    models = st.multiselect("Видео-модели", ["xclip", "videomae"], default=["xclip"])
    mode = st.radio("Окна", ["вокруг циклов (пик ± 1.25 с)", "свой интервал"], horizontal=True)
    wins = []
    if mode.startswith("вокруг"):
        wins = [(i, int(r.tid), float(r.peak_t - 1.25), float(r.peak_t + 1.25)) for i, r in enumerate(cyc.itertuples())]
    else:
        c1, c2, c3 = st.columns(3)
        tid = c1.selectbox("Трек", tr.tids, key="t_tid")
        a = c2.number_input("с", float(t0), float(t1), float(t0), 0.5)
        wins = [(0, int(tid), float(a), float(a + cfg["tube"]["window_sec"]))]
    st.caption(f"{len(wins)} окон × {len(models)} модель(и)")
    out_dir = rd / "tube"
    if st.button("Запустить", type="primary", disabled=not wins or not models):
        b, cb = bar("видео-модели")
        with st.spinner("…"):
            df = run_tube(vid, tr, wins, cfg, tuple(models), progress=cb, save_dir=out_dir / "strips")
        b.empty()
        out_dir.mkdir(exist_ok=True)
        df.to_parquet(out_dir / "tube.parquet", index=False)
        st.session_state["tube_ms"] = df.attrs.get("ms")
    f = out_dir / "tube.parquet"
    if not f.exists():
        return
    df = pd.read_parquet(f)
    if st.session_state.get("tube_ms"):
        st.write("мс на окно:", st.session_state["tube_ms"])
    pcols = [c for c in df.columns if c.startswith("p_")]
    st.dataframe(df.round(3), hide_index=True)
    if pcols:
        fig = go.Figure()
        for c in pcols:
            fig.add_trace(go.Bar(x=[f"w{w}" for w in df.window], y=df[c], name=c[2:]))
        fig.update_layout(barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="X-CLIP: доля промпта")
        st.plotly_chart(fig)
    strips = sorted((out_dir / "strips").glob("*.jpg"))[:12]
    for p in strips:
        st.image(str(p), caption=p.stem)


def tab_events(vid, cfg, t0, t1, rd):
    tr = cached_tracks(rd)
    if tr is None:
        st.info("Сначала запустите поза+трекинг (вкладка 1).")
        return
    ser = stages.stage_features(tr, cfg, rd)
    cyc, _, st_ = stages.stage_cycles(ser, cfg, rd)
    df, evs = stages.stage_events(tr, ser, cyc, cfg, rd, "cam_local", vid.stem)
    st.markdown(f"**Событий: {len(df)}** · оценка цикла здесь — эвристика; с обученной моделью — страница «Оценка»")
    for e in evs:
        st.write(f"ID{e.tid}: {e.start:.1f}–{e.end:.1f} с · confidence {e.confidence:.2f} · {e.explain()} · правило `{e.rule}`")
    st.dataframe(df, hide_index=True)
    if len(df):
        st.download_button("Скачать CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"events_{vid.stem}.csv")
    frame_viewer(vid, tr, cfg, rd, "events", "ev", t_default=float(df.peak_sec.iloc[0]) if len(df) else None)
    if st.button("Видео с событиями", key="vid_ev"):
        b, cb = bar("рендер")
        p = render_video(vid, tr, cfg, rd / "render" / "events.mp4", renderer=renderer_for(tr, cfg, rd, "events"), progress=cb)
        b.empty()
        st.video(str(p))


def tab_label(vid, cfg, t0, t1, rd):
    from sd.tube import tube_frames

    tr = cached_tracks(rd)
    if tr is None:
        st.info("Сначала запустите поза+трекинг (вкладка 1).")
        return
    ser = stages.stage_features(tr, cfg, rd)
    cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
    if cyc.empty:
        st.info("Циклов нет. Ослабьте пороги циклов слева (th_in/th_out) — для разметки нужен высокий recall.")
        return
    table = attach_labels(build_cycle_table(vid, tr, ser, cyc), use_weak=False)
    done = table.label.notna().sum()
    st.progress(done / len(table), text=f"размечено {done} из {len(table)}")
    todo = table[table.label.isna()]
    i = int(st.number_input("Цикл №", 0, len(table) - 1, int(todo.index[0]) if len(todo) else 0))
    r = table.loc[i]
    st.write(f"**Цикл #{i}** · ID{int(r.tid)} · {r.start:.1f}–{r.end:.1f} с · пауза {r.hold:.1f} с · текущая метка: `{r.label}` ({r.label_source or '—'}) · папка видео: `{r.weak_label}`")
    frames, ts, raw = tube_frames(vid, tr, int(r.tid), float(r.start - 0.3), float(r.end + 0.3), cfg, n=8)
    if frames:
        cols = st.columns(8)
        for k, (fr, t) in enumerate(zip(raw, ts)):
            cols[k].image(cv2.cvtColor(cv2.resize(fr, (200, 200)), cv2.COLOR_BGR2RGB), caption=f"{t:.1f} с")
    names = {"smoke": "затяжка (курение)", "drink": "питьё", "phone": "телефон", "eat": "еда", "touch_face": "касание лица", "other_neg": "другое (не курение)",
             "ignore": "игнор (не видно/засветка)", "unsure": "не уверен"}
    cols = st.columns(len(names))
    for (k, nm), col in zip(names.items(), cols):
        if col.button(nm, key=f"lab_{k}_{i}"):
            save_label(r.video, int(r.tid), float(r.peak_t), float(r.cx), float(r.cy), k)
            st.rerun()
    lab = load_labels()
    if len(lab):
        st.caption(f"Всего ручных меток: {len(lab)} · {lab.label.value_counts().to_dict()}")


def tab_train():
    from sd.train import train_cycle_model

    runs = collect_runs()
    st.markdown(f"Запусков с циклами: **{len(runs)}**. Слабые метки (по папке) годятся только для проверки конвейера.")
    use_weak = st.checkbox("Использовать слабые метки по папке", value=True)
    if st.button("Собрать таблицу циклов по всем запускам"):
        parts = []
        for vname, rd in runs:
            tr = Tracks.load(rd / "pose")
            cfg = cfg_current_for_run(rd)
            ser = stages.stage_features(tr, cfg, rd)
            cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
            vp = Path(tr.meta["video"])
            parts.append(build_cycle_table(vp, tr, ser, cyc))
        if parts:
            tab = pd.concat([p for p in parts if len(p)], ignore_index=True)
            (OUTPUTS / "dataset").mkdir(parents=True, exist_ok=True)
            tab.to_parquet(OUTPUTS / "dataset" / "cycles.parquet", index=False)
    f = OUTPUTS / "dataset" / "cycles.parquet"
    if not f.exists():
        st.info("Таблица циклов ещё не собрана.")
        return
    tab = attach_labels(pd.read_parquet(f), use_weak=use_weak)
    st.write(f"Циклов: {len(tab)}; размечено: {tab.y.notna().sum()} (ручных {int((tab.label_source == 'manual').sum())}); групп-видео: {tab.group.nunique()}")
    st.dataframe(tab.groupby(["weak_label", "label_source", "label"], dropna=False).size().rename("n").reset_index(), hide_index=True)
    if st.button("Обучить LightGBM (групповая CV, калибровка, абляция)", type="primary"):
        with st.spinner("обучение…"):
            res = train_cycle_model(tab, ROOT / "models" / "cycle")
        st.json(res)


def tab_bench():
    for p in sorted((OUTPUTS / "analysis").glob("*.csv")):
        st.subheader(p.name)
        st.dataframe(pd.read_csv(p), hide_index=True)
    for p in sorted((ROOT / "docs" / "img").glob("*.png")):
        st.image(str(p), caption=p.name)


def tab_vlm(vid, cfg, t0, t1, rd):
    """Этап 7: VLM-верификатор (локальный Ollama): вопрос «затяжка?» по кадрам цикла, вероятность по логитам ответа, прогон по всем циклам запуска."""
    from sd import ollama_ctl as O
    from sd.evidence import mouth_frames
    from sd.tube import tube_frames
    from sd.vlm import OllamaVLM

    tr = cached_tracks(rd)
    if tr is None:
        st.info("Сначала запустите поза+трекинг (вкладка 1).")
        return
    ser = stages.stage_features(tr, cfg, rd)
    cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
    info = O.status()
    if not info["up"]:
        st.warning("Сервер Ollama не запущен. Он локальный (127.0.0.1), без облака: кадры никуда не уходят.")
        igpu = st.checkbox("Встроенная графика (Vulkan): в 2 раза быстрее, но большие модели могут не поместиться в память", value=True)
        if st.button("Запустить сервер Ollama"):
            with st.spinner("запускаем…"):
                st.success(O.start({"OLLAMA_IGPU_ENABLE": "1"} if igpu else None))
            st.rerun()
        return
    models = [m["name"] for m in info["models"]]
    if not models:
        st.info("Моделей нет. В терминале: `sd.cmd ollama pull qwen3.5:2b-q4_K_M`")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    model = c1.selectbox("Модель Ollama", models, index=models.index("qwen3.5:2b-q4_K_M") if "qwen3.5:2b-q4_K_M" in models else 0)
    crop = c2.radio("Кадры для модели", ["рот и кисть (крупно)", "верх тела (трубка)"], horizontal=True)
    c3.metric("ОЗУ свободно, ГБ", info["free_ram_gb"])
    if cyc.empty:
        st.info("Циклов нет: ослабьте пороги циклов слева.")
        return

    def frames_for(r):
        a, b = float(r.peak_t) - 1.25, float(r.peak_t) + 1.25
        if crop.startswith("рот"):
            return mouth_frames(vid, tr, ser, int(r.tid), a, b, cfg, n=6, size=224)[0]
        return tube_frames(vid, tr, int(r.tid), a, b, cfg, n=6)[0]

    vlm = OllamaVLM(model, size=224, mode="yesno")
    k = st.selectbox("Цикл", range(len(cyc)), format_func=lambda i: f"#{i}: ID{cyc.tid[i]} {cyc.start[i]:.1f}–{cyc.end[i]:.1f} с, пауза {cyc.hold[i]:.1f} с", key="vlm_idx")
    if st.button("Спросить VLM про цикл", type="primary"):
        r = cyc.iloc[k]
        fr = frames_for(r)
        if not fr:
            st.warning("Трек не покрывает окно вокруг пика цикла.")
        else:
            with st.spinner("модель думает (первый запрос дольше: загрузка модели в память)…"):
                ans = vlm.verify(fr, 2.5)
            st.image(np.hstack([cv2.resize(f, (170, 170)) for f in fr]), caption="кадры, которые увидела модель (6 шт. за 2.5 с вокруг пика цикла)")
            p = ans.get("p_puff")
            if p is not None:
                st.metric("Вероятность «затяжка» по логитам ответа", f"{p:.0%}")
                st.progress(float(p))
            st.caption(f"ответ модели: «{ans['raw']}» · {ans['ms'] / 1000:.1f} с (prefill {ans['prefill_ms'] / 1000:.1f} с) · токенов промпта {ans['prompt_tokens']} · "
                       "малые модели занижают оценки: ориентир — порог 0.1–0.2 у 0.8B и около 0.5 у 2B")
    key = f"vlm_all::{rd}::{model}::{crop}"
    if st.button("Прогнать все циклы"):
        b, cb = bar("VLM по циклам")
        rows = []
        for i, r in enumerate(cyc.itertuples()):
            fr = frames_for(cyc.iloc[i])
            if fr:
                a = vlm.verify(fr, 2.5)
                rows.append(dict(цикл=i, ID=int(r.tid), начало=round(float(r.start), 1), конец=round(float(r.end), 1), пауза=round(float(r.hold), 1),
                                 p_затяжка=round(float(a["p_puff"]), 3) if a.get("p_puff") is not None else None, секунд=round(a["ms"] / 1000, 1)))
            cb(i + 1, len(cyc))
        b.empty()
        st.session_state[key] = pd.DataFrame(rows)
    if key in st.session_state:
        df = st.session_state[key]
        st.dataframe(df, hide_index=True)
        if len(df) and df.p_затяжка.notna().any():
            fig = go.Figure(go.Bar(x=[f"#{i}" for i in df.цикл], y=df.p_затяжка, marker_color="#e87ba4"))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1], title="вероятность «затяжка»"))
            st.plotly_chart(fig)


STAGES = ["Распознавание окна", "Поза и трекинг", "Признаки", "Циклы", "Предмет", "Видео-модели", "События", "VLM", "Метки циклов", "Обучение", "Бенчмарки", "Железо и веса"]


def render() -> None:
    vid, cfg, t0, t1, rd = sidebar()
    stage = st.radio("Этап", STAGES, horizontal=True, label_visibility="collapsed", key="stage")   # radio, а не st.tabs: выполняется только выбранный этап
    from sd.ui.tabs_recognize import tab_recognize

    fn = {"Распознавание окна": lambda: tab_recognize(vid, cfg, t0, t1, rd), "Поза и трекинг": lambda: tab_pose(vid, cfg, t0, t1, rd),
          "Признаки": lambda: tab_features(vid, cfg, t0, t1, rd), "Циклы": lambda: tab_cycles(vid, cfg, t0, t1, rd), "Предмет": lambda: tab_object(vid, cfg, t0, t1, rd),
          "Видео-модели": lambda: tab_tube(vid, cfg, t0, t1, rd), "События": lambda: tab_events(vid, cfg, t0, t1, rd), "VLM": lambda: tab_vlm(vid, cfg, t0, t1, rd),
          "Метки циклов": lambda: tab_label(vid, cfg, t0, t1, rd), "Обучение": tab_train, "Бенчмарки": tab_bench, "Железо и веса": lambda: tab_overview(vid, cfg)}[stage]
    fn()
