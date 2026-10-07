"""Страница «Просмотр»: видео с эталоном, событиями системы, рамками людей и ошибками; переход между ошибками клавишами E / Shift+E."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sd import experiments as XP
from sd import pipeline as P
from sd import viewdata as VD
from sd.ui.player import build_data, player_html


@st.cache_data(show_spinner=False)
def _run(rid: str, mtime: float):
    return XP.load_run(rid)


@st.cache_data(show_spinner=False)
def _tracks(pose_dir: str, mtime: float) -> dict:
    return VD.tracks_payload(pose_dir)


def _metrics(rep) -> None:
    m, b = rep.metrics, rep.budget
    c = st.columns(6)
    c[0].metric("F1", f"{m['f1']:.3f}", f"{m['f1'] - b['target']:+.3f} к цели {b['target']:.2f}")
    c[1].metric("Precision", f"{m['precision']:.3f}")
    c[2].metric("Recall", f"{m['recall']:.3f}")
    c[3].metric("TP / FP / FN", f"{m['tp']} / {m['fp']} / {m['fn']}")
    c[4].metric("Ложных тревог/ч", "—" if m["fp_per_hour"] is None else f"{m['fp_per_hour']:.1f}")
    c[5].metric("Порог", f"{m['threshold']:.2f}")


def _experiment() -> None:
    runs = XP.list_runs()
    if runs.empty:
        st.info("Экспериментов пока нет: запустите оценку на странице «Оценка».")
        return
    ids = runs.id.tolist()
    want = st.session_state.get("view_run")
    rid = st.selectbox("Эксперимент", ids, index=ids.index(want) if want in ids else 0,
                       format_func=lambda i: f"{runs.set_index('id').loc[i, 'name']} · {runs.set_index('id').loc[i, 'created']} · F1 {runs.set_index('id').loc[i, 'f1']:.3f}")
    rep, meta, ev, gt = _run(rid, (XP.EXP_DIR / rid / "meta.json").stat().st_mtime)
    _metrics(rep)
    clips = meta["clips"]
    err_n = rep.errors.groupby("clip_id").size().to_dict() if len(rep.errors) else {}
    only_err = st.checkbox("Только клипы с ошибками", value=bool(err_n))
    shown = [c for c in clips if not only_err or err_n.get(c["clip_id"])] or clips
    shown = sorted(shown, key=lambda c: (-err_n.get(c["clip_id"], 0), c["clip_id"]))
    want_c = st.session_state.get("view_clip")
    idx = next((i for i, c in enumerate(shown) if c["clip_id"] == want_c), 0)
    clip = shown[st.selectbox("Клип", range(len(shown)), index=idx, format_func=lambda i: f"{shown[i]['clip_id']} · ошибок {err_n.get(shown[i]['clip_id'], 0)}")]
    cid, video = clip["clip_id"], Path(clip["path"])
    if not video.exists():
        st.warning(f"Видео не найдено: {video}")
        return
    from sd.proxy import ensure_proxy

    with st.spinner("готовим копию видео…"):
        prox = ensure_proxy(video)
    from sd.ui.labeler import video_data_uri

    uri, warn = video_data_uri(prox)
    if warn:
        st.warning(warn)
    tracks, src = {}, (0, 0)
    pose = Path(clip["run_dir"]) / "pose" if clip.get("run_dir") else None
    if pose and (pose / "tracks.parquet").exists():
        tp = _tracks(str(pose), (pose / "tracks.parquet").stat().st_mtime)
        tracks, src = tp["tracks"], (tp["src_w"], tp["src_h"])
    cy = pd.DataFrame()
    if clip.get("out_dir") and (Path(clip["out_dir"]) / "cycles_scored.parquet").exists():
        cy = pd.read_parquet(Path(clip["out_dir"]) / "cycles_scored.parquet")
    th = rep.settings["threshold"]
    e = ev[(ev.clip_id == cid) & (ev.confidence >= th)] if len(ev) else ev
    data = build_data(cy, e, clip["duration"], 12.0, 0.0, gt=VD.gt_payload(gt, cid) if len(gt) else [], errors=VD.errors_payload(rep.errors, cid), tracks=tracks, src_size=src)
    data["offset"], data["duration"] = 0.0, float(max(clip["duration"] + clip["window"][0], 1.0))
    html = player_html(None, data, video_uri=uri)[0]
    components.html(html, height=680, scrolling=False)
    if not tracks:
        st.caption("Рамок людей нет: для этого прогона не сохранены треки (кэш позы удалён).")
    errs = rep.errors[rep.errors.clip_id == cid] if len(rep.errors) else rep.errors
    if len(errs):
        st.dataframe(errs[["kind", "start", "end", "confidence", "cause_ru"]].rename(columns={"kind": "тип", "start": "начало", "end": "конец", "confidence": "confidence", "cause_ru": "причина"})
                     .round(2), hide_index=True)


def _run_dir() -> None:
    res = P.list_results()
    if res.empty:
        st.info("Отдельных прогонов нет: `sd recognize` или страница «Этапы» → «Распознавание окна».")
        return
    default = st.session_state.get("player_dir")
    idx = int(res.index[res.dir == default][0]) if default in set(res.dir) else 0
    k = st.selectbox("Прогон", res.index, index=idx, format_func=lambda i: f"{res['clip'][i]} · {res.window[i]} · {res.created[i]} · циклов {res.cycles[i]}, событий {res.events[i]}")
    d = Path(res.dir[k])
    meta = json.loads((d / "result.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(d / "events.csv") if (d / "events.csv").exists() else pd.DataFrame()
    cy = pd.read_parquet(d / "cycles_scored.parquet") if (d / "cycles_scored.parquet").exists() else pd.DataFrame()
    w0, w1 = float(meta["window"][0] or 0.0), meta["window"][1]
    video = d / "overlay.mp4" if (d / "overlay.mp4").exists() else None
    data = build_data(cy, ev, 0.0, 15.0, w0)
    data["duration"] = float((w1 - w0) if w1 is not None else (cy["end"].max() + 5 - w0 if len(cy) else 30.0))
    html, warn = player_html(video, data)
    if video is None:
        st.warning("В этом прогоне нет overlay.mp4 (запуск без видео).")
    if warn:
        st.warning(warn)
    components.html(html, height=720, scrolling=False)
    if len(ev):
        st.dataframe(ev, hide_index=True)
    with st.expander("Что запускалось"):
        st.json(meta)


def render() -> None:
    src = st.radio("Источник", ["Эксперимент", "Отдельный прогон"], horizontal=True, label_visibility="collapsed", key="view_src")
    _experiment() if src == "Эксперимент" else _run_dir()
