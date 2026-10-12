"""Просмотр одного клипа прогона: видео с эталоном, событиями системы, рамками людей, циклами и ошибками (клавиши E / Shift+E — к следующей ошибке)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sd import viewdata as VD
from sd.paths import from_portable
from sd.ui.player import build_data, player_height, player_html


@st.cache_data(show_spinner=False)
def _tracks(pose_dir: str, mtime: float) -> dict:
    return VD.tracks_payload(pose_dir)


def show(rep, meta: dict, ev: pd.DataFrame, gt: pd.DataFrame, cid: str) -> None:
    clip = next((c for c in meta.get("clips", []) if c["clip_id"] == cid), None)
    if clip is None:
        st.warning("клипа нет в этом прогоне")
        return
    video = from_portable(clip["path"])
    if video is None or not video.exists():
        st.warning(f"видео не найдено: {video} (папка данных перенесена? задайте SD_DATA)")
        return
    from sd.proxy import ensure_proxy
    from sd.ui.media import media_src

    with st.spinner("готовим копию видео для браузера…"):
        prox = ensure_proxy(video)
    uri, warn = media_src(prox)
    if warn:
        st.warning(warn)
    tracks, src = {}, (0, 0)
    pose = from_portable(clip["run_dir"]) / "pose" if clip.get("run_dir") else None
    if pose and (pose / "tracks.parquet").exists():
        tp = _tracks(str(pose), (pose / "tracks.parquet").stat().st_mtime)
        tracks, src = tp["tracks"], (tp["src_w"], tp["src_h"])
    cy = pd.DataFrame()
    out_dir = from_portable(clip.get("out_dir"))
    if out_dir and (out_dir / "cycles_scored.parquet").exists():
        cy = pd.read_parquet(out_dir / "cycles_scored.parquet")
    th = rep.settings["threshold"]
    e = ev[(ev.clip_id == cid) & (ev.confidence >= th)] if len(ev) else ev
    data = build_data(cy, e, clip["duration"], 12.0, 0.0, gt=VD.gt_payload(gt, cid) if len(gt) else [], errors=VD.errors_payload(rep.errors, cid), tracks=tracks, src_size=src)
    data["offset"], data["duration"] = 0.0, float(max(clip["duration"] + (clip.get("window") or [0])[0], 1.0))
    components.html(player_html(None, data, video_uri=uri)[0], height=player_height(data), scrolling=False)
    if not tracks:
        st.caption("Рамок людей нет: кэш позы для этого прогона удалён.")
