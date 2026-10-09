"""Страница «Разметка»: эталон событий на видео — интервал, человек, метка. Результат — `labels/events_gt.csv` (формат организаторов), по нему считается оценка."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from sd import gt as GT
from sd import library as LIB
from sd import viewdata as VD
from sd.paths import OUTPUTS
from sd.ui.labeler import labeler
from sd.ui.media import media_src


@st.cache_data(show_spinner=False)
def _probe(path: str, mtime: float) -> dict:
    from sd.video_io import probe

    i = probe(path)
    return dict(w=i.width, h=i.height, dur=i.duration, fps=i.fps)


@st.cache_data(show_spinner=False)
def _tracks(pose_dir: str, mtime: float) -> dict:
    return VD.tracks_payload(pose_dir)


def _proxy(video) -> str:
    from sd.proxy import ensure_proxy

    return str(ensure_proxy(video, codec=os.environ.get("SD_PROXY_CODEC", "h264")))


def _apply(act: dict, cid: str, video, who: str) -> None:
    cam = video.parent.name
    try:
        if act["action"] in ("add", "update"):
            i = act["interval"]
            kw = dict(person=i.get("person") or "", peak=i.get("peak"), box=i.get("box"), note=i.get("note") or "", camera_id=cam, labeler=who)
            if act["action"] == "add":
                GT.add(cid, i["start"], i["end"], i["label"], **kw)
            else:   # правка: проверяем те же правила, что при добавлении, затем заменяем строку
                new = GT.add(cid, i["start"], i["end"], i["label"], **kw)
                GT.delete([i["id"]])
                st.session_state["lab_last"] = new
        elif act["action"] == "delete":
            GT.delete([act["id"]])
        elif act["action"] == "clean":
            GT.mark_clean(cid, _probe(str(video), video.stat().st_mtime)["dur"], camera_id=cam, labeler=who)
    except ValueError as e:
        st.session_state["lab_error"] = str(e)


def render() -> None:
    gt = GT.load()
    dirs_df = LIB.discover_dirs()
    if dirs_df.empty:
        st.info("Видео не найдены: положите клипы в `data/` или `data_external/`.")
        return
    c1, c2, c3 = st.columns([3, 3, 1.4])
    names = dirs_df.name.tolist()
    pick = c1.multiselect("Папки", names, default=st.session_state.get("lab_dirs") or names[:1], key="lab_dirs")
    if not pick:
        st.info("Выберите хотя бы одну папку.")
        return
    paths = dirs_df.set_index("name").loc[pick, "path"].tolist()
    try:
        ids = LIB.clip_ids(LIB.videos_in(paths))
    except ValueError as e:
        st.error(str(e))
        return
    status = GT.clip_status(gt, ids).set_index("clip_id")
    todo = [c for c in ids if not status.loc[c, "reviewed"]]
    label = lambda c: f"{'готов' if status.loc[c, 'reviewed'] else 'нет'} · {c}" + (f" · {status.loc[c, 'positive']} пол." if status.loc[c, "positive"] else "")   # noqa: E731
    if st.session_state.get("lab_clip") not in ids:
        st.session_state["lab_clip"] = todo[0] if todo else next(iter(ids))
    cid = c2.selectbox("Клип", list(ids), format_func=label, key="lab_clip")
    who = c3.text_input("Разметчик", key="lab_who", placeholder="инициалы")
    nxt = st.columns([1, 1, 6])
    if nxt[0].button("Следующий неразмеченный", disabled=not todo):
        st.session_state["lab_clip"] = todo[0]
        st.rerun()
    nxt[1].caption(f"размечено {len(ids) - len(todo)} из {len(ids)}")
    video = ids[cid]
    info = _probe(str(video), video.stat().st_mtime)
    with st.spinner("готовим копию видео для просмотра…"):
        prox = _proxy(video)
    uri, warn = media_src(prox)
    if warn:
        st.warning(warn)
    runs = VD.find_pose_runs(cid)
    tracks, src = {}, (info["w"], info["h"])
    if runs:
        run = st.selectbox("Подсказки рамок: треки", runs, format_func=lambda p: p.parent.name, help="клик по рамке в кадре выбирает человека; без треков рамку можно нарисовать мышью")
        tp = _tracks(str(run), (run / "tracks.parquet").stat().st_mtime)
        tracks = tp["tracks"]
        src = (tp["src_w"] or info["w"], tp["src_h"] or info["h"])
    else:
        st.caption("Треков для клипа нет — рамку рисуйте мышью.")
    act = labeler(key=f"lab_{cid}", video=uri, video_key=f"{cid}:{prox}", duration=info["dur"], fps=12.0, src_size=src, intervals=VD.gt_payload(gt, cid), tracks=tracks, clip_id=cid)
    nonce_key = f"lab_nonce_{cid}"
    if act and act.get("nonce") != st.session_state.get(nonce_key):
        st.session_state[nonce_key] = act["nonce"]
        _apply(act, cid, video, who)
        st.rerun()
    err = st.session_state.pop("lab_error", None)
    if err:
        st.error(err)

    st.divider()
    cur = GT.for_clip(gt, cid)
    for w in GT.validate(cur):
        st.warning(w)
    with st.expander(f"Таблица интервалов клипа ({len(cur)})", expanded=False):
        st.dataframe(cur.drop(columns=["id", "clip_id", "ts"]), hide_index=True)
    with st.expander("Файл эталона: экспорт и импорт"):
        st.caption(f"{GT.default_path()} · строк {len(gt)} · отпечаток {GT.fingerprint(gt) if len(gt) else '—'}")
        out = GT.export_organizer(gt)
        st.download_button("Скачать в формате организаторов", out.to_csv(index=False).encode("utf-8-sig"), file_name="events_gt_organizer.csv", disabled=out.empty)
        up = st.file_uploader("Импорт CSV (clip_id, start_sec, end_sec, label, …)", type=["csv"], key="gt_up")
        repl = st.checkbox("Заменить прежнюю разметку клипов из файла", value=False)
        if up is not None and st.button("Импортировать"):
            try:
                n = GT.import_rows(pd.read_csv(up, encoding="utf-8-sig"), replace_clips=repl)
                st.success(f"добавлено строк: {n}")
            except Exception as e:
                st.error(str(e))
