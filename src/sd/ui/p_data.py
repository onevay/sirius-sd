"""Вкладка «Данные»: что за видео есть, что размечено, где разметка противоречит себе, версии разметки с откатом и решения оператора, которые ждут переноса в эталон.

Разделы:
  Каталог              — по папкам и по видео: длительность, разрешение, слабая метка, эталон, метки жестов, готовые запуски
  Проблемы             — конфликт метки папки и эталона, неразмеченные клипы, затяжки без эталона, дубликаты файлов, нечитаемые видео
  Версии разметки      — история файлов разметки, отличия двух версий, откат (откат тоже сохраняется версией)
  Метки циклов         — метки циклов (затяжка / не затяжка) из эталона событий по выбранному прогону
  Обратная связь       — решения оператора (подтверждено / ложное) → эталон: сколько ждёт переноса, перенос
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sd import data_catalog as DC
from sd import experiments as XP
from sd import labels_history as LH
from sd import library as LIB
from sd.paths import from_portable
from sd.ui import common as UC
from sd.ui import nav

SECTIONS = ["Каталог", "Проблемы", "Версии разметки", "Метки циклов", "Обратная связь оператора"]


@st.cache_data(show_spinner="читаем каталог видео…", ttl=60)
def _catalog(dirs: tuple, probe: bool, stamp: tuple) -> pd.DataFrame:
    return DC.videos_table(list(dirs), with_probe=probe)


def _dirs_pick(key: str) -> list[str]:
    dd = LIB.discover_dirs()
    if dd.empty:
        st.info("Видео не найдены: положите клипы в `data/` (или задайте SD_DATA).")
        return []
    names = dd.name.tolist()
    if key not in st.session_state:
        st.session_state[key] = [n for n in names if Path(n).name in ("курение", "лжекурение", "custom")] or names[:3]
    pick = st.multiselect("Папки", names, key=key)
    return dd.set_index("name").loc[pick, "path"].tolist() if pick else []


def _stamp() -> tuple:
    from sd import gt as GT
    from sd.start_eval import GT_CSV

    return (UC.file_mtime(GT.default_path()), UC.file_mtime(GT_CSV), UC.file_mtime(DC.PROBE_CACHE))


def _catalog_section() -> None:
    paths = _dirs_pick("data_dirs")
    if not paths:
        return
    probe = st.toggle("Читать свойства видео (длительность, разрешение) — первый раз медленно, дальше из кэша", value=True, key="data_probe")
    tab = _catalog(tuple(paths), probe, _stamp())
    if tab.empty:
        st.info("В выбранных папках нет видео.")
        return
    k = st.columns(5)
    k[0].metric("Видео", len(tab))
    k[1].metric("Минут", f"{tab.duration.fillna(0).sum() / 60:.0f}" if probe else "—")
    k[2].metric("Размечено", f"{int(tab.reviewed.sum())} из {len(tab)}")
    k[3].metric("Эпизодов курения", int(tab.pos.sum()))
    k[4].metric("Окон жестов (затяжек)", f"{int(tab.gestures.sum())} ({int(tab.smokes.sum())})")
    st.markdown("**По папкам**")
    st.dataframe(DC.coverage(tab), hide_index=True)
    st.markdown("**По видео**")
    show = tab.drop(columns=["path", "dup_key"]).rename(columns={"clip_id": "клип", "folder": "папка", "file": "файл", "size_mb": "МБ", "weak": "метка папки", "reviewed": "размечен",
                                                                 "pos": "POSITIVE", "neg": "NEGATIVE", "ign": "IGNORE", "gestures": "жестов", "smokes": "затяжек", "runs": "запусков",
                                                                 "duration": "длит., с", "fps": "к/с", "width": "ширина", "height": "высота", "readable": "читается"})
    sel = st.dataframe(show, hide_index=True, on_select="rerun", selection_mode="single-row", key="data_tbl", height=420)
    rows = sel["selection"]["rows"] if sel and sel.get("selection") else []
    if rows:
        r = tab.iloc[rows[0]]
        c = st.columns(3)
        if c[0].button(f"Разметить {r.clip_id}", key="data_to_lab"):
            nav.go("tasks", tasks_section="Разметка событий", lab_clip=r.clip_id, lab_dirs=[r.folder])
        if c[1].button("Метки жестов клипа", key="data_to_g"):
            nav.go("tasks", tasks_section="Метки жестов", an_g_v=[r.clip_id])


def _issues_section() -> None:
    paths = _dirs_pick("data_dirs_i")
    if not paths:
        return
    tab = _catalog(tuple(paths), False, _stamp())
    iss = DC.issues(tab)
    if iss.empty:
        st.success("Проблем не найдено.")
        return
    cnt = iss.groupby(["level", "code"]).agg(клипов=("clip_id", "nunique"), что=("text", "first")).reset_index()
    st.dataframe(cnt, hide_index=True)
    lv = st.multiselect("Уровни", ["error", "warning", "info"], default=["error", "warning"], key="data_lv")
    view = iss[iss.level.isin(lv)]
    sel = st.dataframe(view[["level", "clip_id", "code", "detail"]], hide_index=True, on_select="rerun", selection_mode="single-row", key="data_iss")
    rows = sel["selection"]["rows"] if sel and sel.get("selection") else []
    if rows:
        r = view.iloc[rows[0]]
        st.info(r.text)
        if st.button(f"Открыть разметку {r.clip_id}", key="data_iss_lab"):
            nav.go("tasks", tasks_section="Разметка событий", lab_clip=r.clip_id, lab_dirs=[r.folder])
    st.caption("Конфликт «папка ↔ эталон»: оценка по эталону верна, а оценка «по имени папки» и слабые метки обучения — нет. Дубликаты: один ролик в обучении и проверке завышает качество.")


def _versions_section() -> None:
    kind = st.radio("Файл", list(LH.TITLES), horizontal=True, format_func=LH.TITLES.get, key="lh_kind")
    path = LH.files()[kind]
    c = st.columns([3, 1])
    note = c[0].text_input("Комментарий к версии", key="lh_note", placeholder="что изменилось в разметке")
    if c[1].button("Зафиксировать текущую"):
        r = LH.snapshot(kind, UC.who(), note, "manual")
        st.success(f"версия {r['snapshot']}" if r else "содержимое не менялось с прошлой версии")
    v = LH.versions(kind)
    st.caption(f"{path} · {'есть' if path.exists() else 'нет файла'} · отпечаток содержимого {LH.content_sha(path) or '—'}")
    if v.empty:
        st.info("Версий ещё нет: они появляются при сохранении правок, импорте, перестройке эталона или кнопкой выше.")
        return
    st.dataframe(v[["ts", "action", "rows", "author", "note", "sha", "gt_fp", "snapshot"]].rename(columns={"ts": "время", "action": "действие", "rows": "строк", "author": "кто",
                                                                                                          "note": "комментарий", "gt_fp": "отпечаток эталона"}),
                 hide_index=True, height=260)
    st.caption("«отпечаток эталона» совпадает с `gt_fingerprint` прогонов оценки — по нему видно, на какой версии разметки получена цифра.")
    snaps = v.snapshot.tolist()
    c = st.columns(2)
    a = c[0].selectbox("Версия", snaps, key="lh_a")
    b = c[1].selectbox("Сравнить с", ["текущий файл", *snaps], key="lh_b")
    try:
        old = LH.read_version(kind, a)
        new = pd.read_csv(path, encoding="utf-8-sig") if b == "текущий файл" else LH.read_version(kind, b)
        d = LH.diff(kind, old, new)
        st.markdown(f"**{a} → {b}:** {d['summary']}")
        t1, t2, t3 = st.tabs([f"добавлено ({len(d['added'])})", f"удалено ({len(d['removed'])})", f"изменена метка ({len(d['changed'])})"])
        t1.dataframe(d["added"], hide_index=True)
        t2.dataframe(d["removed"], hide_index=True)
        t3.dataframe(d["changed"], hide_index=True)
    except Exception as e:      # noqa: BLE001
        st.error(f"версия не читается: {e}")
    UC.confirm(f"Откатить файл к версии {a}", "lh_restore", "Файл разметки будет заменён выбранной версией. Текущее содержимое сначала сохранится версией — откат можно отменить.",
               lambda: LH.restore(kind, a, UC.who()) and f"файл возвращён к {a}")


def _feedback_section() -> None:
    s = DC.feedback_status()
    if not s.get("ok"):
        st.info(f"Хранилище мониторинга недоступно ({s.get('error')}): решения оператора появляются после работы `sd monitor` / приложения оператора.")
        return
    k = st.columns(4)
    k[0].metric("Подтверждено оператором", s["confirmed"])
    k[1].metric("Ложных", s["false"])
    k[2].metric("Уже в эталоне", s["exported"])
    k[3].metric("Ждут переноса", s.get("pending", 0))
    st.caption("Подтверждённая тревога → POSITIVE (рамка и интервал из тревоги), ложная → NEGATIVE (жёсткий негатив). Перенос не дублирует строки (ключ — заметка alert:<id>); "
               "эталон перед переносом сохраняется версией. Перенесённое стоит выборочно проверить на «Разметке событий»: оператор мог ошибиться.")

    def _do():
        from sd.realtime.feedback import export_reviewed
        from sd.realtime.store import AlertStore

        LH.snapshot("events", UC.who(), "перед переносом решений оператора", "import", force=True)
        return export_reviewed(AlertStore())

    UC.confirm("Перенести решения в эталон", "fb_export", "Решения оператора будут добавлены в labels/events_gt.csv. Текущий эталон сохранится версией.", _do,
               disabled=not s.get("pending"))


def _cycle_labels() -> None:
    from sd import dataset as DS
    from sd import labels_from_gt as LG
    from sd.tracks import Tracks

    rid = UC.run_select("Прогон (по ручному эталону)", key="cl_run", mode="events")
    if not rid:
        return
    neg = st.checkbox("Циклы вне размеченных интервалов просмотренного клипа считать негативами", value=True,
                      help="отключите, если в клипах размечены не все курения: иначе неразмеченное курение станет ложным негативом")
    iou_th = st.slider("Порог IoU рамок", 0.1, 0.9, 0.3, 0.05)
    _, meta, _, gt = XP.load_run(rid)
    parts, missing, boxes = [], [], {}
    for clip in meta["clips"]:
        out, rd = from_portable(clip.get("out_dir")), from_portable(clip.get("run_dir"))
        f = out / "analysis" / "dataset_cycles.parquet" if out else None
        if not (f and f.exists() and rd and (rd / "pose" / "tracks.parquet").exists()):
            missing.append(clip["clip_id"])
            continue
        boxes[clip["clip_id"]] = Tracks.load(rd / "pose")
        parts.append(pd.read_parquet(f))
    if missing:
        st.caption(f"Без таблицы циклов или треков (кэш удалён или прогон без циклов): {len(missing)} клипов.")
    if not parts:
        return
    cycles = pd.concat(parts, ignore_index=True)
    labels = LG.cycle_labels_from_gt(cycles, gt, lambda v, tid, t: boxes[v].box_at(tid, t) if v in boxes else None, iou_th=iou_th, unlabeled_as_negative=neg)
    st.write(f"Циклов {len(cycles)}, получено меток {len(labels)}: {LG.summarize(labels)}")
    st.dataframe(labels.round(2).head(200), hide_index=True)

    def _do():
        from sd import labels_history as LH

        LH.snapshot("cycles", UC.who(), f"перед записью меток циклов из эталона ({rid})", "import", force=True)
        return f"записано: {DS.save_labels_bulk(labels)}; ручные метки не затронуты"

    UC.confirm("Записать в labels/cycle_labels.csv", "cl_save", "Прежние метки циклов из эталона (source=gt) будут заменены; ручные останутся. Текущий файл сохранится версией.", _do,
               disabled=labels.empty)


def render() -> None:
    UC.style()
    UC.header("Данные", "Что есть, что размечено и где разметка противоречит себе; версии разметки с откатом; решения оператора → эталон.")
    sec = UC.sections(SECTIONS, "data_section")
    {"Каталог": _catalog_section, "Проблемы": _issues_section, "Версии разметки": _versions_section, "Метки циклов": _cycle_labels,
     "Обратная связь оператора": _feedback_section}[sec]()
