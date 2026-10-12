"""Вкладка «Задачи»: всё, что улучшает модель и пополняет данные, — от разметки до обученного пакета.

Разделы (порядок = рабочий процесс):
  Разметка событий — эталон на видео (интервал, человек, рамка); подсказки: затяжки из меток жестов, треки людей; версия файла перед удалением
  Метки жестов     — таблица кандидатов (затяжка / питьё / телефон …) с проверкой и версией; перестройка эталона из жестов
  Признаки         — источник циклов (прогон оценки) → расчёт признаков предмета/фото/VLM фоновой задачей; заполненность и сила признаков
  Обучение         — спецификация классификатора (признаки, члены, CV), предупреждения, диагностика и обучение фоновыми задачами, их результаты
  Результаты       — реестр пакетов (источник, видео обучения, AUC), честная вложенная оценка наборов признаков
  Очередь задач    — все фоновые задачи; запуск любой задачи каталога вручную
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sd import classifier as C
from sd import cycle_models as CM
from sd import gestures as G
from sd import gt as GT
from sd import labels_history as LH
from sd import library as LIB
from sd import oof_eval as O
from sd import profiles as PR
from sd import tasks as T
from sd import viewdata as VD
from sd.paths import CYCLE_MODELS, OUTPUTS
from sd.ui import common as UC

SECTIONS = ["Разметка событий", "Метки жестов", "Признаки", "Обучение", "Результаты", "Очередь задач"]
CLF_OUT = OUTPUTS / "classifier"


# ============================================================================================== разметка событий
@st.cache_data(show_spinner=False)
def _probe(path: str, mtime: float) -> dict:
    from sd.video_io import probe

    i = probe(path)
    return dict(w=i.width, h=i.height, dur=i.duration, fps=i.fps)


@st.cache_data(show_spinner=False)
def _tracks(pose_dir: str, mtime: float) -> dict:
    return VD.tracks_payload(pose_dir)


def _apply(act: dict, cid: str, video: Path, who: str) -> None:
    cam = video.parent.name
    try:
        if act["action"] in ("add", "update"):
            i = act["interval"]
            kw = dict(person=i.get("person") or "", peak=i.get("peak"), box=i.get("box"), note=i.get("note") or "", camera_id=cam, labeler=who)
            if act["action"] == "add":
                GT.add(cid, i["start"], i["end"], i["label"], **kw)
            else:
                LH.snapshot("events", who, f"перед правкой интервала {i['id']} ({cid})", "auto")
                new = GT.add(cid, i["start"], i["end"], i["label"], **kw)
                GT.delete([i["id"]])
                st.session_state["lab_last"] = new
        elif act["action"] == "delete":
            LH.snapshot("events", who, f"перед удалением интервала {act['id']} ({cid})", "auto")
            GT.delete([act["id"]])
        elif act["action"] == "clean":
            GT.mark_clean(cid, _probe(str(video), video.stat().st_mtime)["dur"], camera_id=cam, labeler=who)
    except ValueError as e:
        st.session_state["lab_error"] = str(e)


def _labeling() -> None:
    from sd.ui.labeler import labeler
    from sd.ui.media import media_src

    gt = GT.load()
    dirs_df = LIB.discover_dirs()
    if dirs_df.empty:
        st.info("Видео не найдены: положите клипы в `data/`.")
        return
    names = dirs_df.name.tolist()
    c1, c2 = st.columns([3, 4])
    if "lab_dirs" not in st.session_state:
        st.session_state["lab_dirs"] = names[:1]
    pick = c1.multiselect("Папки", names, key="lab_dirs")
    if not pick:
        st.info("Выберите хотя бы одну папку.")
        return
    try:
        ids = LIB.clip_ids(LIB.videos_in(dirs_df.set_index("name").loc[pick, "path"].tolist()))
    except ValueError as e:
        st.error(str(e))
        return
    status = GT.clip_status(gt, ids).set_index("clip_id")
    todo = [c for c in ids if not status.loc[c, "reviewed"]]
    if st.session_state.get("lab_clip") not in ids:
        st.session_state["lab_clip"] = todo[0] if todo else next(iter(ids))
    lab = lambda c: f"{'✓' if status.loc[c, 'reviewed'] else '○'} {c}" + (f" · {status.loc[c, 'positive']} кур." if status.loc[c, "positive"] else "")   # noqa: E731
    cid = c2.selectbox("Клип", list(ids), format_func=lab, key="lab_clip")
    who = UC.who()
    n = st.columns([2, 2, 4])
    if n[0].button("Следующий неразмеченный ▶", disabled=not todo):
        st.session_state["lab_clip"] = todo[0]
        st.rerun()
    n[1].caption(f"размечено {len(ids) - len(todo)} из {len(ids)}")
    if not who:
        n[2].warning("Укажите имя в боковой панели — оно пишется в разметку и версии.")
    video = ids[cid]
    info = _probe(str(video), video.stat().st_mtime)
    from sd.proxy import ensure_proxy

    with st.spinner("готовим копию видео для просмотра…"):
        prox = str(ensure_proxy(video, codec=os.environ.get("SD_PROXY_CODEC", "h264")))
    uri, warn = media_src(prox)
    if warn:
        st.warning(warn)
    runs = VD.find_pose_runs(cid)
    tracks, src = {}, (info["w"], info["h"])
    if runs:
        run = st.selectbox("Треки людей (подсказки рамок; клик по рамке выбирает человека)", runs, format_func=lambda p: p.parent.name, key="lab_tr")
        tp = _tracks(str(run), (run / "tracks.parquet").stat().st_mtime)
        tracks, src = tp["tracks"], (tp["src_w"] or info["w"], tp["src_h"] or info["h"])
    else:
        st.caption("Треков для клипа нет — рамку рисуйте мышью (или запустите оценку, чтобы посчитать позу).")
    gest = G.load()
    smokes = gest[(gest.video == cid) & (gest.label == "smoke")].peak_t.astype(float).sort_values()
    if len(smokes):
        st.caption("Подсказка — затяжки по меткам жестов: " + ", ".join(f"{t:.1f} с" for t in smokes.head(20)) + (" …" if len(smokes) > 20 else ""))
    act = labeler(key=f"lab_{cid}", video=uri, video_key=f"{cid}:{prox}", duration=info["dur"], fps=12.0, src_size=src, intervals=VD.gt_payload(gt, cid), tracks=tracks, clip_id=cid)
    nk = f"lab_nonce_{cid}"
    if act and act.get("nonce") != st.session_state.get(nk):
        st.session_state[nk] = act["nonce"]
        _apply(act, cid, video, who)
        st.rerun()
    err = st.session_state.pop("lab_error", None)
    if err:
        st.error(err)
    cur = GT.for_clip(gt, cid)
    w = GT.validate(cur)
    weak = "smoking" if video.parent.name == "курение" else ("fake" if video.parent.name == "лжекурение" else None)
    if weak == "fake" and (cur.label == "POSITIVE").any():
        w.append("клип лежит в папке «лжекурение», а размечено курение — проверьте файл; эталон главнее имени папки")
    if len(smokes) and not len(cur):
        w.append(f"в метках жестов {len(smokes)} затяжек, а эталона событий нет — разметьте или постройте эталон из жестов («Метки жестов»)")
    for x in w:
        st.warning(x)
    with st.expander(f"Интервалы клипа ({len(cur)})"):
        st.dataframe(cur.drop(columns=["id", "clip_id", "ts"]), hide_index=True)
    with st.expander("Версия эталона, экспорт и импорт"):
        c = st.columns([3, 1])
        note = c[0].text_input("Комментарий к версии", key="lab_note", placeholder="например: разметил sm_6 и noyabrsk_3")
        if c[1].button("Зафиксировать версию"):
            r = LH.snapshot("events", who, note, "manual")
            st.success(f"версия сохранена: {r['snapshot']}" if r else "изменений с прошлой версии нет")
        st.caption(f"{GT.default_path()} · строк {len(gt)} · отпечаток {GT.fingerprint(gt) if len(gt) else '—'}. История и откат — вкладка «Данные → Версии разметки».")
        out = GT.export_organizer(gt)
        st.download_button("Скачать в формате организаторов", out.to_csv(index=False).encode("utf-8-sig"), file_name="events_gt_organizer.csv", disabled=out.empty)
        up = st.file_uploader("Импорт CSV (clip_id, start_sec, end_sec, label, …)", type=["csv"], key="gt_up")
        repl = st.checkbox("Заменить прежнюю разметку клипов из файла", value=False, key="gt_up_repl")
        if up is not None:
            def _do():
                LH.snapshot("events", who, f"перед импортом {up.name}", "import", force=True)
                return f"добавлено строк: {GT.import_rows(pd.read_csv(up, encoding='utf-8-sig'), replace_clips=repl)}"
            UC.confirm("Импортировать", "gt_imp", ("Строки этих клипов будут ЗАМЕНЕНЫ. " if repl else "") + "Текущий эталон сохранится версией (откат — «Данные → Версии разметки»).", _do)


# ============================================================================================== метки жестов
def _gestures() -> None:
    df = G.load()
    c1, c2, c3 = st.columns([3, 3, 2])
    vids = sorted(df.video.dropna().unique())
    pick_v = c1.multiselect("Видео", vids, key="an_g_v")
    pick_l = c2.multiselect("Метка", G.LABELS, key="an_g_l", format_func=lambda x: f"{x} — {G.LABEL_RU.get(x, x)}")
    only_low = c3.checkbox("только «неясно»", key="an_g_u")
    view = df
    if pick_v:
        view = view[view.video.isin(pick_v)]
    if pick_l:
        view = view[view.label.isin(pick_l)]
    if only_low:
        view = view[view.label == "unsure"]
    st.caption(f"{G.path()} · строк {len(df)}, показано {len(view)}. Исправляйте метки, с которыми не согласны; «неясно» не идёт в обучение.")
    edited = st.data_editor(view[G.COLS], num_rows="dynamic", hide_index=False, width="stretch", key="an_g_editor",
                            column_config={"label": st.column_config.SelectboxColumn("метка", options=G.LABELS, required=True),
                                           "start_quality": st.column_config.SelectboxColumn("старт", options=["ok", "late", "early"]),
                                           "peak_t": st.column_config.NumberColumn("пик, с", format="%.2f"), "start": st.column_config.NumberColumn("начало, с", format="%.2f"),
                                           "tid": st.column_config.NumberColumn("трек", step=1)})
    out = pd.concat([df.drop(index=view.index), edited], ignore_index=True)
    changed = not edited.equals(view[G.COLS])
    issues = G.validate(out)
    if changed:
        d = LH.diff("gestures", df, out)
        st.info(f"Несохранённые изменения: {d['summary']}")
        UC.checklist(issues)
    note = st.text_input("Комментарий к правке", key="an_g_note", placeholder="что и почему исправлено")
    if st.button("Сохранить метки жестов", type="primary", disabled=not changed or any(l_ == "error" for l_, _ in issues)):
        try:
            snap = G.save(out, UC.who(), note)
            st.success(f"сохранено ({len(out)} строк); прежняя версия: {snap['snapshot'] if snap else '—'}")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    with st.expander("Сводка по видео"):
        st.dataframe(G.summary(df), hide_index=True)
    st.divider()
    st.markdown("**Эталон событий из меток жестов** — затяжки одного человека (разрыв ≤ 15 с, ≥ 2) → POSITIVE, одиночные → IGNORE, клип без курения → NEGATIVE. "
                "Строки разметчика `assistant` заменяются, ручная разметка остаётся; прежний эталон сохраняется версией.")
    UC.task_button("gt_from_gestures", dict(labeler="assistant"), "Перестроить эталон (фон)", key="g_rebuild", primary=False)
    UC.task_panel(("gt_from_gestures",), key="g_tp", limit=2, title=None)


# ============================================================================================== источник таблицы циклов
@st.cache_data(show_spinner=False, ttl=30)
def _sources() -> list[dict]:
    return O.runs_with_tables()


def _source_select(key: str) -> str | None:
    rows = _sources()
    if not rows:
        st.warning("Нет прогонов оценки с запусками по полным клипам — запустите оценку («Валидация → Запуск»). Пока используются последние запуски.")
        return None
    lab = {r["id"]: f"{r['created']} · {r['name']} · {r['clips']} клипов · {r['fps']} к/с · {r['cycle_model']}" for r in rows}
    ids = [r["id"] for r in rows]
    return st.selectbox("Источник циклов: прогон оценки (одна конфигурация позы и автомата)", ids, key=key, format_func=lab.get,
                        help="таблица собирается из запусков распознавания именно этого прогона: результат воспроизводим и не смешивает 5 и 10 к/с")


def _stamp() -> tuple:
    from sd import enrich as EN
    from sd.start_eval import GT_CSV

    return (UC.file_mtime(GT_CSV), UC.file_mtime(EN.ROOT), st.session_state.get("tbl_refresh", 0))


@st.cache_data(show_spinner="собираем таблицу циклов…")
def _table(run: str | None, enriched: bool, stamp: tuple) -> pd.DataFrame:
    return C.load_table(enriched, run=run)


# ============================================================================================== признаки
def _features() -> None:
    from sd import catalog as CAT
    from sd.feature_auc import _auc

    run = _source_select("feat_run")
    st.markdown("##### Посчитать признаки цикла (фоновая задача, кэш outputs/enrich)")
    c = st.columns(3)
    dets = [d.id for d in CAT.detector_models()]
    objs = c[0].multiselect("Детекторы предмета на кропе рта", dets, default=[d for d in dets if "beehzod" in d and "smoke_cig" not in d][:1], key="feat_obj",
                            help="≈ 1.8 с на цикл на iGPU; obj_any_* — максимум по выбранным")
    photos = [b.id for b in CAT.photo_bundles()]
    pb = c[1].selectbox("Фото-модель (zero-shot CLIP и photo_*)", ["—", *photos], index=1 if photos else 0, key="feat_pb", format_func=lambda p: Path(p).name if p != "—" else "— не считать —")
    vl = c[2].text_input("VLM (тег Ollama, пусто — без VLM)", key="feat_vlm", help="≈ 11 с на цикл; сервер: `sd ollama up --igpu`")
    clips = st.text_input("Только клипы (через запятую; пусто — все клипы источника)", key="feat_clips")
    items = []
    if not objs and pb == "—" and not vl:
        items.append(("error", "ничего не выбрано для расчёта"))
    if vl:
        items.append(("warning", "VLM медленная (секунды на цикл) и не даёт выигрыша вне шума в прошлых замерах; проверьте, что Ollama запущен"))
    ok = UC.checklist(items) if items else True
    UC.task_button("enrich", dict(run=run or "", objects=",".join(objs), photo_bundle="" if pb == "—" else pb, vlm=vl.strip(), clips=clips.strip()), "Посчитать признаки (фон)",
                   key="feat_go", disabled=not ok)
    UC.task_panel(("enrich",), key="feat_tp", limit=2, title=None)
    st.divider()
    st.markdown("##### Таблица циклов этого источника: заполненность и сила признаков")
    c = st.columns([1, 5])
    if c[0].button("Обновить", key="feat_refresh"):
        st.session_state["tbl_refresh"] = st.session_state.get("tbl_refresh", 0) + 1
    try:
        tab = _table(run, True, _stamp())
    except Exception as e:      # noqa: BLE001
        st.error(f"таблица не собрана: {e}")
        return
    lab = tab[tab.y.notna()]
    k = st.columns(4)
    k[0].metric("Циклов", len(tab))
    k[1].metric("Размечено", len(lab))
    k[2].metric("Затяжек / не затяжек", f"{int((lab.y == 1).sum())} / {int((lab.y == 0).sum())}")
    k[3].metric("Видео", tab.video.nunique())
    groups = dict(кинематика=CM.KIN, ритм=CM.RHYTHM, поза=CM.POSE, предмет=CM.OBJ, фото=CM.PHOTO, VLM=CM.VLM, видео=CM.VIDEO)
    rows = []
    y = lab.y.to_numpy(float)
    for g, cols in groups.items():
        for col in cols:
            if col not in tab:
                rows.append(dict(группа=g, признак=col, заполнено=0.0, AUC=np.nan))
                continue
            a = _auc(y, lab[col].to_numpy(float)) if len(lab) else np.nan
            rows.append(dict(группа=g, признак=col, заполнено=float(tab[col].notna().mean()), AUC=a, **{"сила": abs(a - 0.5) * 2 if np.isfinite(a) else np.nan}))
    fd = pd.DataFrame(rows).sort_values("сила", ascending=False, na_position="last")
    st.dataframe(fd, hide_index=True, height=360, column_config={"заполнено": st.column_config.ProgressColumn("заполнено", min_value=0, max_value=1, format="%.0f%%"),
                                                                 "AUC": st.column_config.NumberColumn("AUC (одиночный)", format="%.3f"),
                                                                 "сила": st.column_config.ProgressColumn("сила |AUC−0.5|·2", min_value=0, max_value=1, format="%.2f")})
    st.caption("AUC одиночного признака по размеченным циклам (без кросс-валидации): ориентир, а не оценка модели. Признаки с заполненностью < 50 % — посчитайте их выше для этого источника.")


# ============================================================================================== обучение
def _members_editor(spec: dict, tab: pd.DataFrame) -> dict:
    cols = [c for c in tab.columns if c not in CM.CONFOUNDS and pd.api.types.is_numeric_dtype(tab[c]) and c not in ("y", "peak_t", "tid", "start", "end")]
    filled = {c: float(tab[c].notna().mean()) for c in cols}
    members = {}
    for n, m in list(spec["members"].items()):
        with st.expander(f"{n} — {C.KINDS.get(m['kind'], m['kind'])}", expanded=False):
            on = st.checkbox("включён", True, key=f"clf_on_{n}")
            kind = st.selectbox("тип", list(C.KINDS), index=list(C.KINDS).index(m["kind"]) if m["kind"] in C.KINDS else 0, key=f"clf_kind_{n}", format_func=lambda k: C.KINDS[k])
            feats = st.multiselect("признаки", sorted(set(cols) | set(m["features"])), default=list(m["features"]), key=f"clf_f_{n}",
                                   format_func=lambda c: f"{c} ({filled.get(c, 0):.0%})")
            params = dict(C.PARAMS[kind])
            params.update({k: v for k, v in (m.get("params") or {}).items() if k in params} if kind == m["kind"] else {})
            new = {}
            pc = st.columns(max(len(params), 1))
            for i, (k, v) in enumerate(params.items()):
                if isinstance(C.PARAMS[kind][k], int) and not isinstance(C.PARAMS[kind][k], bool):
                    new[k] = int(pc[i].number_input(k, 1, 1000, int(v), key=f"clf_p_{n}_{k}"))
                else:
                    new[k] = float(pc[i].number_input(k, 0.0, 1000.0, float(v), format="%.4g", key=f"clf_p_{n}_{k}"))
            if on:
                members[n] = dict(kind=kind, features=feats, params=new)
    return members


def _spec_block(tab: pd.DataFrame) -> dict:
    saved = C.list_specs()
    c1, c2 = st.columns(2)
    src = c1.selectbox("Спецификация", ["новая из набора признаков", *saved], key="clf_src")
    if src == "новая из набора признаков":
        sname = c2.selectbox("Набор признаков", list(CM.SETS), index=list(CM.SETS).index("obj_hold_zsd"), key="clf_set")
        kinds = st.multiselect("Члены ансамбля", ["lr", "gb", "nn"], default=["lr", "gb"], key="clf_kinds", format_func=lambda k: {"lr": "логрегрессия", "gb": "LightGBM", "nn": "MLP"}[k])
        base = C.default_spec(sname, kinds, tab)
        ident = f"new_{sname}_{'-'.join(kinds)}"
    else:
        base = C.load_spec(src)
        ident = f"saved_{src}"
    st.session_state.setdefault("clf_ident", ident)
    if st.session_state["clf_ident"] != ident:       # смена источника спецификации сбрасывает значения виджетов членов
        for k in [k for k in st.session_state if str(k).startswith(("clf_on_", "clf_kind_", "clf_f_", "clf_p_"))]:
            del st.session_state[k]
        st.session_state["clf_ident"] = ident
        st.rerun()
    members = _members_editor(base, tab)
    cv = {**C.DEFAULT_CV, **(base.get("cv") or {})}
    a, b, c, d = st.columns(4)
    n_splits = int(a.number_input("Фолдов", 2, 10, int(cv["n_splits"]), key="clf_ns"))
    repeats = int(b.number_input("Повторов", 1, 10, int(cv["repeats"]), key="clf_rep"))
    by_scene = c.checkbox("Фолды по сценам", bool(cv["by_scene"]), key="clf_scene", help="клипы одной камеры/человека не попадают одновременно в обучение и проверку — строже")
    seed = int(d.number_input("Seed", 0, 10_000, int(cv["seed"]), key="clf_seed"))
    calibrate = st.checkbox("Изотоническая калибровка", bool(base.get("calibrate", False)), key="clf_cal", help="на малой выборке сама переобучается; MVP обучен без неё")
    nm = st.text_input("Имя спецификации", (base.get("name") if src != "новая из набора признаков" else "") or "", key="clf_name", placeholder="например obj_hold_strict")
    return dict(name=nm.strip(), description=base.get("description", ""), set=base.get("set"), features=sorted({f for m in members.values() for f in m["features"]}),
                members=members, cv=dict(n_splits=n_splits, repeats=repeats, by_scene=by_scene, seed=seed), calibrate=calibrate)


def _diags(spec_name: str) -> list[Path]:
    return sorted((d for d in CLF_OUT.glob(f"*_{spec_name}") if (d / "diag.json").exists()), reverse=True) if CLF_OUT.exists() and spec_name else []


def _show_diag(d: Path) -> dict:
    j = json.loads((d / "diag.json").read_text(encoding="utf-8"))
    rd = lambda n: pd.read_csv(d / n, encoding="utf-8-sig") if (d / n).exists() and (d / n).stat().st_size > 3 else pd.DataFrame()   # noqa: E731
    members, curve, imp = rd("members.csv"), rd("curve.csv"), rd("importance.csv")
    st.caption(f"диагностика {d.name} · источник: {(j.get('source') or {}).get('run') or 'последние запуски'} · циклов {j['n']}, затяжек {j['n_pos']}, групп {j['n_groups']}")
    for lvl, t in j.get("warnings", []):
        st.markdown(f"{UC.LEVEL_ICON.get(lvl, '•')} {t}")
    for v in j.get("verdict", []):
        st.markdown(f"• {v}")
    st.dataframe(members, hide_index=True)
    if j.get("perm_auc"):
        st.caption(f"AUC при перемешанных метках (должно быть ≈ 0.5): {j['perm_mean']:.2f}")
    c1, c2 = st.columns(2)
    if len(curve):
        f = go.Figure(go.Scatter(x=curve["доля_видео"], y=curve["AUC_oof"], mode="lines+markers"))
        f.update_layout(title="Кривая обучения (растёт — данных мало)", xaxis_title="доля обучающих видео", yaxis_title="AUC oof", height=280, margin=dict(l=10, r=10, t=40, b=10))
        c1.plotly_chart(f, width="stretch", key=f"dg_c_{d.name}")
    if len(imp):
        f = go.Figure(go.Bar(x=imp["падение_AUC"], y=imp["признак"], orientation="h"))
        f.update_layout(title="Важность: падение AUC при перемешивании", height=280, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
        c2.plotly_chart(f, width="stretch", key=f"dg_i_{d.name}")
    return j


def _training() -> None:
    run = _source_select("clf_run")
    try:
        tab = _table(run, True, _stamp())
    except Exception as e:      # noqa: BLE001
        st.error(f"таблица циклов не собрана: {e}")
        return
    lab = tab[tab.y.notna()].reset_index(drop=True)
    lab.attrs["source"] = tab.attrs.get("source")
    k = st.columns(4)
    k[0].metric("Размечено циклов", len(lab), help=f"всего циклов {len(tab)}")
    k[1].metric("Затяжек", int((lab.y == 1).sum()))
    k[2].metric("Не затяжек", int((lab.y == 0).sum()))
    k[3].metric("Видео", lab.video.nunique())
    st.markdown("##### 1. Спецификация")
    spec = _spec_block(lab)
    st.markdown("##### 2. Проверка")
    diag_dirs = _diags(spec["name"])
    last_diag = None
    if diag_dirs:
        j = json.loads((diag_dirs[0] / "diag.json").read_text(encoding="utf-8"))
        same = j.get("spec_full", {}).get("members") == spec["members"] and (j.get("source") or {}).get("run") == (run or None)
        if same:
            m = pd.read_csv(diag_dirs[0] / "members.csv", encoding="utf-8-sig")
            imp = pd.read_csv(diag_dirs[0] / "importance.csv", encoding="utf-8-sig") if (diag_dirs[0] / "importance.csv").stat().st_size > 3 else None
            last_diag = dict(members=m, perm_auc=j.get("perm_auc"), perm_mean=j.get("perm_mean", 0), importance=imp)
    items = [(lvl, t) for lvl, t in C.warnings_for(lab, spec, last_diag)]
    if not spec["name"]:
        items.insert(0, ("error", "задайте имя спецификации — под ним сохраняются диагностика и пакет"))
    if not last_diag:
        items.append(("info", "диагностики этой спецификации на этом источнике ещё нет — запустите её перед обучением"))
    ok = UC.checklist(items)
    st.markdown("##### 3. Запуск (фон)")
    c = st.columns([2, 2, 2])
    perm = int(c[0].number_input("Перестановок меток", 0, 20, 3, key="clf_np"))
    imp = c[1].checkbox("Важность признаков (долго)", True, key="clf_imp")
    if c[2].button("Сохранить спецификацию и запустить диагностику", disabled=not ok, type="primary"):
        C.save_spec(spec)
        tid = T.start("clf_diagnose", dict(spec=spec["name"], run=run or "", perm=perm, importance=imp))
        st.toast(f"диагностика: {T.status(tid)['state']}", icon="⏳")
    c = st.columns([2, 2, 2])
    bname = c[0].text_input("Имя пакета", f"cycle_{spec['name']}" if spec["name"] else "", key="clf_bundle")
    names = PR.list_profiles()
    attach = c[1].selectbox("Подключить к профилю", ["", *names], key="clf_attach", format_func=lambda x: x or "— не подключать —")
    exists = bool(bname.strip()) and (CYCLE_MODELS / bname.strip()).exists()
    valid_name = bool(bname.strip()) and all(ch.isalnum() or ch in "-_." for ch in bname.strip())
    vals = dict(spec=spec["name"], bundle=bname.strip(), run=run or "", overwrite=exists, attach=attach)
    if exists:
        def _go():
            C.save_spec(spec)
            return f"задача: {T.start('clf_train', vals)}"
        with c[2]:
            UC.confirm("Перезаписать пакет и обучить", "clf_over", f"Пакет models/cycle/{bname.strip()} будет перезаписан: профили, которые на него ссылаются, начнут давать другие оценки.",
                       _go, disabled=not ok or not valid_name)
    elif c[2].button("Сохранить спецификацию и обучить пакет", disabled=not ok or not valid_name):
        C.save_spec(spec)
        T.start("clf_train", vals)
        st.toast("обучение поставлено в очередь задач", icon="⏳")
    if attach:
        st.caption(f"После обучения пакет будет подключён к профилю «{attach}»: его отпечаток изменится — профиль нужно заново оценить («Валидация»).")
    UC.task_panel(("clf_diagnose", "clf_train"), key="clf_tp", limit=3, title="Задачи классификатора")
    if diag_dirs:
        st.markdown("##### Результат последней диагностики")
        pick = st.selectbox("Диагностика", diag_dirs, format_func=lambda p: p.name, key="clf_diag_pick")
        _show_diag(pick)


# ============================================================================================== результаты
def _bundles() -> pd.DataFrame:
    from sd import bundle as B

    used = {}
    for n in PR.list_profiles():
        try:
            cb = PR.load_named(n).opts().get("cycle_bundle")
            if cb:
                used.setdefault(Path(cb).name, []).append(n)
        except Exception:       # noqa: BLE001
            continue
    rows = []
    for d in sorted(CYCLE_MODELS.glob("*/manifest.json")) if CYCLE_MODELS.exists() else []:
        try:
            m = json.loads(d.read_text(encoding="utf-8"))
        except Exception:       # noqa: BLE001
            continue
        vids, src = B.training_videos(d.parent)
        ts = m.get("table_source") or {}
        rows.append(dict(пакет=d.parent.name, создан=m.get("created"), набор=m.get("feature_set") or (m.get("spec") or {}).get("set"), признаков=len(m.get("features", [])),
                         циклов=m.get("n"), затяжек=m.get("n_pos"), видео=len(vids), AUC_oof=(m.get("cv") or {}).get("auc_ensemble"), калибровка=m.get("calibrated"),
                         источник=ts.get("run") or ("последние запуски" if ts else "—"), в_профилях=", ".join(used.get(d.parent.name, [])) or "—"))
    return pd.DataFrame(rows)


def _results() -> None:
    st.markdown("##### Пакеты классификатора")
    df = _bundles()
    if df.empty:
        st.info("Пакетов нет: обучите в разделе «Обучение».")
    else:
        st.dataframe(df.sort_values("создан", ascending=False), hide_index=True, column_config={"AUC_oof": st.column_config.ProgressColumn("AUC oof", min_value=0.5, max_value=1.0, format="%.3f")})
        st.caption("AUC oof — по видео, которых модель не видела при обучении (фолды по видео), но на тех же клипах, на которых потом считается F1: для честного сравнения конфигураций "
                   "используйте вложенную оценку ниже. «видео» — клипы обучения: оценка на них оптимистична (вкладка «Валидация» предупреждает).")
    st.divider()
    st.markdown("##### Честная оценка наборов признаков (вложенный подбор порогов)")
    run = _source_select("oof_run")
    c = st.columns([3, 2, 1, 1])
    sets = c[0].multiselect("Наборы признаков", list(CM.SETS), default=["obj_hold", "obj_hold_zsd"], key="oof_sets")
    kinds = c[1].multiselect("Члены", ["lr", "gb", "nn"], default=["lr", "gb"], key="oof_kinds")
    rep = int(c[2].number_input("Повторов", 1, 10, 3, key="oof_rep"))
    scene = c[3].checkbox("По сценам", key="oof_scene")
    tag = f"_ui_{time.strftime('%m%d_%H%M')}"
    UC.task_button("oof_eval", dict(run=run or "", sets=",".join(sets), kinds=",".join(kinds), repeats=rep, by_scene=scene, tag=tag), "Запустить честную оценку (фон)",
                   key="oof_go", disabled=not sets or not kinds)
    UC.task_panel(("oof_eval",), key="oof_tp", limit=2, title=None)
    files = sorted((OUTPUTS / "analysis").glob("oof_study*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        f = st.selectbox("Результат", files, format_func=lambda p: f"{p.name} · {UC.ago(p.stat().st_mtime)}", key="oof_file")
        st.dataframe(pd.read_csv(f, encoding="utf-8-sig"), hide_index=True)
        st.caption("«F1 вложенный» — пороги цикла и события подбираются без видео, на котором считается F1: это и есть ожидаемое качество; «F1 лучший» — оптимистичен.")


# ============================================================================================== очередь
def _form(t: T.TaskType) -> dict:
    vals: dict = {}
    for f in t.fields:
        key = f"task_{t.id}_{f.key}"
        if f.kind == "bool":
            vals[f.key] = st.checkbox(f.label, bool(f.default), key=key, help=f.help or None)
        elif f.kind == "int":
            vals[f.key] = int(st.number_input(f.label, 0, 100000, int(f.default), key=key, help=f.help or None))
        elif f.kind == "float":
            vals[f.key] = float(st.number_input(f.label, 0.0, 10000.0, float(f.default), key=key, help=f.help or None))
        elif f.kind == "choice":
            vals[f.key] = st.selectbox(f.label, list(f.choices), index=list(f.choices).index(f.default) if f.default in f.choices else 0, key=key, help=f.help or None)
        elif f.kind == "lines":
            vals[f.key] = st.text_area(f.label, str(f.default), key=key, help=f.help or None, height=80)
        else:
            vals[f.key] = st.text_input(f.label, str(f.default), key=key, help=f.help or None)
    return vals


def _queue() -> None:
    UC.task_panel(None, key="q_all", limit=15, title="Все задачи (новые сверху; тяжёлые выполняются по одной, остальные ждут в очереди)")
    with st.expander("Запустить задачу вручную"):
        groups = sorted({t.group for t in T.CATALOG.values()})
        g = st.radio("Раздел", groups, horizontal=True, key="q_group")
        ids = [t.id for t in T.CATALOG.values() if t.group == g]
        tid = st.selectbox("Задача", ids, format_func=lambda i: T.CATALOG[i].title, key="q_pick")
        t = T.CATALOG[tid]
        st.write(t.description)
        if t.needs_net:
            st.warning("Нужна сеть (загрузка весов). Данные площадки при этом никуда не отправляются.")
        vals = _form(t)
        st.code("python -m sd " + " ".join(T.command(tid, vals)), language="bash")
        UC.task_button(tid, vals, "Запустить", key="q_go")


def render() -> None:
    UC.style()
    UC.header("Задачи", "Всё, что улучшает модель и пополняет данные: разметка → признаки → обучение → честная оценка. Тяжёлое — фоновыми задачами.")
    sec = UC.sections(SECTIONS, "tasks_section")
    {"Разметка событий": _labeling, "Метки жестов": _gestures, "Признаки": _features, "Обучение": _training, "Результаты": _results, "Очередь задач": _queue}[sec]()
