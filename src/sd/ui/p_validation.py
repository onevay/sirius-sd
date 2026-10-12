"""Вкладка «Валидация»: проверить выбранную конфигурацию на размеченных папках, получить отчёт (в том числе PDF) и разобрать по кэшу, где модель слаба.

Разделы:
  Запуск            — данные, профиль, классификатор, правки; чек-лист до запуска (устройство, утечка обучения, кэш, повтор прогона); оценка — фоновой задачей
  Результат и PDF   — метрики с интервалом, порог, клипы, ошибки; PDF-отчёт (с парным сравнением с базовым прогоном)
  Анализ кэша       — клип × метрика по всем клипам прогона, слабые места по правилам, видео клипа с циклами, история клипа по всем прогонам
  Сравнение         — два прогона на общих клипах: парный бутстрэп, изменившиеся клипы
  Поток             — имитация реального времени по видео (тревоги, задержка, × реального времени)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from sd import cache_analysis as CA
from sd import catalog as CAT
from sd import experiments as XP
from sd import gt as GT
from sd import library as LIB
from sd import profiles as PR
from sd import runner as RN
from sd import solver as SV
from sd.config import stable_hash
from sd.paths import portable, video_id
from sd.ui import common as UC
from sd.ui import nav, profile_editor, report_view

SECTIONS = ["Запуск", "Результат и PDF", "Анализ кэша", "Сравнение", "Поток"]
POLICIES = {"fixed": "порог из профиля (честно)", "plateau": "центр плато F1 (подбор на этих же клипах)", "best": "вершина кривой (оптимистично)"}


@st.cache_data(show_spinner=False, ttl=30)
def _dirs() -> pd.DataFrame:
    return LIB.discover_dirs()


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flat(v, kk + "."))
        else:
            out[kk] = v
    return out


def diff_sets(base: PR.Profile, prof: PR.Profile) -> list[str]:
    """Правки `ключ=значение`, превращающие base в prof (по действующей конфигурации и опциям) — для команды `sd eval -s …`."""
    a, b = _flat(base.cfg().to_dict()), _flat(prof.cfg().to_dict())
    out = [f"{k}={yaml.safe_dump(v, default_flow_style=True, allow_unicode=True).strip().removesuffix('...').strip()}" for k, v in b.items() if a.get(k) != v]
    oa, ob = base.opts(), prof.opts()
    out += [f"options.{k}={json.dumps(v, ensure_ascii=False)}" for k, v in ob.items() if oa.get(k) != v]
    return out


def _cached_count(prof: PR.Profile, videos: list[Path], max_sec: float) -> tuple[int, int] | None:
    """Сколько клипов возьмётся из кэша оценки (ключ как в runner._key; только для полного клипа)."""
    if max_sec:
        return None
    try:
        fp = prof.fingerprint()
    except Exception:       # noqa: BLE001
        return None
    hit = 0
    for v in videos:
        stt = v.stat()
        key = stable_hash(dict(fp=fp, v=portable(v), size=stt.st_size, mt=stt.st_mtime_ns, w=[0.0, None]), 12)
        hit += int((RN.CACHE_DIR / key / "events.csv").exists())
    return hit, len(videos)


def _launch() -> None:
    gt = GT.load()
    dirs = _dirs()
    if dirs.empty:
        st.info("Папок с видео нет: положите клипы в `data/` (или задайте SD_DATA).")
        return
    st.markdown("##### 1. Данные")
    default = [n for n in dirs.name if Path(n).name in ("курение", "лжекурение")] or dirs.name.tolist()[:1]
    chosen = st.multiselect("Папки с размеченными видео", dirs.name.tolist(), default=st.session_state.get("val_dirs", default), key="val_dirs")
    paths = dirs.set_index("name").loc[chosen, "path"].tolist() if chosen else []
    vids = LIB.videos_in(paths) if paths else []
    ids = {video_id(v): v for v in vids}
    status = GT.clip_status(gt, list(ids)) if ids else pd.DataFrame(columns=["clip_id", "reviewed", "positive"])
    labeled = status[status.reviewed].clip_id.tolist() if len(status) else []
    c = st.columns(5)
    mode = c[0].radio("Эталон", ["events", "clips"], key="val_mode", format_func=lambda m: {"events": "ручная разметка", "clips": "по имени папки"}[m])
    role = c[1].radio("Набор", ["validation", "hidden"], key="val_role", format_func=lambda m: {"validation": "валидация", "hidden": "скрытый (порог заморожен)"}[m])
    policy = c[2].selectbox("Порог", list(POLICIES), format_func=POLICIES.get, key="val_policy", disabled=role == "hidden")
    target = c[3].number_input("Целевая F1", 0.0, 1.0, 0.80, 0.01, key="val_target")
    max_sec = c[4].number_input("Только первые N с клипа (0 — целиком)", 0, 3600, 0, 10, key="val_maxsec")
    only = st.text_input("Точечно: только клипы (подстроки id через запятую; пусто — все)", key="val_clips", placeholder="например курение__3,sm_6")

    st.markdown("##### 2. Конфигурация")
    names = PR.list_profiles()
    if not names:
        st.error("Сохранённых профилей нет: создайте на вкладке «Модели» или установите решатель на вкладке «Выпуск».")
        return
    c = st.columns([2, 2, 3])
    pname = c[0].selectbox("Профиль", names, index=names.index("mvp") if "mvp" in names else 0, key="val_profile")
    base = PR.load_named(pname)
    bundles = [b.id for b in CAT.cycle_bundles()]
    cls = c[1].selectbox("Классификатор цикла", [None, *bundles], key="val_cls", format_func=lambda x: f"как в профиле ({Path(base.opts().get('cycle_bundle') or '—').name})" if x is None else Path(x).name)
    c[2].caption(base.description[:300] or "—")
    sets: list[str] = []
    with st.expander("Параметры и модели (правки не меняют файл профиля — уходят в прогон как `-s ключ=значение`)"):
        try:
            edited = profile_editor.editor(base, key="val_pe", classifier=cls or base.opts().get("cycle_bundle"))
            sets = diff_sets(base, edited)
        except Exception as e:      # noqa: BLE001
            st.error(f"редактор параметров: {e}")
        extra = st.text_area("Дополнительные правки (по одной на строку, `ключ=значение`)", key="val_sets", height=70, placeholder="video.process_fps=10")
        sets += [x.strip() for x in extra.splitlines() if x.strip()]
        if sets:
            st.code("\n".join(sets), language=None)
    prof, perr = None, None
    try:
        prof = PR.with_overrides(RN.with_classifier(base, cls), sets)
    except Exception as e:      # noqa: BLE001
        perr = str(e)

    st.markdown("##### 3. Проверка до запуска")
    items: list[tuple[str, str]] = []
    if not chosen:
        items.append(("error", "не выбраны папки"))
    if perr:
        items.append(("error", f"правки профиля: {perr}"))
    want = [w.strip() for w in only.split(",") if w.strip()]
    scope = [c_ for c_ in (labeled if mode == "events" else list(ids)) if not want or any(w in c_ for w in want)]
    if chosen and mode == "events":
        if not labeled:
            items.append(("error", "в выбранных папках нет размеченных клипов — разметьте («Задачи → Разметка событий») или выберите «по имени папки»"))
        else:
            items.append(("ok", f"размеченных клипов {len(labeled)} из {len(ids)}; в оценку — {len(scope)}; эпизодов курения {int(status[status.reviewed].positive.sum())}"))
            if len(ids) > len(labeled):
                items.append(("info", f"неразмеченные клипы ({len(ids) - len(labeled)}) будут пропущены"))
    if want and not scope:
        items.append(("error", f"ни один клип не подходит под «{only}»"))
    if prof is not None:
        for i in SV.check(prof, devices=True):
            items.append(("error" if i.level == "error" else "warning", i.text))
        lk = RN.training_leak(prof, scope)
        if lk["clips"]:
            lvl = "error" if role == "hidden" else "warning"
            items.append((lvl, f"классификатор обучен на {len(lk['clips'])} из {len(scope)} оцениваемых клипов — F1 будет оптимистичной"
                                + (" (на скрытом наборе так нельзя)" if role == "hidden" else "; честная оценка — «Задачи → Результаты» (вложенная oof)")))
        elif scope:
            items.append(("ok", f"клипы оценки не пересекаются с обучением классификатора (источник: {lk['source']})"))
        cc = _cached_count(prof, [ids[c_] for c_ in scope], max_sec)
        if cc:
            items.append(("info", f"из кэша: {cc[0]} клипов, считать заново: {cc[1] - cc[0]}" + (" — поза и признаки пересчитаются, это долго (≈ длительность видео и больше)" if cc[1] - cc[0] else "")))
        try:
            fp = prof.fingerprint()
            runs = UC.runs_df()
            same = runs[(runs.dirs == ", ".join(Path(p).name for p in paths)) & (runs["mode"] == mode)] if len(runs) else runs
            same = [r for r in same.itertuples() if json.loads((XP.EXP_DIR / r.id / "meta.json").read_text(encoding="utf-8")).get("fingerprint") == fp]
            if same:
                items.append(("info", f"такая конфигурация на этих папках уже оценивалась: {same[0].id} (F1 {same[0].f1:.3f}) — повтор нужен, только если менялся эталон"))
        except Exception:       # noqa: BLE001
            pass
    if role == "hidden":
        items.append(("info", "скрытый набор: порог берётся из профиля, подбор запрещён"))
    if sets:
        items.append(("info", f"правок профиля: {len(sets)} — попадут в отпечаток и журнал, файл профиля не меняется"))
    ok = UC.checklist(items)
    values = dict(dirs="\n".join(str(p) for p in paths), profile=pname, classifier=cls or "", sets="\n".join(sets), mode=mode, role=role,
                  policy="fixed" if role == "hidden" else policy, target_f1=float(target), max_sec=float(max_sec), clips=only.strip(), no_cache=False)
    c = st.columns([2, 2, 4])
    UC.task_button("eval", values, "▶ Запустить оценку", key="val_go", disabled=not ok or not chosen)
    if c[1].checkbox("показать команду", key="val_showcmd"):
        st.code("python -m sd " + " ".join(f'"{a}"' if " " in a else a for a in __import__("sd.tasks", fromlist=["x"]).command("eval", values)), language="bash")
    UC.task_panel(("eval", "variants"), key="val_tp", title="Оценки (фон)")
    st.caption("Готовый прогон появится в разделе «Результат и PDF» и в журнале; закрытие страницы задачу не прерывает.")


def _result() -> None:
    rid = UC.run_select("Прогон", key="val_run", mode=None, prefer=st.session_state.pop("val_run_prefer", None))
    if not rid:
        return
    rep, meta, ev, gt = XP.load_run(rid)
    warn = []
    leak = meta.get("leak")
    if leak is None:
        try:
            p = meta["profile"]
            leak = RN.training_leak(PR.Profile(p.get("name", "x"), "", p.get("base"), p.get("config") or {}, p.get("options") or {}), [c["clip_id"] for c in meta.get("clips", [])])
        except Exception:       # noqa: BLE001
            leak = {}
    if (leak or {}).get("clips"):
        warn.append(("warning", f"классификатор обучен на {len(leak['clips'])} из {len(meta.get('clips', []))} клипов прогона: F1 оптимистична"))
    failed = [c["clip_id"] for c in meta.get("clips", []) if c.get("error")]
    if failed:
        warn.append(("error", f"клипов со сбоем обработки: {len(failed)} ({', '.join(failed[:4])}) — посчитаны как «событий нет»"))
    if warn:
        UC.checklist(warn)
    report_view.show(rep, meta, rid, ev)
    st.divider()
    st.markdown("##### PDF-отчёт")
    c = st.columns([4, 1.4, 1.4], vertical_alignment="bottom")
    with c[0]:
        base = UC.run_select("Сравнить с базовым прогоном (необязательно)", key="val_pdf_base", mode="events", allow_none=True)
    if base == rid:
        st.caption("базовый совпадает с отчётным — сравнение не делается")
        base = None
    if c[1].button("Собрать PDF", type="primary", key="val_pdf"):
        from sd import reports as R

        with st.spinner("собираем PDF (несколько секунд)…"):
            try:
                st.session_state["val_pdf_path"] = str(R.build(rid, baseline=base, author=UC.who()))
            except Exception as e:      # noqa: BLE001
                st.error(f"PDF не собран: {type(e).__name__}: {e}")
    pdfs = sorted((XP.EXP_DIR / rid).glob("report_*.pdf"), key=lambda f: f.stat().st_mtime, reverse=True)
    if pdfs:
        f = pdfs[0]
        c[2].download_button("Скачать PDF", f.read_bytes(), file_name=f"{rid}.pdf", mime="application/pdf", key="val_pdf_dl")
        st.caption(f"{f} · собран {UC.ago(f.stat().st_mtime)}. Содержание: конфигурация и отпечатки, данные, метрики с интервалом, порог, ошибки, клипы, "
                   "производительность, сравнение, слабые места, команда воспроизведения.")


@st.cache_data(show_spinner="собираем сводку по клипам…")
def _table(rid: str, mtime: float, refresh: int) -> pd.DataFrame:
    return CA.run_table(rid, refresh=bool(refresh))


@st.cache_data(show_spinner=False)
def _gestures(mtime: float) -> pd.DataFrame:
    from sd import gestures as G

    return G.load()


def _heatmap(tab: pd.DataFrame) -> go.Figure:
    vals, good = CA.matrix(tab.sort_values("f1", na_position="last"))
    text = vals.map(lambda v: "—" if pd.isna(v) else (f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"))
    fig = go.Figure(go.Heatmap(z=good.to_numpy(), x=list(vals.columns), y=list(vals.index), text=text.to_numpy(), texttemplate="%{text}", zmin=-1, zmax=1,
                               colorscale=[[0, "#dc2626"], [0.35, "#fca5a5"], [0.5, "#f8fafc"], [0.65, "#bbf7d0"], [1, "#16a34a"]], showscale=False,
                               hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    fig.update_layout(height=max(260, 24 * len(vals) + 60), margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(side="top"), yaxis=dict(autorange="reversed"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _cache() -> None:
    c = st.columns([5, 1])
    rid = UC.run_select("Прогон", key="ca_run", mode="events", prefer=st.session_state.pop("ca_run_prefer", None))
    if not rid:
        return
    if c[1].button("Пересчитать сводку", help="перечитать кэш поз и запусков (после удаления/пересчёта)"):
        st.session_state["ca_refresh"] = st.session_state.get("ca_refresh", 0) + 1
    rep, meta, ev, gt = XP.load_run(rid)
    tab = _table(rid, (XP.EXP_DIR / rid / "meta.json").stat().st_mtime, st.session_state.get("ca_refresh", 0))
    from sd.start_eval import GT_CSV

    gest = _gestures(UC.file_mtime(GT_CSV))
    spots = CA.weak_spots(tab, rep.errors, gest)
    k = st.columns(5)
    k[0].metric("Клипов", len(tab))
    k[1].metric("С ошибками", int((tab.status.fillna("ok") != "ok").sum()) if "status" in tab else 0)
    k[2].metric("Слабых мест", int((spots.level == "error").sum()), help="уровня «ошибка»")
    k[3].metric("Предупреждений", int((spots.level == "warning").sum()))
    k[4].metric("× реального времени (медиана)", "—" if tab.rt_factor.isna().all() else f"{tab.rt_factor.median():.2f}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Клип × метрика** (зелёный — лучше медианы, красный — хуже; F1 и ошибки — по абсолютной шкале)")
        st.plotly_chart(_heatmap(tab), width="stretch", key="ca_heat")
    with right:
        st.markdown("**Что чинить в первую очередь**")
        st.dataframe(CA.summary(spots), hide_index=True, width="stretch")
        lv = st.multiselect("Уровни", ["error", "warning", "info"], default=["error", "warning"], key="ca_lv")
        view = spots[spots.level.isin(lv)]
        sel = st.dataframe(view[["level", "clip_id", "rule", "detail"]], hide_index=True, on_select="rerun", selection_mode="single-row", key="ca_spots", height=280)
        rows = sel["selection"]["rows"] if sel and sel.get("selection") else []
        if rows:
            st.session_state["ca_clip"] = view.iloc[rows[0]].clip_id
    st.divider()
    clips = tab.sort_values(["f1", "fn"], ascending=[True, False], na_position="last").clip_id.tolist()
    want = st.session_state.get("ca_clip")
    if want not in clips:
        want = clips[0]
        st.session_state["ca_clip"] = want
    cid = st.selectbox("Клип для разбора", clips, key="ca_clip", format_func=lambda c_: f"{c_} · " + "; ".join(spots[spots.clip_id == c_].rule.unique()[:4]) if c_ in set(spots.clip_id) else c_)
    row = tab[tab.clip_id == cid].iloc[0]
    st.caption(" · ".join(f"{k_}: {v}" for k_, v in [("F1", "—" if pd.isna(row.get("f1")) else f"{row.get('f1'):.2f}"), ("TP/FP/FN", f"{row.get('tp', 0):.0f}/{row.get('fp', 0):.0f}/{row.get('fn', 0):.0f}"),
                                                    ("циклов", row.get("cycles")), ("рост людей", "—" if pd.isna(row.get("h_med")) else f"{row.get('h_med'):.0f} px"),
                                                    ("× р.в.", row.get("rt_factor")), ("в обучении", "да" if row.get("leak") else "нет")]))
    from sd.ui import clip_view

    clip_view.show(rep, meta, ev, gt, cid)
    for s_ in spots[spots.clip_id == cid].itertuples():
        st.markdown(f"{UC.LEVEL_ICON[s_.level]} **{s_.rule}** — {s_.text}" + (f" ({s_.detail})" if s_.detail else ""))
    errs = rep.errors[rep.errors.clip_id == cid] if len(rep.errors) else rep.errors
    if len(errs):
        from sd import gestures as G

        e2 = errs[["kind", "start", "end", "confidence", "peak", "cause_ru"]].copy()
        e2["жест рядом"] = [", ".join(G.LABEL_RU.get(x, x) for x in G.near(gest, cid, float(p)).label) or "—" for p in e2.peak]
        st.dataframe(e2.round(2), hide_index=True)
    c = st.columns(3)
    if c[0].button("Разметить события этого клипа", key="ca_to_label"):
        folder = Path(meta["clips"][[x["clip_id"] for x in meta["clips"]].index(cid)]["path"]).parent.name
        nav.go("tasks", tasks_section="Разметка событий", lab_clip=cid, lab_dirs=[folder])
    if c[1].button("Метки жестов клипа", key="ca_to_gest"):
        nav.go("tasks", tasks_section="Метки жестов", an_g_v=[cid])
    with st.expander("История клипа по всем прогонам журнала"):
        st.dataframe(CA.clip_history(cid), hide_index=True)


def _compare() -> None:
    from sd import sweep as SW

    c = st.columns(2)
    with c[0]:
        a = UC.run_select("Первый (базовый)", key="cmp_a", mode="events")
    with c[1]:
        b = UC.run_select("Второй (новый)", key="cmp_b", mode="events")
    if not a or not b:
        return
    if a == b:
        st.info("Выберите два разных прогона.")
        return
    try:
        r = SW.compare_saved(a, b)
    except Exception as e:      # noqa: BLE001
        st.warning(str(e))
        return
    if r["diff"] is None:
        st.warning("Мало общих клипов для сравнения (нужно ≥ 3).")
        return
    k = st.columns(3)
    k[0].metric(f"F1: второй − первый ({r['clips']} общих клипов)", f"{r['diff'] * 100:+.1f} п.п.", f"95 %: {r['lo'] * 100:+.1f} … {r['hi'] * 100:+.1f}")
    k[1].metric("Вероятность «второй лучше»", f"{r['p_better']:.0%}")
    verdict = "второй лучше" if r["lo"] > 0 else ("первый лучше" if r["hi"] < 0 else "различие в пределах шума")
    k[2].metric("Вывод", verdict)
    if not r["same_gt"]:
        st.error("Эталон в прогонах различается — сравнение некорректно: пересчитайте оба на текущей разметке.")
    ra, ma, _, _ = XP.load_run(a)
    rb, mb, _, _ = XP.load_run(b)
    j = ra.per_clip[["clip_id", "f1", "status"]].merge(rb.per_clip[["clip_id", "f1", "status"]], on="clip_id", suffixes=("_1", "_2"))
    j = j[(j.f1_1.fillna(-1) != j.f1_2.fillna(-1)) | (j.status_1 != j.status_2)]
    st.markdown("**Клипы, где результат изменился**")
    st.dataframe(j, hide_index=True)
    da, db = _flat(ma.get("profile", {}).get("config") or {}), _flat(mb.get("profile", {}).get("config") or {})
    keys = sorted(set(da) | set(db))
    diff = [dict(параметр=k_, первый=da.get(k_), второй=db.get(k_)) for k_ in keys if da.get(k_) != db.get(k_)]
    oa, ob = ma.get("profile", {}).get("options") or {}, mb.get("profile", {}).get("options") or {}
    diff += [dict(параметр=f"options.{k_}", первый=str(oa.get(k_)), второй=str(ob.get(k_))) for k_ in sorted(set(oa) | set(ob)) if oa.get(k_) != ob.get(k_)]
    st.markdown("**Чем отличаются конфигурации**")
    st.dataframe(pd.DataFrame(diff), hide_index=True) if diff else st.caption("параметры профилей совпадают")


def render() -> None:
    st.header("Валидация")
    if "val_section" not in st.session_state:
        st.session_state["val_section"] = SECTIONS[0]
    sec = st.radio("Раздел", SECTIONS, horizontal=True, label_visibility="collapsed", key="val_section")
    if sec == "Запуск":
        _launch()
    elif sec == "Результат и PDF":
        _result()
    elif sec == "Анализ кэша":
        _cache()
    elif sec == "Сравнение":
        _compare()
    else:
        from sd.ui import page_stream

        page_stream.render()
