"""Страница «Классификатор»: полное управление оценкой цикла — признаки, члены ансамбля, регуляризация, схема кросс-валидации, диагностика переобучения, обучение пакета и подключение его к профилю.

Порядок работы:
  1. Данные       — таблица циклов с метками жестов (`sd enrich` добавляет предмет/фото/VLM); сколько затяжек и видео, какие признаки заполнены.
  2. Спецификация — набор признаков, члены ансамбля (тип, признаки, гиперпараметры), число фолдов, повторы, фолды по сценам, калибровка; сохраняется в configs/classifiers/.
  3. Диагностика  — oof против train, контроль перестановкой, кривая обучения, важность признаков, словесный вывод о переобучении.
  4. Обучение     — пакет models/cycle/<имя> (без pickle) и запись его в выбранный профиль.
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sd import classifier as C
from sd import cycle_models as CM
from sd import profiles as PR
from sd.paths import MODELS


@st.cache_data(show_spinner="собираем таблицу циклов…")
def _table(enriched: bool, stamp: int) -> pd.DataFrame:
    return C.load_table(enriched)


def _data_block() -> pd.DataFrame | None:
    c1, c2, c3 = st.columns([2, 2, 2])
    enriched = c1.checkbox("с признаками `enrich` (предмет, фото, VLM)", value=True, key="clf_enr")
    if c2.button("Пересобрать таблицу", help="после новых прогонов, меток жестов или `enrich`"):
        st.session_state["clf_stamp"] = st.session_state.get("clf_stamp", 0) + 1
    try:
        tab = _table(enriched, st.session_state.get("clf_stamp", 0))
    except Exception as e:                       # noqa: BLE001 — нет запусков/меток: подсказываем, что сделать
        st.error(f"таблица циклов не собрана: {type(e).__name__}: {e}")
        st.info("Нужны: полные запуски распознавания (`outputs/recognize`), метки жестов (страница «Анализ») и, для признаков предмета/фото/VLM, задача «Признаки циклов» на странице «Задачи».")
        return None
    lab = tab[tab.y.notna()]
    c3.metric("циклов с меткой", len(lab), help=f"всего циклов {len(tab)}")
    m1, m2, m3 = st.columns(3)
    m1.metric("затяжек", int((lab.y == 1).sum()))
    m2.metric("не затяжек", int((lab.y == 0).sum()))
    m3.metric("видео с метками", lab.video.nunique())
    if len(lab) < 60 or lab.y.nunique() < 2:
        st.warning("мало размеченных циклов: оценка будет шумной. Разметьте больше жестов на странице «Анализ».")
    return lab.reset_index(drop=True)


def _members_editor(spec: dict, tab: pd.DataFrame) -> dict:
    cols = [c for c in tab.columns if c not in CM.CONFOUNDS and pd.api.types.is_numeric_dtype(tab[c]) and c not in ("y", "peak_t", "tid", "start", "end")]
    filled = {c: float(tab[c].notna().mean()) for c in cols}
    members = {}
    st.caption("Каждый член — отдельная модель; оценка цикла — среднее по членам. Меньше признаков и сильнее регуляризация — меньше риск переобучения на сотне циклов.")
    for n, m in list(spec["members"].items()):
        with st.expander(f"{n} — {C.KINDS.get(m['kind'], m['kind'])}", expanded=False):
            on = st.checkbox("включён", True, key=f"clf_on_{n}")
            kind = st.selectbox("тип", list(C.KINDS), index=list(C.KINDS).index(m["kind"]) if m["kind"] in C.KINDS else 0, key=f"clf_kind_{n}", format_func=lambda k: C.KINDS[k])
            feats = st.multiselect("признаки", sorted(set(cols) | set(m["features"])), default=[f for f in m["features"] if f in cols or f in m["features"]], key=f"clf_f_{n}",
                                   format_func=lambda c: f"{c} ({filled.get(c, 0):.0%} заполнено)")
            params = dict(C.PARAMS[kind])
            params.update({k: v for k, v in (m.get("params") or {}).items() if k in params} if kind == m["kind"] else {})
            new = {}
            pc = st.columns(max(len(params), 1))
            for i, (k, v) in enumerate(params.items()):
                key = f"clf_p_{n}_{k}"
                if isinstance(C.PARAMS[kind][k], int) and not isinstance(C.PARAMS[kind][k], bool):
                    new[k] = int(pc[i].number_input(k, 1, 1000, int(v), key=key))
                else:
                    new[k] = float(pc[i].number_input(k, 0.0, 1000.0, float(v), format="%.4g", key=key))
            if on:
                members[n] = dict(kind=kind, features=feats, params=new)
    return members


def _spec_block(tab: pd.DataFrame) -> dict:
    saved = C.list_specs()
    c1, c2 = st.columns([2, 2])
    src = c1.selectbox("Откуда взять спецификацию", ["новая из набора признаков", *saved], key="clf_src")
    if src == "новая из набора признаков":
        sname = c2.selectbox("Набор признаков", list(CM.SETS), index=list(CM.SETS).index("obj_hold_zsd"), key="clf_set")
        kinds = st.multiselect("Члены ансамбля", ["lr", "gb", "nn"], default=["lr", "gb", "nn"], key="clf_kinds", format_func=lambda k: {"lr": "логрегрессия", "gb": "LightGBM", "nn": "MLP"}[k])
        base = C.default_spec(sname, kinds, tab)
        miss = [c for c in CM.SETS[sname] if c not in tab.columns]
        if miss:
            st.warning(f"в таблице нет колонок набора: {', '.join(dict.fromkeys(miss))} — выполните «Признаки циклов» на странице «Задачи».")
        ident = f"new_{sname}_{'-'.join(kinds)}"
    else:
        base = C.load_spec(src)
        ident = f"saved_{src}"
    # ключи виджетов зависят от источника: смена набора не тащит старые значения
    st.session_state.setdefault("clf_ident", ident)
    if st.session_state["clf_ident"] != ident:
        for k in [k for k in st.session_state if str(k).startswith("clf_") and k not in ("clf_src", "clf_set", "clf_kinds", "clf_enr", "clf_stamp", "clf_ident")]:
            del st.session_state[k]
        st.session_state["clf_ident"] = ident
        st.rerun()
    members = _members_editor(base, tab)
    st.markdown("**Кросс-валидация и калибровка**")
    cv = {**C.DEFAULT_CV, **(base.get("cv") or {})}
    a, b, c, d = st.columns(4)
    n_splits = int(a.number_input("Фолдов", 2, 10, int(cv["n_splits"]), key="clf_ns", help="меньше фолдов — быстрее, но модель обучается на меньшем числе видео"))
    repeats = int(b.number_input("Повторов", 1, 10, int(cv["repeats"]), key="clf_rep", help="разные случайные разбиения видео; оценка усредняется"))
    by_scene = c.checkbox("Фолды по сценам", bool(cv["by_scene"]), key="clf_scene", help="клипы одной камеры/человека не попадают одновременно в обучение и проверку — строже, ближе к скрытому набору")
    seed = int(d.number_input("Seed", 0, 10_000, int(cv["seed"]), key="clf_seed"))
    calibrate = st.checkbox("Изотоническая калибровка оценки ансамбля", bool(base.get("calibrate", False)), key="clf_cal",
                            help="на малой выборке калибровка сама переобучается; MVP обучен без неё — порог цикла подбирается по среднему членов")
    spec = dict(name=base.get("name") if src != "новая из набора признаков" else "", description=base.get("description", ""), set=base.get("set"), features=sorted({f for m in members.values() for f in m["features"]}),
                members=members, cv=dict(n_splits=n_splits, repeats=repeats, by_scene=by_scene, seed=seed), calibrate=calibrate)
    nm = st.text_input("Имя спецификации (для сохранения)", spec["name"] or "", key="clf_name", placeholder="например obj_hold_strict")
    spec["name"] = nm.strip()
    bad = C.validate(spec, tab)
    for p in bad:
        st.error(p)
    if st.button("Сохранить спецификацию", disabled=not spec["name"] or bool(bad)):
        f = C.save_spec(spec)
        st.success(f"сохранено: {f.relative_to(f.parents[2])}")
    st.session_state["clf_bad"] = bad
    return spec


def _diag_block(tab: pd.DataFrame, spec: dict) -> None:
    c1, c2, c3 = st.columns(3)
    n_perm = int(c1.number_input("Перестановок меток", 0, 20, 3, key="clf_np"))
    imp = c2.checkbox("Важность признаков", True, key="clf_imp", help="долго: переобучение для каждого признака")
    if c3.button("Запустить диагностику", type="primary", disabled=bool(st.session_state.get("clf_bad"))):
        bar = st.progress(0.0, "старт")
        res = C.diagnose(tab, spec, n_perm=n_perm, importance=imp, progress=lambda i, n, m: bar.progress(min(i / n, 1.0), m))
        bar.empty()
        st.session_state["clf_diag"] = res
    res = st.session_state.get("clf_diag")
    if not res:
        st.info("Диагностика отвечает на вопрос «не запоминает ли модель текущий датасет». Цикл — минуты на этом ноутбуке.")
        return
    for v in res["verdict"]:
        (st.warning if ("велик" in v or "утечка" in v) else st.info)(v)
    st.dataframe(res["members"], hide_index=True)
    st.caption(f"циклов {res['n']}, затяжек {res['n_pos']}, групп (видео/сцен) {res['n_groups']}. «разрыв» = AUC на обучающих данных минус AUC на невиденных видео: чем меньше, тем надёжнее.")
    if res["perm_auc"]:
        st.metric("AUC при перемешанных метках (должен быть ≈ 0.5 или ниже)", f"{res['perm_mean']:.2f}", help=f"значения: {[round(x, 2) for x in res['perm_auc']]}")
    c1, c2 = st.columns(2)
    if len(res["curve"]):
        f = go.Figure(go.Scatter(x=res["curve"]["доля_видео"], y=res["curve"]["AUC_oof"], mode="lines+markers"))
        f.update_layout(title="Кривая обучения", xaxis_title="доля обучающих видео", yaxis_title="AUC (oof)", height=300, margin=dict(l=10, r=10, t=40, b=10))
        c1.plotly_chart(f, use_container_width=True)
    if len(res["importance"]):
        f = go.Figure(go.Bar(x=res["importance"]["падение_AUC"], y=res["importance"]["признак"], orientation="h"))
        f.update_layout(title="Падение AUC при перемешивании признака", height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
        c2.plotly_chart(f, use_container_width=True)


def _train_block(tab: pd.DataFrame, spec: dict) -> None:
    st.caption("Пакет не содержит pickle: числа и текстовые деревья; переносится копированием папки. Порог цикла и события после смены модели подбирайте заново (страница «Анализ» → «Оценка циклов»).")
    c1, c2 = st.columns([2, 2])
    name = c1.text_input("Имя пакета", spec["name"] and f"cycle_{spec['name']}" or "", key="clf_bundle", placeholder="cycle_new")
    names = PR.list_profiles()
    prof = c2.selectbox("Подключить к профилю (необязательно)", ["— не подключать —", *names], key="clf_prof")
    ok = bool(name.strip()) and not st.session_state.get("clf_bad") and all(ch.isalnum() or ch in "-_." for ch in name.strip())
    exists = (MODELS / "cycle" / name.strip()).exists()
    if exists:
        st.warning("пакет с таким именем есть: будет перезаписан (профили, ссылающиеся на него, получат другие оценки)")
    if st.button("Обучить и сохранить пакет", type="primary", disabled=not ok):
        with st.spinner("обучаем…"):
            man = C.train(tab, spec, MODELS / "cycle" / name.strip(), meta=dict(source="ui"))
        cv = man["cv"]
        st.success(f"models/cycle/{name.strip()}: AUC oof ансамбля {cv['auc_ensemble']}, циклов {man['n']}, затяжек {man['n_pos']}")
        if prof != "— не подключать —":
            p = PR.load(prof)
            opts = {**p.options, "cycle_bundle": f"models/cycle/{name.strip()}"}
            PR.save(PR.Profile(name=p.name, description=p.description, base=p.base, config=p.config, options=opts))
            st.success(f"профиль «{prof}» теперь использует этот пакет (изменится его отпечаток — кэш оценок пересчитается)")


def render() -> None:
    st.title("Классификатор цикла")
    st.caption("Решает, является ли подъём руки ко рту затяжкой. Обучен на сотне циклов, поэтому главный риск — переобучение: ниже можно менять каждую часть и сразу проверять.")
    tab = _data_block()
    if tab is None or tab.empty:
        return
    t1, t2, t3 = st.tabs(["Спецификация", "Диагностика переобучения", "Обучение и подключение"])
    with t1:
        spec = _spec_block(tab)
    with t2:
        _diag_block(tab, spec)
    with t3:
        _train_block(tab, spec)
