"""Вкладка «Аналитика»: сравнение подходов по всему журналу прогонов.

Разделы:
  Журнал               — все прогоны; тег «подхода», заметка, закрепление (пишутся рядом с прогоном, на метрики не влияют); открыть, разобрать, удалить
  Сравнение подходов   — выбранные прогоны: метрики с интервалом, точность против скорости, чем различаются конфигурации
  По клипам            — клип × подход: F1 или исход TP/FP/FN; клипы, где подходы расходятся
  Ошибки и порог       — причины ошибок по подходам, кривые F1 от порога
  Влияние параметров   — значение параметра → F1 по журналу; честные пары «отличаются только этим»; попарные сравнения с бутстрэпом
  Классификаторы       — диагностики спецификаций и честная (вложенная) оценка наборов признаков

Сравнивать можно только прогоны с одним эталоном: фильтр «эталон» сверху по умолчанию оставляет прогоны с отпечатком эталона последнего прогона.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sd import cache_analysis as CA
from sd import compare as CP
from sd import experiments as XP
from sd.ui import common as UC
from sd.ui import nav

SECTIONS = ["Журнал", "Сравнение подходов", "По клипам", "Ошибки и порог", "Влияние параметров", "Классификаторы"]
PALETTE = px.colors.qualitative.Safe
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10))


@st.cache_data(ttl=20, show_spinner=False)
def _catalog(stamp: float) -> pd.DataFrame:
    return CP.catalog()


def catalog() -> pd.DataFrame:
    return _catalog(UC.file_mtime(XP.EXP_DIR) + sum(UC.file_mtime(f) for f in XP.EXP_DIR.glob(f"*/{XP.NOTES}")) if XP.EXP_DIR.exists() else 0.0)


@st.cache_data(show_spinner=False)
def _speed(rid: str, mtime: float) -> float:
    try:
        t = CA.run_table(rid)
        return float(t.rt_factor.median()) if "rt_factor" in t and t.rt_factor.notna().any() else np.nan
    except Exception:       # noqa: BLE001
        return np.nan


def _filter_bar(cat: pd.DataFrame) -> pd.DataFrame:
    """Фильтр «эталон» и выбор прогонов для сравнения (общий для разделов)."""
    ev = cat[cat["mode"] == "events"].copy()
    if ev.empty:
        st.info("Прогонов по ручному эталону нет — запустите оценку («Валидация → Запуск»).")
        return ev
    ev["подпись"] = CP.label(ev)
    gts = ev.gt_fingerprint.fillna("—").value_counts()
    c = st.columns([2, 5])
    gt = c[0].selectbox("Эталон (отпечаток)", list(gts.index), key="an_gt", format_func=lambda g: f"{g} · {gts[g]} прогонов",
                        help="сравнение корректно только на одной версии разметки; версии — «Данные → Версии разметки»")
    pool = ev[ev.gt_fingerprint.fillna("—") == gt]
    default = pool[pool.pinned].id.tolist() or pool.id.tolist()[:4]
    lab = dict(zip(pool.id, pool["подпись"] + " · F1 " + pool.f1.round(3).astype(str)))
    if "an_runs" in st.session_state:
        st.session_state["an_runs"] = [r for r in st.session_state["an_runs"] if r in lab]
    else:
        st.session_state["an_runs"] = default
    tags = sorted(t for t in pool.tag.unique() if t)
    with c[1]:
        sel = st.multiselect("Подходы для сравнения", list(lab), key="an_runs", format_func=lab.get,
                             help="по умолчанию — закреплённые в журнале прогоны, иначе 4 последних; тег прогона — в «Журнале»")
        if tags:
            def _add_tag() -> None:           # колбэк: менять состояние виджетов можно только до их отрисовки
                t = st.session_state.get("an_tagpick")
                if t:
                    st.session_state["an_runs"] = list(dict.fromkeys([*st.session_state.get("an_runs", []), *pool[pool.tag == t].id]))
                st.session_state["an_tagpick"] = None

            st.pills("Добавить все прогоны с тегом", tags, key="an_tagpick", selection_mode="single", on_change=_add_tag)
    return pool[pool.id.isin(sel)].set_index("id").loc[sel].reset_index() if sel else pool.iloc[0:0]


# ============================================================================================== журнал
def _journal(cat: pd.DataFrame) -> None:
    if cat.empty:
        st.info("Журнал пуст: запустите оценку («Валидация → Запуск»).")
        return
    view = cat[["pinned", "tag", "note", "created", "name", "mode", "f1", "ci_lo", "ci_hi", "precision", "recall", "fp_per_hour", "fps", "imgsz", "cycle_model", "clips", "failed",
                "leak_clips", "id"]].copy()
    view["leak_clips"] = view.leak_clips.astype("Float64")
    ed = st.data_editor(view, hide_index=True, key="an_journal", height=420, width="stretch",
                        disabled=[c for c in view.columns if c not in ("pinned", "tag", "note")],
                        column_config={"pinned": st.column_config.CheckboxColumn("📌", help="закреплённые выбираются для сравнения по умолчанию", width="small"),
                                       "tag": st.column_config.TextColumn("подход (тег)", help="например «5 к/с», «детектор B», «новые метки»", width="medium"),
                                       "note": st.column_config.TextColumn("заметка", width="large"),
                                       "created": st.column_config.TextColumn("создан", width="medium"), "name": "профиль", "mode": "эталон",
                                       "f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"),
                                       "ci_lo": st.column_config.NumberColumn("от", format="%.2f"), "ci_hi": st.column_config.NumberColumn("до", format="%.2f"),
                                       "precision": st.column_config.NumberColumn("P", format="%.2f"), "recall": st.column_config.NumberColumn("R", format="%.2f"),
                                       "fp_per_hour": st.column_config.NumberColumn("FP/ч", format="%.2f"), "fps": "к/с", "imgsz": "вход", "cycle_model": "классификатор",
                                       "clips": "клипов", "failed": "сбоев", "leak_clips": st.column_config.NumberColumn("в обучении", help="клипов прогона, на которых обучен классификатор"),
                                       "id": "прогон"})
    changed = ed[(ed.pinned != view.pinned) | (ed.tag.fillna("") != view.tag.fillna("")) | (ed.note.fillna("") != view.note.fillna(""))]
    c = st.columns([2, 3, 3])
    if c[0].button(f"Сохранить пометки ({len(changed)})", type="primary", disabled=changed.empty):
        for r in changed.itertuples():
            XP.set_notes(r.id, tag=str(r.tag or "").strip(), note=str(r.note or "").strip(), pinned=bool(r.pinned))
        st.toast(f"сохранено пометок: {len(changed)}", icon="✅")
        st.rerun()
    rid = c[1].selectbox("Прогон", cat.id.tolist(), key="an_j_pick", format_func=lambda i: f"{i} · F1 {cat.set_index('id').loc[i, 'f1']:.3f}")
    with c[2]:
        b = st.columns(3)
        if b[0].button("Результат", help="метрики, порог, PDF"):
            nav.go("validation", val_section="Результат и PDF", val_run_prefer=rid)
        if b[1].button("Разбор", help="анализ кэша по клипам"):
            nav.go("validation", val_section="Анализ кэша", ca_run_prefer=rid)
        with b[2]:
            UC.confirm("Удалить", "an_del", f"Прогон {rid} будет удалён из журнала вместе с отчётами в его папке. Кэш поз и событий останется.",
                       lambda: XP.delete_run(rid) and f"удалён {rid}")
    st.caption("Тег объединяет прогоны в «подход» для аналитики; заметки и закрепление хранятся в notes.json рядом с прогоном и на метрики не влияют.")


# ============================================================================================== сравнение подходов
def _approaches(sel: pd.DataFrame, cat: pd.DataFrame) -> None:
    if sel.empty:
        st.info("Выберите подходы для сравнения выше.")
        return
    sel = sel.copy()
    sel["скорость"] = [_speed(r, UC.file_mtime(XP.EXP_DIR / r / "meta.json")) for r in sel.id]
    best = sel.f1.max()
    k = st.columns(4)
    k[0].metric("Подходов", len(sel))
    k[1].metric("Лучший F1", f"{best:.3f}", help=sel.loc[sel.f1.idxmax(), "подпись"])
    k[2].metric("Разброс F1", f"{(sel.f1.max() - sel.f1.min()) * 100:.1f} п.п.", help="меньше ширины интервалов — различие, скорее всего, шум; проверьте попарно")
    k[3].metric("Самый быстрый", "—" if sel["скорость"].isna().all() else f"{sel['скорость'].max():.2f}×", help="медиана × реального времени по замерам кэша")
    c1, c2 = st.columns([3, 2])
    order = sel.sort_values("f1")
    fig = go.Figure(go.Bar(x=order.f1, y=order["подпись"], orientation="h", marker_color=PALETTE[0], text=order.f1.round(3), textposition="outside",
                           error_x=dict(type="data", symmetric=False, array=(order.ci_hi - order.f1).fillna(0), arrayminus=(order.f1 - order.ci_lo).fillna(0), color="#6b7280")))
    fig.update_layout(title="Event F1 с 95 % интервалом", height=90 + 36 * len(order), xaxis=dict(range=[0, 1.08]), yaxis_title=None, **LAYOUT)
    c1.plotly_chart(fig, width="stretch", key="an_f1bar")
    sc = sel.dropna(subset=["скорость"])
    if len(sc):
        fig = px.scatter(sc, x="скорость", y="f1", color=sc.cycle_model.astype(str), symbol=sc.fps.astype(str), hover_name="подпись", color_discrete_sequence=PALETTE,
                         labels={"скорость": "× реального времени (медиана)", "f1": "Event F1", "color": "классификатор", "symbol": "к/с"})
        fig.update_traces(marker_size=13)
        fig.add_vline(x=1.0, line_dash="dash", line_color="#9ca3af")
        fig.update_layout(title="Точность против скорости", height=360, **LAYOUT)
        c2.plotly_chart(fig, width="stretch", key="an_scatter")
    else:
        c2.info("Скорость появится после анализа кэша прогонов (считается автоматически при первом открытии).")
    tbl = sel[["подпись", "f1", "ci_lo", "ci_hi", "precision", "recall", "tp", "fp", "fn", "fp_per_hour", "скорость", "leak_clips", "failed", "id"]]
    st.dataframe(tbl, hide_index=True, width="stretch",
                 column_config={"подпись": "подход", "f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"),
                                "ci_lo": st.column_config.NumberColumn("от", format="%.2f"), "ci_hi": st.column_config.NumberColumn("до", format="%.2f"),
                                "precision": st.column_config.NumberColumn("P", format="%.2f"), "recall": st.column_config.NumberColumn("R", format="%.2f"),
                                "fp_per_hour": st.column_config.NumberColumn("FP/ч", format="%.2f"), "скорость": st.column_config.NumberColumn("× р.в.", format="%.2f"),
                                "leak_clips": st.column_config.NumberColumn("в обучении", help="клипов, на которых обучен классификатор: F1 оптимистична"), "failed": "сбоев",
                                "id": "прогон"})
    st.markdown("**Чем различаются конфигурации**")
    var = CP.varying(sel)
    if not var:
        st.caption("Параметры выбранных прогонов совпадают — различие только в данных, эталоне или версии кода.")
    else:
        d = sel.set_index("подпись")[var].T
        d.index = [i.split(".", 1)[1] for i in d.index]
        st.dataframe(d.astype(str), width="stretch")


# ============================================================================================== по клипам
def _by_clip(sel: pd.DataFrame) -> None:
    if len(sel) < 2:
        st.info("Выберите хотя бы два подхода.")
        return
    ids = sel.id.tolist()
    lab = dict(zip(sel.id, sel["подпись"]))
    f1 = CP.per_clip(ids).rename(columns=lab)
    out = CP.outcome_matrix(ids).rename(columns=lab)
    dis = CP.disagreement(f1)
    diff_outcome = out.nunique(axis=1) > 1
    c = st.columns([2, 2, 3])
    only = c[0].toggle("Только клипы, где исходы различаются", value=True, key="an_bc_only")
    show = c[1].radio("Показывать", ["F1", "TP/FP/FN"], horizontal=True, key="an_bc_show")
    clips = out.index[diff_outcome] if only else out.index
    c[2].caption(f"Исходы различаются на {int(diff_outcome.sum())} из {len(out)} клипов. Цвет — F1 клипа (клипы без курения: зелёный = без ложных).")
    if not len(clips):
        st.success("На всех клипах исходы совпадают.")
        return
    order = dis.reindex(clips).sort_values(ascending=False).index
    z = f1.reindex(order)
    no_pos = z.isna()
    fp = CP.per_clip(ids, value="fp").rename(columns=lab).reindex(order)
    z = z.where(~no_pos, np.where(fp.fillna(0) > 0, 0.0, 1.0))
    text = out.reindex(order) if show == "TP/FP/FN" else f1.reindex(order).map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    fig = go.Figure(go.Heatmap(z=z.to_numpy(float), x=list(z.columns), y=list(z.index), text=text.to_numpy(), texttemplate="%{text}", zmin=0, zmax=1, showscale=False,
                               colorscale=[[0, "#dc2626"], [0.5, "#fde68a"], [1, "#16a34a"]], hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    fig.update_layout(height=max(260, 26 * len(z) + 80), xaxis=dict(side="top"), yaxis=dict(autorange="reversed"), **LAYOUT)
    st.plotly_chart(fig, width="stretch", key="an_clipheat")
    c = st.columns([3, 2, 2])
    clip = c[0].selectbox("Клип", list(order), key="an_bc_clip")
    rid = c[1].selectbox("Подход", ids, key="an_bc_run", format_func=lab.get)
    if c[2].button("Разобрать этот клип в этом прогоне"):
        nav.go("validation", val_section="Анализ кэша", ca_run_prefer=rid, ca_clip=clip)
    with st.expander("Этот клип во всех прогонах журнала"):
        st.dataframe(CA.clip_history(clip), hide_index=True, width="stretch")


# ============================================================================================== ошибки и порог
def _errors_threshold(sel: pd.DataFrame) -> None:
    if sel.empty:
        st.info("Выберите подходы выше.")
        return
    ids = sel.id.tolist()
    lab = dict(zip(sel.id, sel["подпись"]))
    er = CP.error_causes(ids)
    c1, c2 = st.columns(2)
    if len(er):
        er["подход"] = er.id.map(lab)
        er["причина"] = er["тип"] + ": " + er["причина"].str.slice(0, 48)
        fig = px.bar(er, x="n", y="подход", color="причина", orientation="h", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title="Ошибки по причинам", height=140 + 34 * len(ids), legend=dict(orientation="h", y=-0.25, font=dict(size=10)), xaxis_title="число", yaxis_title=None,
                          **LAYOUT)
        c1.plotly_chart(fig, width="stretch", key="an_err")
    else:
        c1.success("Ошибок нет ни у одного подхода.")
    cv = CP.curves(ids)
    if len(cv):
        fig = go.Figure()
        for i, rid in enumerate(ids):
            d = cv[cv.id == rid]
            col = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(x=d.threshold, y=d.f1, name=lab[rid], mode="lines", line=dict(color=col, width=2.2)))
            th = d.chosen.iloc[0]
            near = d.iloc[(d.threshold - th).abs().argsort()[:1]]
            fig.add_trace(go.Scatter(x=[th], y=near.f1, mode="markers", marker=dict(color=col, size=10, symbol="diamond"), showlegend=False,
                                     hovertemplate=f"{lab[rid]}: порог {th:.2f}<extra></extra>"))
        fig.update_layout(title="F1 от порога события (◆ — выбранный порог)", height=380, yaxis=dict(range=[0, 1.02]), xaxis_title="порог", legend=dict(font=dict(size=10)), **LAYOUT)
        c2.plotly_chart(fig, width="stretch", key="an_curves")
        st.caption("Широкое плато F1 — порог устойчив; узкий пик — на новых данных качество будет прыгать. Порог для скрытого набора фиксируется заранее.")


# ============================================================================================== влияние параметров
@st.cache_data(show_spinner="попарные сравнения (бутстрэп)…")
def _pairwise(ids: tuple, stamp: float) -> pd.DataFrame:
    return CP.pairwise(list(ids))


def _effects(sel: pd.DataFrame, cat: pd.DataFrame) -> None:
    ev = cat[cat["mode"] == "events"]
    pool = ev[ev.gt_fingerprint.fillna("—") == st.session_state.get("an_gt", "—")] if len(ev) else ev
    var = CP.varying(pool)
    if not var:
        st.info("В журнале (с этим эталоном) нет прогонов, различающихся параметрами.")
    else:
        c = st.columns([3, 2])
        param = c[0].selectbox("Параметр", var, key="an_eff_param", format_func=lambda p: p.split(".", 1)[1] + ("  (опция профиля)" if p.startswith("opt.") else ""))
        metric = c[1].selectbox("Метрика", ["f1", "precision", "recall", "fp_per_hour"], key="an_eff_metric")
        v = pool[param].map(lambda x: "по умолчанию" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))
        fig = px.strip(pool.assign(значение=v, подпись=CP.label(pool)), x="значение", y=metric, hover_name="подпись", color="значение", color_discrete_sequence=PALETTE)
        fig.update_traces(marker_size=13)
        fig.update_layout(height=320, showlegend=False, title=f"{param.split('.', 1)[1]} → {metric} по журналу ({len(pool)} прогонов)", **LAYOUT)
        st.plotly_chart(fig, width="stretch", key="an_eff_strip")
        c1, c2 = st.columns(2)
        c1.markdown("**Сводка по значениям**")
        c1.dataframe(CP.effect(pool, param, metric), hide_index=True, width="stretch")
        c2.markdown("**Честные пары** — прогоны, различающиеся только этим параметром")
        pairs = CP.controlled_pairs(pool, param)
        if pairs.empty:
            c2.caption("Таких пар нет: прогоны различаются и другими параметрами. Сделайте серию вариантов во вкладке «Конструктор».")
        else:
            c2.dataframe(pairs, hide_index=True, width="stretch", column_config={"разница": st.column_config.NumberColumn("разница F1", format="%+.3f")})
        st.caption("Сводка по журналу — обзор, а не эксперимент: прогоны различаются ещё и другими параметрами. Вывод делайте по честным парам и попарному сравнению ниже.")
    st.divider()
    st.markdown("**Попарное сравнение выбранных подходов** (строка − столбец, п.п. F1, парный бутстрэп по общим клипам; ↑/↓ — различие значимо, ≈ — в пределах шума)")
    ids = tuple(sel.id.tolist()[:6])
    if len(ids) < 2:
        st.caption("Выберите хотя бы два подхода выше (не больше 6).")
        return
    if st.button(f"Посчитать для {len(ids)} подходов", key="an_pw_go") or st.session_state.get("an_pw_ids") == ids:
        st.session_state["an_pw_ids"] = ids
        m = _pairwise(ids, UC.file_mtime(XP.EXP_DIR))
        lab = dict(zip(sel.id, sel["подпись"]))
        st.dataframe(m.rename(index=lab, columns=lab), width="stretch")


# ============================================================================================== классификаторы
def _classifiers() -> None:
    d = CP.classifier_studies()
    st.markdown("**Диагностики спецификаций** («Задачи → Обучение»)")
    if d.empty:
        st.caption("Диагностик ещё не было.")
    else:
        st.dataframe(d, hide_index=True, width="stretch",
                     column_config={"AUC_oof": st.column_config.ProgressColumn("AUC oof", min_value=0.5, max_value=1, format="%.3f"),
                                    "AUC_train": st.column_config.NumberColumn("AUC train", format="%.3f"), "разрыв": st.column_config.NumberColumn("разрыв", format="%.3f"),
                                    "перестановка": st.column_config.NumberColumn("перестановка", format="%.2f", help="AUC при перемешанных метках: должна быть ≈ 0.5")})
    o = CP.oof_studies()
    st.markdown("**Честная оценка наборов признаков** (`sd oof-eval`, вложенный подбор порогов)")
    if o.empty:
        st.caption("Оценок ещё не было («Задачи → Результаты»).")
        return
    files = sorted(o["файл"].unique(), reverse=True)
    pick = st.multiselect("Результаты", files, default=files[:3], key="an_oof_files")
    v = o[o["файл"].isin(pick)]
    ycol = next((c for c in ("f1_вложенный_медиана", "F1 вложенный") if c in v), None)
    if ycol and len(v):
        fig = px.bar(v, x="набор", y=ycol, color="файл", barmode="group", color_discrete_sequence=PALETTE,
                     error_y=(v["f1_вложенный_макс"] - v[ycol]) if "f1_вложенный_макс" in v else None,
                     error_y_minus=(v[ycol] - v["f1_вложенный_мин"]) if "f1_вложенный_мин" in v else None)
        fig.update_layout(title="F1 событий при честном (вложенном) подборе порогов — ожидаемое качество", height=360, yaxis=dict(range=[0, 1]), **LAYOUT)
        st.plotly_chart(fig, width="stretch", key="an_oof")
    st.dataframe(v, hide_index=True, width="stretch")


def render() -> None:
    UC.style()
    cat = catalog()
    UC.header("Аналитика", "Сравнение подходов по журналу прогонов: что лучше, где именно и благодаря чему.",
              [(f"прогонов: {len(cat)}", ""), (f"закреплено: {int(cat.pinned.sum()) if len(cat) else 0}", "")])
    sec = UC.sections(SECTIONS, "an_section")
    if sec == "Журнал":
        _journal(cat)
        return
    if sec == "Классификаторы":
        _classifiers()
        return
    sel = _filter_bar(cat)
    st.divider()
    {"Сравнение подходов": lambda: _approaches(sel, cat), "По клипам": lambda: _by_clip(sel), "Ошибки и порог": lambda: _errors_threshold(sel),
     "Влияние параметров": lambda: _effects(sel, cat)}[sec]()
