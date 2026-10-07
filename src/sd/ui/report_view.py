"""Отображение отчёта оценки: F1 как главная цифра, остальные метрики вокруг неё, кривая порога, клипы, ошибки."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sd import evaluation as EV
from sd import runner as RN
from sd.ui import nav

BLUE, ORANGE, GREEN, GRAY, RED = "#0072B2", "#E69F00", "#009E73", "#8a8f98", "#D55E00"


def _pct(x: float | None) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.3f}"


def headline(rep: EV.Report) -> None:
    m, b, ci = rep.metrics, rep.budget, rep.ci
    c = st.columns([1.5, 1, 1, 1.2, 1.2, 1.2])
    c[0].metric("Event F1", f"{m['f1']:.3f}", f"{m['f1'] - b['target']:+.3f} к цели {b['target']:.2f}", delta_color="normal")
    f1ci = ci.get("f1") or (None, None)
    c[0].caption("интервал 95%: " + (f"{f1ci[0]:.2f}–{f1ci[1]:.2f}" if f1ci[0] is not None else "мало клипов для интервала"))
    c[1].metric("Precision", _pct(m["precision"]))
    c[2].metric("Recall", _pct(m["recall"]))
    c[3].metric("TP · FP · FN", f"{m['tp']} · {m['fp']} · {m['fn']}")
    c[4].metric("Ложных тревог/час", "—" if m["fp_per_hour"] is None else f"{m['fp_per_hour']:.1f}", help="по клипам без позитивов в эталоне")
    c[5].metric("Задержка, с", "—" if m["median_latency"] is None else f"{m['median_latency']:.1f}", help="медиана max(0, start_pred − start_gt)")
    st.progress(min(max(m["f1"] / b["target"], 0.0), 1.0) if b["target"] else 0.0, text=f"{m['f1']:.3f} из {b['target']:.2f}" + (" — цель достигнута" if b["reached"] else ""))
    if b["reached"]:
        st.success(f"Цель F1 ≥ {b['target']:.2f} достигнута. Запас: ещё {int(b['allowed'] - b['errors'])} ошибок при текущем числе TP.")
    else:
        st.warning(f"До цели F1 ≥ {b['target']:.2f}: допустимо ошибок {int(b['allowed'])}, сейчас {b['errors']} (FP {m['fp']}, FN {m['fn']}). "
                   f"Исправить одну FP = {b['gain_fix_fp'] * 100:+.1f} п.п. F1, одну FN = {b['gain_fix_fn'] * 100:+.1f} п.п.")
    for n in rep.notes:
        st.caption("• " + n)


def curve_chart(rep: EV.Report) -> go.Figure:
    cur, pl, th = rep.curve, rep.plateau, rep.settings["threshold"]
    fig = go.Figure()
    if pl.get("lo") is not None:
        fig.add_vrect(x0=pl["lo"], x1=pl["hi"], fillcolor=BLUE, opacity=0.10, line_width=0)
    for col, name, color, w in (("f1", "F1", BLUE, 3), ("precision", "Precision", ORANGE, 1.8), ("recall", "Recall", GREEN, 1.8)):
        fig.add_trace(go.Scatter(x=cur.threshold, y=cur[col], name=name, mode="lines", line=dict(color=color, width=w), hovertemplate=f"{name} %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=rep.settings["target_f1"], line_dash="dash", line_color=GRAY, annotation_text=f"цель {rep.settings['target_f1']:.2f}", annotation_position="top left")
    fig.add_vline(x=th, line_color=RED, line_width=2, annotation_text=f"порог {th:.2f}", annotation_position="bottom right")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", showlegend=True, legend=dict(orientation="h", y=1.12, x=0), xaxis_title="порог confidence события", yaxis=dict(range=[0, 1.02], title=None),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.2)")
    return fig


def _open_button(df: pd.DataFrame, key: str, rid: str | None, clip_col: str = "clip_id", t_col: str | None = None) -> None:
    sel = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="single-row", key=key)
    rows = sel["selection"]["rows"] if sel and sel.get("selection") else []
    if rows and rid and st.button("Открыть в просмотре", key=key + "_open"):
        r = df.iloc[rows[0]]
        nav.go("view", view_src="Эксперимент", view_run=rid, view_clip=r[clip_col])


def show(rep: EV.Report, meta: dict, rid: str | None, events: pd.DataFrame | None = None) -> None:
    headline(rep)
    t1, t2, t3, t4 = st.tabs(["Порог", "Клипы", "Ошибки", "Запуск"])
    with t1:
        pl = rep.plateau
        st.plotly_chart(curve_chart(rep), width="stretch")
        if pl.get("center") is not None:
            st.caption(f"Лучший F1 {pl['best_f1']:.3f} при пороге {pl['best_threshold']:.2f}; плато (F1 в пределах {pl['tol']:.2f} от лучшего) {pl['lo']:.2f}–{pl['hi']:.2f}, "
                       f"центр {pl['center']:.2f}. Выбирайте центр плато, а не вершину.")
        with st.expander("Таблица кривой"):
            st.dataframe(rep.curve.round(3), hide_index=True)
    with t2:
        pc = rep.per_clip
        if pc.empty:
            st.info("Нет данных по клипам.")
        else:
            only = st.checkbox("Только клипы с ошибками", value=True, key=f"pc_only_{rid}")
            _open_button(pc[pc.status != "ok"] if only else pc, f"pc_{rid}", rid)
    with t3:
        er = rep.errors
        if er.empty:
            st.success("Ошибок нет.")
        else:
            cnt = er.groupby(["kind", "cause_ru"]).size().reset_index(name="n").sort_values("n")
            fig = go.Figure(go.Bar(x=cnt.n, y=cnt.cause_ru, orientation="h", marker_color=[RED if k == "FP" else ORANGE for k in cnt.kind], text=cnt.kind + " · " + cnt.n.astype(str), textposition="outside"))
            fig.update_layout(height=60 + 38 * len(cnt), margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(visible=False), yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            _open_button(er.drop(columns=["cause"]).round(2), f"er_{rid}", rid)
    with t4:
        d = meta.get("describe", {})
        st.dataframe(pd.DataFrame([d]), hide_index=True)
        st.caption(f"отпечаток профиля {meta.get('fingerprint')} · эталон {meta.get('gt_fingerprint') or '—'} · режим {rep.mode} · набор {meta.get('role')} · "
                   f"время прогона {meta.get('total_seconds', '—')} с · клипов {len(meta.get('clips', []))}")
        st.write("Папки: " + ", ".join(meta.get("dirs", [])))
        c1, c2 = st.columns(2)
        th = rep.settings["threshold"]
        if events is not None and len(events):
            c1.download_button("Предсказания выше порога (CSV)", events[events.confidence >= th][RN.EVENT_COLUMNS].to_csv(index=False).encode("utf-8-sig"), file_name="preds.csv")
        c2.download_button("Отчёт (JSON)", pd.Series({**rep.metrics}).to_json(force_ascii=False).encode("utf-8"), file_name="metrics.json")
        with st.expander("Профиль целиком"):
            st.json(meta.get("profile", {}))
