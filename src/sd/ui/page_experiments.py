"""Страница «Эксперименты»: журнал и парное сравнение прогонов, серия вариантов профиля («что если»), метки циклов из эталона для обучения."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sd import experiments as XP
from sd import gt as GT
from sd import library as LIB
from sd import profiles as PR
from sd import sweep as SW
from sd.paths import from_portable
from sd.ui import page_eval

BLUE = "#0072B2"


def _journal() -> None:
    runs = XP.list_runs()
    if runs.empty:
        st.info("Журнал пуст: запустите оценку на странице «Оценка».")
        return
    cols = ["created", "name", "mode", "role", "clips", "f1", "ci_lo", "ci_hi", "precision", "recall", "fp_per_hour", "threshold", "reached", "pose", "cycle_model", "vlm", "id"]
    sel = st.dataframe(runs[cols], hide_index=True, on_select="rerun", selection_mode="multi-row", key="xp_hist",
                       column_config={"f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f")})
    rows = sel["selection"]["rows"] if sel and sel.get("selection") else []
    st.caption("Выберите две строки, чтобы сравнить прогоны на общих клипах (парный бутстрэп).")
    if len(rows) == 2:
        a, b = runs.iloc[rows[0]]["id"], runs.iloc[rows[1]]["id"]
        try:
            r = SW.compare_saved(a, b)
        except Exception as e:
            st.warning(str(e))
            return
        if r["diff"] is None:
            st.warning("Мало общих клипов для сравнения (нужно ≥ 3).")
            return
        st.metric(f"F1: второй − первый ({r['clips']} общих клипов)", f"{r['diff'] * 100:+.1f} п.п.", f"95%: {r['lo'] * 100:+.1f} … {r['hi'] * 100:+.1f}")
        verdict = "второй профиль лучше" if r["lo"] > 0 else ("первый профиль лучше" if r["hi"] < 0 else "различие в пределах шума")
        st.write(f"**{verdict}**; вероятность «второй лучше»: {r['p_better']:.0%}." + ("" if r["same_gt"] else " Внимание: эталон в прогонах различается."))


def _variants() -> None:
    gt = GT.load()
    dirs = LIB.discover_dirs()
    if dirs.empty:
        st.info("Папок с видео нет.")
        return
    c = st.columns([3, 2, 2])
    chosen = c[0].multiselect("Папки", dirs.name.tolist(), default=st.session_state.get("eval_dirs") or dirs.name.tolist(), key="xp_dirs", placeholder="Выберите папки")
    names = ["default", *[n for n in PR.list_profiles() if n != "default"]]
    base_name = c[1].selectbox("Базовый профиль", names, key="xp_base")
    policy = c[2].selectbox("Порог", ["plateau", "fixed"], format_func=lambda p: page_eval.POLICIES[p], key="xp_policy")
    base = PR.default_profile() if base_name == "default" and "default" not in PR.list_profiles() else PR.load(base_name)
    st.caption("Строка = одно изменение базового профиля. Одинаковое имя у нескольких строк — одно изменение из нескольких параметров. Параметр: ключ конфига (`pose.imgsz`, `events.cycle_th`, …) "
               "или опция профиля (`cycle_bundle`, `photo_bundle`, `objects`, `vlm_model`, `vlm_mode`, `grey`). Значение — YAML: `640`, `false`, `models/cycle/x`, `[a, b]`.")
    ed = st.data_editor(pd.DataFrame([dict(name="imgsz640", param="pose.imgsz", value="640")]), num_rows="dynamic", key="xp_rows", hide_index=True, width="stretch")
    if st.button("Запустить варианты", type="primary", disabled=not chosen):
        try:
            vs = SW.variants_from_rows(base, ed.to_dict("records"))
        except Exception as e:
            st.error(f"Параметры: {e}")
            return
        paths = dirs.set_index("name").loc[chosen, "path"].tolist()
        bar = st.progress(0.0, text="старт")
        outs = []
        for i, v in enumerate(vs):
            try:
                outs.append(SW.run_variants(paths, [v], policy=policy, gt_df=gt, progress=lambda k, n, m, i=i, v=v: bar.progress((i + k / max(n, 1)) / len(vs), text=f"{v.name}: {k}/{n} · {m}"))[0])
            except Exception as e:
                st.error(f"{v.name}: {e}")
        bar.empty()
        st.session_state["xp_last"] = [o.run_id for o in outs]
    last = st.session_state.get("xp_last")
    if not last:
        return
    rows = []
    for rid in last:
        rep, meta, _, _ = XP.load_run(rid)
        ci = rep.ci.get("f1") or (None, None)
        rows.append(dict(вариант=meta["name"], F1=rep.metrics["f1"], lo=ci[0], hi=ci[1], precision=rep.metrics["precision"], recall=rep.metrics["recall"], FP_в_час=rep.metrics["fp_per_hour"],
                         порог=rep.metrics["threshold"], id=rid))
    df = pd.DataFrame(rows).sort_values("F1")
    fig = go.Figure(go.Bar(x=df.F1, y=df["вариант"], orientation="h", marker_color=BLUE, text=df.F1.round(3), textposition="outside",
                           error_x=dict(type="data", symmetric=False, array=(df.hi - df.F1).fillna(0), arrayminus=(df.F1 - df.lo).fillna(0))))
    fig.update_layout(height=80 + 46 * len(df), margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(range=[0, 1.05], title="Event F1 (95% интервал)"), yaxis_title=None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.sort_values("F1", ascending=False), hide_index=True)
    if len(df) >= 2:
        st.caption("Интервалы отдельных вариантов перекрываются почти всегда; для вывода «лучше / не лучше» выберите два прогона в «Журнале» — там парное сравнение.")


def _labels() -> None:
    runs = XP.list_runs()
    runs = runs[runs["mode"] == "events"] if len(runs) else runs
    if runs.empty:
        st.info("Нужен прогон оценки по эталону событий («Оценка», режим «ручная разметка»).")
        return
    rid = st.selectbox("Прогон", runs.id.tolist(), format_func=lambda i: f"{runs.set_index('id').loc[i, 'name']} · {runs.set_index('id').loc[i, 'created']}")
    neg = st.checkbox("Циклы вне размеченных интервалов просмотренного клипа считать негативами", value=True,
                      help="отключите, если в клипах размечены не все курения: иначе неразмеченное курение станет ложным негативом")
    iou_th = st.slider("Порог IoU рамок", 0.1, 0.9, 0.3, 0.05)
    _, meta, _, gt = XP.load_run(rid)
    from sd import dataset as DS
    from sd import labels_from_gt as LG
    from sd.tracks import Tracks

    parts, missing = [], []
    boxes: dict[str, Tracks] = {}
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
    if st.button("Записать в labels/cycle_labels.csv", type="primary", disabled=labels.empty):
        st.success(f"записано: {DS.save_labels_bulk(labels)}. Дальше: «Этапы» → «Обучение» или `sd train-bundle`. Ручные метки не затронуты; повторная запись заменяет прежние метки из эталона.")


def render() -> None:
    sec = st.radio("Раздел", ["Журнал и сравнение", "Варианты профиля", "Метки циклов из эталона"], horizontal=True, label_visibility="collapsed")
    {"Журнал и сравнение": _journal, "Варианты профиля": _variants, "Метки циклов из эталона": _labels}[sec]()
