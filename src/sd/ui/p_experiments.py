"""Вкладка «Эксперименты»: журнал всех прогонов оценки, серии «что если» по профилю и метки циклов из эталона.

Разделы:
  Журнал             — все прогоны с фильтрами; открыть результат / разобрать по кэшу / удалить (с подтверждением)
  Варианты профиля   — строки «имя / параметр / значение» → серия оценок фоновой задачей; сводка с интервалами
  Метки циклов       — метки циклов (затяжка / не затяжка) из эталона событий по выбранному прогону
"""
from __future__ import annotations

import json
import time

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sd import experiments as XP
from sd import library as LIB
from sd import profiles as PR
from sd import tasks as T
from sd.paths import from_portable, portable
from sd.ui import common as UC
from sd.ui import nav

SECTIONS = ["Журнал", "Варианты профиля", "Метки циклов из эталона"]


def _journal() -> None:
    runs = UC.runs_df()
    if runs.empty:
        st.info("Журнал пуст: запустите оценку («Валидация → Запуск»).")
        return
    c = st.columns([2, 2, 2, 2])
    names = c[0].multiselect("Профиль", sorted(runs.name.dropna().unique()), key="xp_f_name")
    models = c[1].multiselect("Классификатор", sorted(runs.cycle_model.dropna().unique()), key="xp_f_cls")
    mode = c[2].multiselect("Эталон", sorted(runs["mode"].dropna().unique()), key="xp_f_mode")
    only_ok = c[3].checkbox("без сбоев", key="xp_f_ok")
    v = runs
    if names:
        v = v[v.name.isin(names)]
    if models:
        v = v[v.cycle_model.isin(models)]
    if mode:
        v = v[v["mode"].isin(mode)]
    if only_ok:
        v = v[v.failed == 0]
    cols = ["created", "name", "mode", "role", "clips", "failed", "f1", "ci_lo", "ci_hi", "precision", "recall", "fp_per_hour", "threshold", "pose", "cycle_model", "dirs", "id"]
    sel = st.dataframe(v[cols], hide_index=True, on_select="rerun", selection_mode="single-row", key="xp_tbl",
                       column_config={"f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"), "failed": st.column_config.NumberColumn("сбоев")})
    rows = sel["selection"]["rows"] if sel and sel.get("selection") else []
    if not rows:
        st.caption("Выберите строку: открыть результат, разобрать по кэшу или удалить. Сравнение двух прогонов — «Валидация → Сравнение».")
        return
    rid = v.iloc[rows[0]].id
    c = st.columns(3)
    if c[0].button("Открыть результат и PDF", key="xp_open"):
        nav.go("validation", val_section="Результат и PDF", val_run_prefer=rid)
    if c[1].button("Разобрать по кэшу", key="xp_cache"):
        nav.go("validation", val_section="Анализ кэша", ca_run_prefer=rid)
    with c[2]:
        UC.confirm("Удалить прогон", "xp_del", f"Прогон {rid} будет удалён из журнала (отчёты и сводки в его папке тоже). Кэш поз и событий останется.",
                   lambda: XP.delete_run(rid) and f"удалён {rid}")


def _variants() -> None:
    dirs = LIB.discover_dirs()
    if dirs.empty:
        st.info("Папок с видео нет.")
        return
    names = PR.list_profiles()
    if not names:
        st.info("Нет сохранённых профилей.")
        return
    c = st.columns([3, 2, 2])
    chosen = c[0].multiselect("Папки", dirs.name.tolist(), default=[n for n in dirs.name if Path(n).name in ("курение", "лжекурение")], key="var_dirs")
    base = c[1].selectbox("Базовый профиль", names, index=names.index("mvp") if "mvp" in names else 0, key="var_base")
    policy = c[2].selectbox("Порог", ["fixed", "plateau"], key="var_policy", format_func=lambda p: {"fixed": "из профиля (честно)", "plateau": "центр плато (оптимистично)"}[p])
    st.caption("Строка = одно изменение базового профиля; одинаковое имя у нескольких строк — один вариант из нескольких правок. Параметр — ключ конфигурации (`pose.imgsz`, "
               "`events.cycle_th`, …) или опция профиля (`cycle_bundle`, `objects`, …); значение — YAML (`640`, `false`, `[a, b]`). Базовый профиль оценивается тоже.")
    ed = st.data_editor(pd.DataFrame([dict(name="fps10", param="video.process_fps", value="10")]), num_rows="dynamic", key="var_rows", hide_index=True, width="stretch")
    rows = [r for r in ed.to_dict("records") if str(r.get("name") or "").strip() and str(r.get("param") or "").strip()]
    items = []
    try:
        from sd import sweep as SW

        vs = SW.variants_from_rows(PR.load_named(base), rows)
        items.append(("ok", f"вариантов: {len(vs)} ({', '.join(v.name for v in vs)}) — каждый = полный прогон оценки; новые параметры позы пересчитывают позу (долго)"))
    except Exception as e:      # noqa: BLE001
        items.append(("error", f"параметры: {e}"))
    if not chosen:
        items.append(("error", "не выбраны папки"))
    ok = UC.checklist(items)
    if st.button("Запустить серию (фон)", type="primary", disabled=not ok, key="var_go"):
        f = T.inputs_dir() / f"variants_{time.strftime('%Y%m%d_%H%M%S')}.json"
        f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        T.start("variants", dict(dirs="\n".join(str(p) for p in dirs.set_index("name").loc[chosen, "path"]), base=base, rows=str(f), policy=policy))
        st.toast("серия поставлена в очередь задач", icon="⏳")
    UC.task_panel(("variants",), key="var_tp", limit=2, title=None)
    runs = UC.runs_df()
    if len(runs):
        recent = runs[runs.name.str.startswith(base, na=False)].head(12)
        if len(recent):
            st.markdown("**Последние прогоны вариантов этого профиля**")
            df = recent.sort_values("f1")
            fig = go.Figure(go.Bar(x=df.f1, y=df.name + " · " + df.created.str[5:16], orientation="h", text=df.f1.round(3), textposition="outside",
                                   error_x=dict(type="data", symmetric=False, array=(df.ci_hi - df.f1).fillna(0), arrayminus=(df.f1 - df.ci_lo).fillna(0))))
            fig.update_layout(height=80 + 34 * len(df), margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(range=[0, 1.05], title="Event F1 (95 %)"), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            st.caption("Интервалы отдельных вариантов почти всегда перекрываются; вывод «лучше / не лучше» — парным сравнением («Валидация → Сравнение»).")


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
    st.header("Эксперименты")
    if "xp_section" not in st.session_state:
        st.session_state["xp_section"] = SECTIONS[0]
    sec = st.radio("Раздел", SECTIONS, horizontal=True, label_visibility="collapsed", key="xp_section")
    {"Журнал": _journal, "Варианты профиля": _variants, "Метки циклов из эталона": _cycle_labels}[sec]()
