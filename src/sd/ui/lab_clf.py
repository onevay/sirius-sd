"""Лаборатория → «Классификатор»: быстрый подбор признаков и членов ансамбля с честной кросс-валидацией (фолды по видео или сценам), разбор ошибок и попыток."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from sd import active as AC
from sd import catalog as CAT
from sd import classifier as CL
from sd import cycle_models as CM
from sd import oof_eval as O
from sd.feature_auc import _auc
from sd.ui import common as UC
from sd.ui import lab_common as LC


def _candidate_features(tab: pd.DataFrame) -> list[str]:
    known = list(dict.fromkeys(c for s in CM.SETS.values() for c in s))
    num = [c for c in tab.columns if pd.api.types.is_numeric_dtype(tab[c]) and c not in CM.CONFOUNDS and c not in ("y", "tid", "_w") and c not in known]
    return [c for c in known if c in tab.columns] + sorted(num)


def _start_spec(tab: pd.DataFrame) -> dict | None:
    src = st.radio("Начать с", ["набор признаков", "пакет", "сохранённая спецификация"], horizontal=True, key="lc_src")
    if src == "набор признаков":
        name = st.selectbox("Набор", list(CM.SETS), index=list(CM.SETS).index("obj_hold_zsd"), key="lc_set", help="заданы заранее; описание — в cycle_models.py")
        return CL.default_spec(name, ("lr", "gb", "nn"), tab)
    if src == "пакет":
        b = [c.id for c in CAT.cycle_bundles()]
        if not b:
            st.info("Пакетов нет.")
            return None
        return AC.spec_from_bundle(st.selectbox("Пакет", b, key="lc_bundle"))
    names = CL.list_specs()
    if not names:
        st.info("Сохранённых спецификаций нет — начните с набора признаков или пакета.")
        return None
    return CL.load_spec(st.selectbox("Спецификация", names, key="lc_specname"))


def _edit(spec: dict, tab: pd.DataFrame) -> dict:
    ver = f"{spec.get('name')}_{len(CL.spec_features(spec))}"
    cands = _candidate_features(tab)
    feats0 = [c for c in CL.spec_features(spec) if c in cands]
    c = st.columns([5, 3])
    feats = c[0].multiselect("Признаки", cands, default=feats0, key=f"lc_f_{ver}", help="короткие наборы устойчивее: на ~100 затяжках длинный список переобучается")
    cov = LC.coverage(tab, feats)
    poor = cov[cov.заполнено < 0.5]
    if len(poor):
        c[1].warning("заполнено < 50 %: " + ", ".join(poor.признак) + " — посчитайте «Признаки» (Задачи) для этого прогона")
    kinds = st.multiselect("Члены ансамбля", ["lr", "gb", "nn"], default=["lr", "gb"], key=f"lc_k_{ver}", format_func={"lr": "логрегрессия", "gb": "LightGBM", "nn": "MLP"}.get)
    mem = CM.members_for(feats, tuple(kinds)) if feats and kinds else {}
    with st.expander("Гиперпараметры"):
        cc = st.columns(3)
        for n, m in mem.items():
            p = m.setdefault("params", {})
            if m["kind"] == "logreg":
                p["C"] = cc[0].select_slider(f"{n}: C", [0.003, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0], value=p.get("C", 0.3), key=f"lc_C_{n}_{ver}")
            elif m["kind"] == "lgbm":
                p.update(num_leaves=cc[1].slider("gb: листьев", 2, 16, 4, key=f"lc_nl_{ver}"), n_estimators=cc[1].slider("gb: деревьев", 20, 400, 150, 10, key=f"lc_ne_{ver}"),
                         min_child_samples=cc[1].slider("gb: мин. в листе", 2, 30, 8, key=f"lc_mc_{ver}"), lambda_l2=cc[1].slider("gb: L2", 0.0, 20.0, 5.0, 0.5, key=f"lc_l2_{ver}"))
            else:
                p.update(hidden=cc[2].slider("nn: нейронов", 2, 32, 8, key=f"lc_h_{ver}"), alpha=cc[2].select_slider("nn: alpha", [0.01, 0.1, 0.5, 1.0, 3.0], value=1.0, key=f"lc_a_{ver}"))
    cv = st.columns(4)
    by_scene = cv[0].toggle("Фолды по сценам", value=True, key="lc_scene", help="клипы одной камеры/человека не делятся между обучением и проверкой")
    reps = cv[1].select_slider("Повторов", [1, 2, 3, 5], value=3, key="lc_reps")
    splits = cv[2].select_slider("Фолдов", [3, 4, 5, 8], value=5, key="lc_spl")
    return {**spec, "features": feats, "members": mem, "cv": {**CL.DEFAULT_CV, "by_scene": by_scene, "repeats": int(reps), "n_splits": int(splits)}}


def _curves(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, th = precision_recall_curve(y, p)
    return fpr, tpr, pr, rc, th


def _fig_roc_pr(runs: list[dict]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC", "Точность–полнота"))
    for r in runs:
        fpr, tpr, pr, rc, _ = _curves(r["y"], r["p"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=r["name"], mode="lines", legendgroup=r["name"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=rc, y=pr, name=r["name"], mode="lines", legendgroup=r["name"], showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="gray"), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.update_xaxes(title="FPR", row=1, col=1)
    fig.update_xaxes(title="полнота", row=1, col=2)
    fig.update_yaxes(title="TPR", row=1, col=1)
    fig.update_yaxes(title="точность", row=1, col=2)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def _calibration(y, p, bins: int = 6):
    import plotly.graph_objects as go

    q = pd.qcut(p, q=min(bins, len(np.unique(p))), duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "b": q}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).reset_index()
    fig = go.Figure(go.Scatter(x=g.p, y=g.y, mode="lines+markers", text=g.n, name="модель", hovertemplate="оценка %{x:.2f} → доля затяжек %{y:.2f} (n=%{text})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="gray"), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title="средняя оценка в группе", range=[0, 1])
    fig.update_yaxes(title="доля настоящих затяжек", range=[0, 1])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def _thresholds(y, p) -> pd.DataFrame:
    rows = []
    for th in np.arange(0.1, 0.91, 0.1):
        pred = p >= th
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        rows.append(dict(порог=round(float(th), 1), точность=round(pr, 3), полнота=round(rc, 3), F1=round(2 * pr * rc / max(pr + rc, 1e-9), 3), TP=tp, FP=fp, FN=fn))
    return pd.DataFrame(rows)


def render() -> None:
    run = LC.pick_run("Источник таблицы циклов (прогон оценки)", "lc_run", help="циклы тех же клипов и той же конфигурации, что в прогоне")
    if not run:
        return
    tab = LC.table_for(run, LC.gestures_stamp())
    lt = tab[tab.y.notna()].reset_index(drop=True)
    m = st.columns(4)
    m[0].metric("Циклов всего", len(tab))
    m[1].metric("Размечено", len(lt))
    m[2].metric("Затяжек", int(lt.y.sum()) if len(lt) else 0)
    m[3].metric("Видео / сцен", f"{lt.video.nunique()} / {lt.video.map(O.scene_group).nunique()}" if len(lt) else "0")
    spec0 = _start_spec(tab)
    if spec0 is None:
        return
    spec = _edit(spec0, tab)
    problems = CL.validate(spec, lt) if spec["members"] else ["не выбраны признаки или члены ансамбля"]
    warns = CL.warnings_for(lt, spec) if not problems else [("error", p) for p in problems]
    ok = UC.checklist(warns, title="Проверка перед оценкой")
    name = st.text_input("Название попытки", f"попытка {len(st.session_state.get('lc_runs', [])) + 1}", key="lc_name")
    if st.button("Оценить (кросс-валидация)", type="primary", disabled=not ok, key="lc_go"):
        with st.spinner("Кросс-валидация…"):
            oof = CL.cv_predict(lt, spec)
        y = lt.y.to_numpy(float).astype(int)
        groups = CL._groups(lt, bool(spec["cv"]["by_scene"]))
        lo, hi = CL._boot_auc(y, oof["__ensemble__"], groups, n=300)
        run_rec = dict(name=name, spec=spec, p=oof["__ensemble__"], members={k: v for k, v in oof.items() if k != "__ensemble__"}, y=y, auc=_auc(y, oof["__ensemble__"]), lo=lo, hi=hi,
                       ap=float(average_precision_score(y, oof["__ensemble__"])), run=run, rows=lt[["video", "tid", "start", "peak_t", "label"]].copy())
        lst = st.session_state.setdefault("lc_runs", [])
        lst.append(run_rec)
        del lst[:-8]
    runs = [r for r in st.session_state.get("lc_runs", []) if r["run"] == run and len(r["y"]) == len(lt)]
    if not runs:
        st.caption("Нажмите «Оценить» — результаты попыток накапливаются ниже, чтобы их сравнивать.")
        return
    cur = runs[-1]
    c = st.columns(4)
    c[0].metric("AUC (out-of-fold)", f"{cur['auc']:.3f}")
    c[1].metric("95 % интервал по видео", f"{cur['lo']:.2f}–{cur['hi']:.2f}")
    c[2].metric("Средняя точность (AP)", f"{cur['ap']:.3f}")
    prev = runs[-2] if len(runs) > 1 else None
    c[3].metric("К предыдущей попытке", f"{cur['auc'] - prev['auc']:+.3f}" if prev else "—", help="разница меньше ширины интервала — шум")
    t_cmp, t_thr, t_err, t_save = st.tabs(["Сравнение попыток", "Пороги и калибровка", "Ошибки", "Сохранить и обучить"])
    with t_cmp:
        tbl = pd.DataFrame([dict(попытка=r["name"], признаков=len(CL.spec_features(r["spec"])), члены=",".join(r["spec"]["members"]), сцены=r["spec"]["cv"]["by_scene"], AUC=round(r["auc"], 3),
                                 нижн=round(r["lo"], 3), верхн=round(r["hi"], 3), AP=round(r["ap"], 3)) for r in runs])
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.plotly_chart(_fig_roc_pr(runs), width="stretch", key="lc_roc")
        mem = pd.DataFrame([dict(член=k, AUC=round(_auc(cur["y"], v), 3)) for k, v in cur["members"].items()])
        st.markdown("**Члены ансамбля текущей попытки**")
        st.dataframe(mem, hide_index=True, width="stretch")
    with t_thr:
        st.dataframe(_thresholds(cur["y"], cur["p"]), hide_index=True, width="stretch")
        st.plotly_chart(_calibration(cur["y"], cur["p"]), width="stretch", key="lc_cal")
        st.caption("Калибровка: точки на диагонали — оценка 0.7 означает ~70 % затяжек. Пакет дополнительно калибруется изотонически (опция `calibrate` в спецификации).")
    with t_err:
        d = cur["rows"].assign(оценка=cur["p"], y=cur["y"])
        d["ошибка"] = np.abs(d.y - d["оценка"])
        d = d.sort_values("ошибка", ascending=False, ignore_index=True)
        st.caption("Сверху самые вопиющие ошибки: либо модель не права, либо метка. Выберите строку — покажем кадры; метку можно поправить в «Задачи → Метки жестов».")
        st.dataframe(d.head(30).round(3), hide_index=True, width="stretch", key="lc_err", on_select="rerun", selection_mode="single-row")
        sel = st.session_state.get("lc_err", {}).get("selection", {}).get("rows", [])
        if sel:
            row = d.iloc[sel[0]]
            full = tab[(tab.video == row.video) & (tab.start.round(3) == round(float(row.start), 3)) & (tab.tid == row.tid)]
            img = LC.preview(run, full.iloc[0] if len(full) else row)
            if img is not None:
                st.image(img, caption=f"{row.video} · {row.peak_t:.1f} с · метка {row.label} · оценка {row['оценка']:.2f}", width="stretch")
            else:
                st.caption("кадры недоступны (нет видео или кэша позы)")
    with t_save:
        sname = st.text_input("Имя спецификации", cur["name"].replace(" ", "_"), key="lc_sname")
        c1 = st.columns(3)
        if c1[0].button("Сохранить спецификацию", key="lc_save"):
            try:
                f = CL.save_spec({**cur["spec"], "name": sname, "description": f"лаборатория: AUC oof {cur['auc']:.3f}"})
                st.success(f"сохранено: {f}")
            except Exception as e:      # noqa: BLE001
                st.error(str(e))
        saved = sname in CL.list_specs()
        UC.task_button("clf_diagnose", dict(spec=sname, run=run, perm=3, importance=True), "Диагностика (фоном)", key="lc_diag", disabled=not saved, primary=False,
                       help="сначала сохраните спецификацию")
        bname = st.text_input("Имя пакета", "cycle_lab", key="lc_bname")
        UC.task_button("clf_train", dict(spec=sname, bundle=bname, run=run), "Обучить пакет (фоном)", key="lc_train", disabled=not saved)
        UC.task_panel(("clf_diagnose", "clf_train"), key="lc_tasks", limit=3)
