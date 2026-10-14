"""Лаборатория → «Временные ряды»: модели по форме рядов d_t, скорости, угла локтя и др. Обучение, честная оценка и смесь с табличным классификатором."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from sd import classifier as CL
from sd import lab
from sd import oof_eval as O
from sd import profiles as PR
from sd import tsmodels as TS
from sd.feature_auc import _auc
from sd.paths import MODELS
from sd.ui import lab_common as LC

TS_DIR = MODELS / "ts"


@st.cache_data(show_spinner="Строим ряды циклов…", max_entries=4)
def _tensor(run: str, profile: str, channels: tuple, length: int, pad: float, stamp: float):
    """Тензор циклов с метками: (X, S, y, видео, сцены, таблица циклов)."""
    tab = LC.table_for(run, stamp)
    lt = tab[tab.y.notna()].reset_index(drop=True)
    clips = O.collect_from_run(run)
    ser = TS.series_for_clips(clips, lambda rd: lab.build_cfg(rd, profile))
    X, S = TS.cycle_tensor(lt, ser, channels, length, pad)
    y = lt.y.to_numpy(float).astype(int)
    return X, S, y, lt.video.to_numpy(), lt.video.map(O.scene_group).to_numpy(), lt


def _saved() -> pd.DataFrame:
    rows = []
    for d in sorted(TS_DIR.glob("*/manifest.json")) if TS_DIR.exists() else []:
        try:
            m = json.loads(d.read_text(encoding="utf-8"))
            rows.append(dict(модель=d.parent.name, тип=m["spec"]["model"], каналы=", ".join(m["spec"]["channels"]), длина=m["spec"]["length"], AUC_oof=m.get("auc_oof"),
                             циклов=m.get("n"), обучена=m.get("created")))
        except Exception:       # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def _params(model: str) -> dict:
    p: dict = {}
    d = TS.DEFAULT_PARAMS[model]
    with st.expander("Параметры модели"):
        if model in ("stats", "rocket"):
            p["C"] = st.select_slider("Регуляризация C (меньше — сильнее)", [0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0], value=d["C"], key=f"lts_C_{model}")
        if model == "rocket":
            p["n_kernels"] = st.select_slider("Случайных ядер", [50, 100, 200, 300, 500, 800], value=d["n_kernels"], key="lts_nk")
        if model == "cnn1d":
            p["hidden"] = st.select_slider("Каналов в слоях", [8, 16, 32], value=d["hidden"], key="lts_h")
            p["epochs"] = st.slider("Эпох", 20, 200, d["epochs"], 10, key="lts_ep")
            p["weight_decay"] = st.select_slider("Штраф за вес", [0.0, 0.0001, 0.001, 0.01], value=d["weight_decay"], key="lts_wd")
    return p


def _proto_fig(X, y, channels):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pr = TS.prototypes(X, y, channels)
    fig = make_subplots(rows=1, cols=len(channels), subplot_titles=list(channels), shared_xaxes=False)
    colors = {"затяжка": "#0f766e", "не затяжка": "#dc2626"}
    fills = {"затяжка": "rgba(15,118,110,0.15)", "не затяжка": "rgba(220,38,38,0.15)"}
    for j, ch in enumerate(channels, start=1):
        for cls, g in pr[pr.channel == ch].groupby("cls"):
            fig.add_trace(go.Scatter(x=g.i, y=g["mean"], name=cls, line=dict(color=colors[cls]), showlegend=j == 1, legendgroup=cls), row=1, col=j)
            fig.add_trace(go.Scatter(x=list(g.i) + list(g.i[::-1]), y=list(g.hi) + list(g.lo[::-1]), fill="toself", line=dict(width=0), fillcolor=fills[cls],
                                     showlegend=False, hoverinfo="skip", legendgroup=cls), row=1, col=j)
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15))
    return fig


def render() -> None:
    st.markdown("Модель смотрит на **форму** рядов цикла (не на ручные признаки): окно `[начало − отступ, конец + отступ]` сжимается до фиксированной длины. "
                "Сравните с табличным классификатором и посмотрите, дают ли ряды что-то сверх него.")
    run = LC.pick_run("Источник циклов (прогон оценки)", "lts_run")
    if not run:
        return
    profs = PR.list_profiles()
    c = st.columns([2, 4, 2, 2], vertical_alignment="bottom")
    profile = c[0].selectbox("Профиль рядов", profs, index=profs.index("mvp") if "mvp" in profs else 0, key="lts_prof", help="параметры сглаживания и рта для рядов d_t")
    chans = c[1].multiselect("Каналы", list(TS.CHANNELS), default=list(TS.DEFAULT_CHANNELS), key="lts_ch", help="; ".join(f"{k} — {v}" for k, v in TS.CHANNELS.items()))
    length = c[2].select_slider("Отсчётов", [32, 48, 64, 96, 128], value=64, key="lts_len")
    pad = c[3].slider("Отступ, с", 0.0, 3.0, 1.0, 0.25, key="lts_pad")
    c2 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")
    model = c2[0].selectbox("Модель", list(TS.MODELS), index=1, key="lts_model", format_func=lambda k: f"{k} — {TS.MODELS[k]}")
    by_scene = c2[1].toggle("Фолды по сценам", value=True, key="lts_scene", help="клипы одной камеры/человека не попадают и в обучение, и в проверку")
    reps = c2[2].select_slider("Повторов CV", [1, 2, 3, 5], value=2, key="lts_rep")
    cnn_ok = TS.torch_available()
    if model == "cnn1d" and not cnn_ok:
        st.warning("torch не установлен — доступны stats и rocket.")
    params = _params(model)
    if not chans:
        st.info("Выберите хотя бы один канал.")
        return
    X, S, y, vid, scn, lt = _tensor(run, profile, tuple(chans), int(length), float(pad), LC.gestures_stamp())
    n_pos = int(y.sum())
    m = st.columns(4)
    m[0].metric("Размеченных циклов", len(y))
    m[1].metric("Затяжек", n_pos)
    m[2].metric("Видео / сцен", f"{len(set(vid))} / {len(set(scn))}")
    m[3].metric("Пропусков в рядах", f"{100 * float(np.isnan(X).mean()):.1f} %")
    if len(y) < 40 or n_pos < 15 or len(y) - n_pos < 15:
        st.error("Слишком мало размеченных циклов для честной оценки (нужно ≥ 15 каждого класса).")
        return
    with st.expander("Как выглядит затяжка: средняя форма рядов по классам (± разброс)", expanded=False):
        st.plotly_chart(_proto_fig(X, y, chans), width="stretch", key="lts_proto")
        st.caption("Если линии классов почти совпадают — по этому каналу форма не различает затяжку и не-затяжку.")
    spec = TS.TSSpec(model, list(chans), int(length), float(pad), params)
    groups = scn if by_scene else vid
    go = st.button("Обучить и оценить (кросс-валидация)", type="primary", key="lts_go", disabled=model == "cnn1d" and not cnn_ok)
    if go:
        t0 = time.time()
        with st.spinner("Кросс-валидация…"):
            oof = TS.cv_oof(X, S, y, groups, spec, repeats=int(reps))
        lo, hi = CL._boot_auc(y, oof, groups, n=300)
        st.session_state["lts_last"] = dict(oof=oof, spec=spec.to_dict(), auc=_auc(y, oof), lo=lo, hi=hi, y=y, groups=groups, run=run, sec=time.time() - t0, by_scene=by_scene, X=X, S=S)
        st.session_state.setdefault("lts_attempts", []).append(dict(время=LC.short_time(), модель=model, каналы=",".join(chans), длина=int(length), сцены=by_scene,
                                                                      AUC=round(_auc(y, oof), 3), нижн=round(lo, 3), верхн=round(hi, 3)))
    last = st.session_state.get("lts_last")
    if not last or last["run"] != run or len(last["y"]) != len(y):
        st.caption("Нажмите кнопку — результат появится здесь.")
        _attempts_and_saved(None, None, None, None, None)
        return
    r = st.columns(3)
    r[0].metric("AUC (out-of-fold)", f"{last['auc']:.3f}", help="оценка цикла моделью, не видевшей это видео/сцену")
    r[1].metric("95 % интервал по видео", f"{last['lo']:.2f}–{last['hi']:.2f}")
    r[2].metric("Время", f"{last['sec']:.0f} с")
    if last["lo"] < 0.5 < last["hi"] or last["auc"] < 0.6:
        st.warning("Интервал захватывает 0.5 или AUC низкий: форма рядов сама по себе слабо различает затяжку — для этих данных на неё нельзя полагаться отдельно.")
    _compare_tab(last, lt, profile)
    _attempts_and_saved(last, spec, run, lt, groups)


def _compare_tab(last: dict, lt: pd.DataFrame, profile: str) -> None:
    from sd import active as AC
    from sd import catalog as CAT

    st.markdown("#### Смесь с табличным классификатором")
    bundles = [c.id for c in CAT.cycle_bundles()]
    if not bundles:
        st.caption("Пакетов классификатора нет.")
        return
    cur = PR.load_named(profile).opts().get("cycle_bundle")
    b = st.selectbox("Табличный классификатор (его спецификация обучается заново по фолдам)", bundles, index=bundles.index(cur) if cur in bundles else 0, key="lts_bundle")
    if st.button("Сравнить и смешать", key="lts_blend"):
        try:
            spec = AC.spec_from_bundle(b)
            sp = {**spec, "cv": {**CL.DEFAULT_CV, **(spec.get("cv") or {}), "by_scene": bool(last["by_scene"]), "repeats": 2}}
            with st.spinner("Табличная модель по тем же фолдам…"):
                p_tab = CL.cv_predict(lt, sp)["__ensemble__"]
            st.session_state["lts_blend_res"] = dict(curve=TS.blend_curve(last["y"], p_tab, last["oof"], last["groups"]), auc_tab=_auc(last["y"], p_tab), bundle=b, run=last["run"])
        except Exception as e:      # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")
    br = st.session_state.get("lts_blend_res")
    if br and br["run"] == last["run"]:
        cv = br["curve"]
        best = cv.loc[cv.AUC.idxmax()]
        c = st.columns(3)
        c[0].metric("Только таблица", f"{br['auc_tab']:.3f}")
        c[1].metric("Только ряды", f"{last['auc']:.3f}")
        c[2].metric("Лучшая смесь", f"{best.AUC:.3f}", delta=f"{best.AUC - br['auc_tab']:+.3f}", help=f"вес рядов {best.w:g}")
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(x=cv.w, y=cv.AUC, mode="lines+markers", name="AUC смеси"))
        fig.add_traces([go.Scatter(x=cv.w, y=cv.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"),
                        go.Scatter(x=cv.w, y=cv.lo, fill="tonexty", line=dict(width=0), fillcolor="rgba(15,118,110,0.15)", showlegend=False, hoverinfo="skip")])
        fig.update_xaxes(title="вес моделей по рядам в смеси (0 — только таблица, 1 — только ряды)")
        fig.update_yaxes(title="AUC")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch", key="lts_blend_fig")
        st.caption("Прирост меньше ширины интервала — шум. Пока ряды не встроены в конвейер, смесь — оценка пользы; встраивать стоит, если прирост устойчив на отложенных видео.")


def _attempts_and_saved(last, spec, run, lt, groups) -> None:
    att = st.session_state.get("lts_attempts", [])
    if att:
        st.markdown("#### Попытки в этой сессии")
        st.dataframe(pd.DataFrame(att[::-1]), hide_index=True, width="stretch")
    if last and spec:
        st.markdown("#### Важность каналов и сохранение")
        c = st.columns([2, 2, 3], vertical_alignment="bottom")
        if c[0].button("Важность каналов", key="lts_imp", help="падение AUC при перемешивании канала среди тестовых циклов"):
            with st.spinner("Перемешиваем каналы…"):
                st.session_state["lts_imp_res"] = TS.channel_importance(last["X"], last["S"], last["y"], last["groups"], spec, last["auc"], repeats=1)
        imp = st.session_state.get("lts_imp_res")
        if imp is not None:
            st.dataframe(imp, hide_index=True, width="stretch")
        name = c[1].text_input("Имя модели", "ts_" + spec.model, key="lts_name")
        if c[2].button("Обучить на всех и сохранить в models/ts", key="lts_save"):
            try:
                dest = TS_DIR / UC_name(name)
                if dest.exists():
                    st.error("Модель с таким именем уже есть — выберите другое имя.")
                else:
                    m = TS.TSModel(spec).fit(last["X"], last["S"], last["y"])
                    m.save(dest, meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), trained_on=run, n=int(len(last["y"])), videos=sorted(set(map(str, lt.video))),
                                           auc_oof=round(float(last["auc"]), 4), by_scene=bool(last["by_scene"])))
                    st.success(f"сохранено: {dest}")
            except Exception as e:      # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")
    saved = _saved()
    if len(saved):
        st.markdown("#### Сохранённые модели")
        st.dataframe(saved, hide_index=True, width="stretch")


def UC_name(n: str) -> str:
    n = str(n).strip()
    if not n or not all(ch.isalnum() or ch in "-_." for ch in n) or n.startswith("."):
        raise ValueError("имя: буквы, цифры, - _ . (без точки в начале)")
    return n
