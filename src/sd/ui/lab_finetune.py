"""Лаборатория → «Дообучение»: переобучение пакета классификатора на новых данных с повышенным весом новых и трудных циклов, с честным сравнением «с весами / без»."""
from __future__ import annotations

import time

import streamlit as st

from sd import active as AC
from sd import bundle as B
from sd import catalog as CAT
from sd import classifier as CL
from sd.paths import CYCLE_MODELS, repo_path
from sd.ui import common as UC
from sd.ui import lab_common as LC


def render() -> None:
    st.markdown("Дообучение берёт пакет за основу (его спецификацию: признаки и членов), переобучает на **объединённых** данных и усиливает **новые** видео и **трудные** циклы весом. "
                "Оценка идёт на отложенных видео/сценах без весов — так видно, помогает ли это на самом деле. Основной пакет не изменяется: результат сохраняется новым пакетом.")
    run = LC.pick_run("Таблица циклов (прогон оценки)", "lf_run")
    bundles = [c.id for c in CAT.cycle_bundles()]
    if not run or not bundles:
        if not bundles:
            st.info("Нет пакетов классификатора.")
        return
    base = st.selectbox("Базовый пакет", bundles, key="lf_base", index=bundles.index("models/cycle/cycle_mvp") if "models/cycle/cycle_mvp" in bundles else 0)
    tab = LC.table_for(run, LC.gestures_stamp())
    lt = tab[tab.y.notna()].reset_index(drop=True)
    man = B.Bundle(repo_path(base)).manifest
    vids, how = B.training_videos(repo_path(base))
    c = st.columns(4)
    c[0].metric("Обучен на циклах", man.get("n", "—"))
    c[1].metric("Затяжек в обучении", man.get("n_pos", "—"))
    c[2].metric("Видео обучения", len(vids) if vids else "неизвестно")
    c[3].metric("AUC oof пакета", man.get("cv", {}).get("auc_ensemble", "—"))
    all_v = sorted(lt.video.unique())
    new_default = [v for v in all_v if vids and v not in vids] if how in ("manifest", "oof") else []
    new_v = st.multiselect("Новые видео (получают повышенный вес)", all_v, default=new_default[:40], key=f"lf_new_{base}",
                           help="по умолчанию — видео, которых не было в обучении пакета; если состав неизвестен, выберите сами")
    cc = st.columns(4, vertical_alignment="bottom")
    w_new = cc[0].slider("Вес новых циклов", 1.0, 6.0, 2.0, 0.5, key="lf_wn")
    use_hard = cc[1].toggle("Усилить трудные циклы", value=True, key="lf_hard", help="размеченные циклы, где out-of-fold оценка ошибается на ≥ 0.5")
    w_hard = cc[2].slider("Вес трудных", 1.0, 6.0, 2.0, 0.5, key="lf_wh", disabled=not use_hard)
    by_scene = cc[3].toggle("Фолды по сценам", value=True, key="lf_scene")
    spec = AC.spec_from_bundle(repo_path(base))
    problems = CL.validate(spec, lt)
    ok = UC.checklist([("error", p) for p in problems] or [("ok", "признаки пакета есть в таблице")], title="Проверка")
    if len(new_v) == 0 and not use_hard:
        st.info("Выберите новые видео или включите усиление трудных циклов — иначе веса равны единице и результат совпадёт с обычным обучением.")
    if st.button("Сравнить: с весами и без", type="primary", disabled=not ok, key="lf_go"):
        with st.spinner("Кросс-валидация с весами и без них…"):
            res = AC.finetune_cv(lt, spec, new_videos=new_v, w_new=w_new, w_hard=w_hard, use_hard=use_hard, by_scene=by_scene)
        st.session_state["lf_res"] = dict(res=res, run=run, base=base, new=new_v, w_new=w_new, w_hard=w_hard, use_hard=use_hard, by_scene=by_scene, spec=spec, n=len(lt))
    r = st.session_state.get("lf_res")
    if not r or r["run"] != run or r["base"] != base or r["n"] != len(lt):
        return
    res = r["res"]
    st.dataframe(res["table"], hide_index=True, width="stretch")
    a, b = res["table"].iloc[0], res["table"].iloc[1]
    d_all = b["AUC_все"] - a["AUC_все"]
    d_new = (b["AUC_новые"] - a["AUC_новые"]) if a["AUC_новые"] is not None and b["AUC_новые"] is not None else None
    m = st.columns(2)
    m[0].metric("Изменение AUC на всех", f"{d_all:+.3f}")
    m[1].metric("Изменение AUC на новых", f"{d_new:+.3f}" if d_new is not None else "—")
    if d_all < -0.01 or (d_new is not None and d_new < -0.01):
        st.warning("Веса ухудшили качество на отложенных видео: оставьте обычное обучение или уменьшите веса. Усиление трудных циклов особенно рискованно, если среди них есть ошибки разметки — проверьте «Активное обучение → Перепроверка меток».")
    elif d_all > 0.01 or (d_new is not None and d_new > 0.01):
        st.success("Веса помогли на отложенных видео. Прирост меньше ширины доверительного интервала (≈ ±0.05–0.1 на сотнях циклов) — считайте предварительным.")
    else:
        st.info("Разница в пределах шума: веса ничего не дали.")
    st.markdown("#### Сохранить новый пакет")
    sc = st.columns([3, 2, 2], vertical_alignment="bottom")
    name = sc[0].text_input("Имя нового пакета", "cycle_ft", key="lf_name")
    overwrite = sc[1].toggle("Перезаписать, если есть", value=False, key="lf_over")
    if sc[2].button("Обучить и сохранить", key="lf_save"):
        try:
            if not name.strip() or not all(ch.isalnum() or ch in "-_." for ch in name) or name.startswith("."):
                raise ValueError("имя: буквы, цифры, - _ . (без точки в начале)")
            dest = CYCLE_MODELS / name.strip()
            if dest.exists() and not overwrite:
                raise ValueError(f"пакет {name} уже есть — включите «Перезаписать» или выберите другое имя")
            sp = {**r["spec"], "name": name.strip(), "cv": {**CL.DEFAULT_CV, "by_scene": r["by_scene"]}}
            tw = res["rows"]
            tw.attrs.update(lt.attrs)
            params = dict(base=r["base"], new_videos=r["new"], w_new=r["w_new"], w_hard=r["w_hard"], use_hard=r["use_hard"], by_scene=r["by_scene"], at=time.strftime("%Y-%m-%d %H:%M:%S"), source_run=run)
            man2 = AC.finetune_save(tw, sp, dest, parent=r["base"], params=params)
            st.success(f"сохранено: {dest} · AUC oof {man2['cv']['auc_ensemble']}")
            st.caption("Пакет обучен на всех размеченных циклах таблицы — оценка на этих же клипах будет оптимистичной (утечка видна в чек-листах оценки и воротах выпуска). "
                       "Подключите его к профилю в «Конструкторе» и оцените на отложенных видео.")
        except Exception as e:      # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")
