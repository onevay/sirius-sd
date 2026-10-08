"""Страница «Оценка»: выбрать папки и модели → получить Event F1 с интервалом, остальные метрики, кривую порога и разбор ошибок. Каждый прогон попадает в журнал."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sd import experiments as XP
from sd import gt as GT
from sd import library as LIB
from sd import profiles as PR
from sd import runner as RN
from sd.paths import video_id
from sd import catalog as CAT
from sd import solver as SV
from sd.ui import profile_editor, report_view

POLICIES = {"plateau": "центр плато F1 (рекомендуется)", "best": "вершина кривой", "fixed": "порог из профиля"}


@st.cache_data(show_spinner=False, ttl=30)
def _dirs(extra: tuple) -> pd.DataFrame:
    return LIB.discover_dirs(extra=extra)


def _dir_table(dirs: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in dirs.itertuples():
        ids = [video_id(v) for v in LIB.videos_in([r.path])]
        s = GT.clip_status(gt, ids)
        rows.append({"папка": r.name, "клипов": r.clips, "размечено": int(s.reviewed.sum()), "позитивов": int(s.positive.sum()), "метка по имени папки": {"smoking": "курение", "fake": "не курение"}.get(r.weak, "—")})
    return pd.DataFrame(rows)


def _progress():
    bar = st.progress(0.0, text="подготовка")

    def cb(i: int, n: int, msg: str) -> None:
        bar.progress(i / max(n, 1), text=f"{i}/{n} · {msg}")

    return bar, cb


def _data_block(gt: pd.DataFrame) -> dict:
    extra = tuple(st.session_state.get("eval_extra", ()))
    dirs = _dirs(extra)
    c1, c2 = st.columns([4, 2])
    with c2.popover("Добавить папку"):
        p = st.text_input("Путь к папке с видео", key="eval_extra_in")
        if st.button("Добавить") and p:
            st.session_state["eval_extra"] = [*extra, p]
            st.rerun()
    if dirs.empty:
        c1.info("Папок с видео нет: положите клипы в `data/`, `data_external/` или добавьте путь.")
        return {}
    chosen = c1.multiselect("Папки для оценки", dirs.name.tolist(), default=st.session_state.get("eval_dirs") or dirs.name.tolist(), key="eval_dirs", placeholder="Выберите папки",
                            help="оцениваются только клипы выбранных папок")
    tbl = _dir_table(dirs, gt)
    st.dataframe(tbl[tbl.папка.isin(chosen)] if chosen else tbl, hide_index=True)
    c = st.columns(4)
    mode = c[0].radio("Эталон", ["events", "clips"], format_func=lambda m: {"events": "ручная разметка событий", "clips": "по имени папки (грубо)"}[m], key="eval_mode")
    role = c[1].radio("Набор", ["validation", "hidden"], format_func=lambda m: {"validation": "валидация", "hidden": "скрытый (порог заморожен)"}[m], key="eval_role")
    policy = c[2].selectbox("Порог", list(POLICIES), format_func=POLICIES.get, key="eval_policy", disabled=role == "hidden")
    target = c[3].number_input("Целевая F1", 0.0, 1.0, float(st.session_state.get("eval_target", 0.80)), 0.01, key="eval_target")
    return dict(dirs=[dirs.set_index("name").loc[n, "path"] for n in chosen], mode=mode, role=role, policy="fixed" if role == "hidden" else policy, target=target)


def _classifier_block(first: PR.Profile | None, many: bool) -> tuple[str | None, bool]:
    """Классификатор цикла — основа решения и всегда обязателен: выбирается здесь, ДО остальных моделей (VLM, предмет и фото-модель — надстройка над ним). Возвращает (пакет | None = как в профиле, ошибки)."""
    bundles = CAT.cycle_bundles()
    if not bundles:
        st.error("В models/cycle/ нет классификаторов цикла: обучите (`sd train-bundle`, страница «Классификатор») или установите решатель (`sd solver install`).")
        return None, True
    ids = [b.id for b in bundles]
    cur = (first.opts()["cycle_bundle"] if first else None)
    options = ([None] if many else []) + ids + ([cur] if cur and cur not in ids else [])
    sel = st.selectbox("Классификатор цикла", options, index=options.index(cur) if cur in options else 0, key="eval_classifier_" + (first.name if first else "") + f"_{many}",
                       format_func=lambda x: "как в профиле" if x is None else Path(x).name, help="обязателен: он оценивает каждый цикл; остальное подключается к нему, не меняя его признаков")
    if sel:
        try:
            d = SV.describe_bundle(sel)
            st.caption(f"{d['n_features']} признаков {d['groups']} · члены {d['members']} · AUC out-of-fold {d['auc_oof']} · обучен на {d['n']} циклах ({d['n_pos']} затяжек, {d['n_groups']} видео)"
                       + (f" · нужны детекторы: {', '.join(d['requires']['objects'])}" if d['requires']['objects'] else "") + (" · нужна фото-модель" if d['requires']['photo'] else "")
                       + (" · нужен VLM" if d['requires']['vlm'] else ""))
        except Exception as e:
            st.error(f"пакет {sel} не читается: {e}")
    return sel, False


def _profile_block() -> tuple[list[PR.Profile], str | None]:
    names = PR.list_profiles()
    c1, c2 = st.columns([3, 2])
    kind = c2.radio("Что запускать", ["one", "many"], format_func=lambda k: {"one": "один профиль", "many": "сравнить профили"}[k], horizontal=True, key="eval_kind")
    if kind == "many":
        sel = c1.multiselect("Профили", names, default=names[:2], key="eval_many")
        if len(names) < 2:
            st.info("Для сравнения нужно минимум два сохранённых профиля (страница «Модели» или кнопка «Сохранить профиль» ниже).")
        profs = [PR.load(n) for n in sel]
        cls, _ = _classifier_block(profs[0] if profs else None, many=True)
        return profs, cls
    options = ["default", *[n for n in names if n != "default"]]
    cur = c1.selectbox("Профиль", options, key="eval_profile")
    base = PR.default_profile() if cur == "default" and "default" not in names else PR.load(cur)
    cls, _ = _classifier_block(base, many=False)
    with st.expander("Модели и параметры", expanded=False):
        try:
            prof = profile_editor.editor(base, classifier=cls)
        except Exception as e:
            st.error(f"профиль «{cur}» не открылся: {e}")
            return [], cls
        c = st.columns([2, 1])
        new = c[0].text_input("Имя для сохранения", base.name, key=f"save_name_{base.name}")
        if c[1].button("Сохранить профиль"):
            try:
                PR.save(PR.Profile(new, base.description, base.base, prof.config, prof.options))
                st.success(f"сохранён: configs/experiments/{new}.yaml")
            except Exception as e:
                st.error(str(e))
    return [prof], cls


def render() -> None:
    gt = GT.load()
    st.subheader("1. Данные")
    sel = _data_block(gt)
    st.subheader("2. Модели")
    profiles, cls = _profile_block()
    problems = [(p.name, i) for p in profiles for i in SV.check(RN.with_classifier(p, cls))]
    for name, i in problems:
        (st.error if i.level == "error" else st.warning)(f"{name}: {i.text}")
    blocked = any(i.level == "error" for _, i in problems)
    st.subheader("3. Запуск")
    c = st.columns([1, 1, 1, 2])
    cache = c[0].checkbox("Брать из кэша", value=True, help="прогон модели повторяется, только если изменились модели, параметры или файл видео")
    cap = c[1].number_input("Длина клипа, с (0 — целиком)", 0, 3600, 0, 10)
    roi = c[2].text_input("ROI (roi.json)", "", placeholder="необязательно")
    ready = bool(sel.get("dirs")) and bool(profiles) and not blocked
    if c[3].button("Оценить", type="primary", disabled=not ready):
        done = []
        for p in profiles:
            bar, cb = _progress()
            try:
                out = RN.evaluate_dirs(sel["dirs"], p, classifier=cls, mode=sel["mode"], role=sel["role"], policy=sel["policy"], target_f1=sel["target"], use_cache=cache, max_sec=cap or None,
                                       roi_path=roi or None, progress=cb, gt_df=gt)
                done.append(out.run_id)
            except Exception as e:
                st.error(f"{p.name}: {e}")
            finally:
                bar.empty()
        if done:
            st.session_state["eval_last"] = done
    if blocked:
        st.caption("Запуск заблокирован: исправьте ошибки профиля выше.")
    elif not ready:
        st.caption("Выберите папки и профиль.")
    last = st.session_state.get("eval_last")
    if last:
        st.divider()
        if len(last) > 1:
            _compare(last)
        rid = last[0] if len(last) == 1 else st.selectbox("Показать прогон", last)
        rep, meta, ev, _ = XP.load_run(rid)
        st.subheader(f"Результат · {meta['name']}")
        report_view.show(rep, meta, rid, ev)
    _history()


def _compare(ids: list[str]) -> None:
    rows = []
    for i in ids:
        rep, meta, _, _ = XP.load_run(i)
        ci = rep.ci.get("f1") or (None, None)
        rows.append(dict(профиль=meta["name"], F1=rep.metrics["f1"], от=ci[0], до=ci[1], precision=rep.metrics["precision"], recall=rep.metrics["recall"], FP_в_час=rep.metrics["fp_per_hour"],
                         ошибок=rep.budget["errors"], допустимо=rep.budget["allowed"], цель=rep.budget["reached"], **{k: v for k, v in meta["describe"].items() if k in ("pose", "cycle_model", "vlm")}))
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    st.subheader("Сравнение профилей на одних и тех же данных")
    st.dataframe(df, hide_index=True, column_config={"F1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f")})


def _history() -> None:
    runs = XP.list_runs()
    with st.expander(f"Журнал экспериментов ({len(runs)})"):
        if runs.empty:
            st.caption("Пока пусто.")
            return
        show = runs[["created", "name", "mode", "role", "clips", "f1", "ci_lo", "ci_hi", "precision", "recall", "fp_per_hour", "threshold", "reached", "dirs", "pose", "cycle_model", "vlm", "id"]]
        sel = st.dataframe(show, hide_index=True, on_select="rerun", selection_mode="single-row", key="hist",
                           column_config={"f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f")})
        rows = sel["selection"]["rows"] if sel and sel.get("selection") else []
        if rows:
            rid = show.iloc[rows[0]]["id"]
            c = st.columns(3)
            if c[0].button("Показать"):
                st.session_state["eval_last"] = [rid]
                st.rerun()
            if c[1].button("Открыть в просмотре"):
                from sd.ui import nav

                nav.go("view", view_src="Эксперимент", view_run=rid)
            if c[2].button("Удалить прогон"):
                XP.delete_run(rid)
                st.rerun()
