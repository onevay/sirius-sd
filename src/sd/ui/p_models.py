"""Вкладка «Модели»: профили конфигураций (просмотр, правка, копия, удаление), реестр весов, подключение и проверка своих моделей, перенос пакетов, фото-модель."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sd import catalog as CAT
from sd import profiles as PR
from sd.ui import common as UC
from sd.ui import profile_editor
from sd.ui import tabs_models as TM

SECTIONS = ["Профили", "Реестр", "Подключить модель", "Проверить модель", "Перенос", "Фото-модель"]


def _profiles() -> None:
    names = PR.list_profiles()
    c = CAT.summary()
    st.caption(f"Доступно: поза {c['pose']} · детекторы {c['detector']} · классификаторы цикла {c['cycle']} · фото {c['photo']} · VLM {c['vlm']}")
    if not names:
        st.info("Профилей нет: установите решатель («Выпуск») или скопируйте YAML в configs/experiments/.")
        return
    rows = []
    for n in names:
        try:
            p = PR.load_named(n)
            rows.append(dict(профиль=n, отпечаток=p.fingerprint(), **p.describe()))
        except Exception as e:      # noqa: BLE001
            rows.append(dict(профиль=n, pose=f"ошибка: {e}"))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown("##### Правка профиля")
    name = st.selectbox("Профиль", names, key="mp_pick")
    base = PR.load_named(name)
    edited = profile_editor.editor(base, key="mp_pe")
    from sd.ui.p_validation import diff_sets

    sets = diff_sets(base, edited)
    if sets:
        st.code("\n".join(sets), language=None)
    c = st.columns([2, 2, 2])
    new = c[0].text_input("Сохранить как", name, key="mp_new")
    try:
        PR._check_name(new)
        valid = "." not in new
    except ValueError:
        valid = False
    target = PR.Profile(new.strip(), base.description, base.base, edited.config, edited.options)

    def _save():
        PR.save(target)
        return f"сохранён configs/experiments/{new.strip()}.yaml — отпечаток {target.fingerprint()} (прежние оценки относятся к старому отпечатку)"

    with c[1]:
        if new.strip() in names:
            UC.confirm("Сохранить (перезаписать)", "mp_save", f"Профиль «{new.strip()}» будет перезаписан: его прежние оценки в журнале перестанут соответствовать файлу "
                                                             "(выпуск проверяет совпадение отпечатка).", _save, disabled=not valid or not sets)
        elif st.button("Сохранить новый профиль", disabled=not valid, key="mp_save_new"):
            st.success(_save())
    with c[2]:
        UC.confirm("Удалить профиль", "mp_del", f"Профиль «{name}» будет удалён (файл configs/experiments/{name}.yaml). Журнал оценок останется.",
                   lambda: PR.delete(name) and f"удалён {name}")
    if not valid:
        st.caption("Имя профиля: буквы, цифры, - и _ (без точек).")


def render() -> None:
    st.header("Модели")
    if "models_section" not in st.session_state:
        st.session_state["models_section"] = SECTIONS[0]
    part = st.radio("Раздел", SECTIONS, horizontal=True, label_visibility="collapsed", key="models_section")
    if part == "Профили":
        _profiles()
    elif part == "Фото-модель":
        TM.tab_photo()
    else:
        TM.tab_models({"Реестр": "Реестр", "Подключить модель": "Подключить свою модель", "Проверить модель": "Проверить модель", "Перенос": "Перенос на другую машину"}[part])
