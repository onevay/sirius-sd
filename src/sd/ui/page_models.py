"""Страница «Модели»: реестр и профили экспериментов, подключение своих моделей, проверка, перенос; фото-модель."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sd import catalog as CAT
from sd import profiles as PR
from sd.ui import tabs_models as TM


def _profiles() -> None:
    names = PR.list_profiles()
    c = CAT.summary()
    st.caption(f"Доступно: поза {c['pose']} · детекторы {c['detector']} · классификаторы цикла {c['cycle']} · фото {c['photo']} · VLM {c['vlm']}")
    if not names:
        st.info("Профилей пока нет: соберите модели на странице «Оценка» и сохраните профиль.")
        return
    rows = []
    for n in names:
        try:
            rows.append(dict(профиль=n, **PR.load(n).describe()))
        except Exception as e:
            rows.append(dict(профиль=n, pose=f"ошибка: {e}"))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    rm = st.selectbox("Удалить профиль", ["—"] + names)
    if rm != "—" and st.button("Удалить"):
        PR.delete(rm)
        st.rerun()


def render() -> None:
    part = st.radio("Раздел", ["Профили", "Реестр", "Подключить модель", "Проверить модель", "Перенос", "Фото-модель"], horizontal=True, label_visibility="collapsed")
    if part == "Профили":
        _profiles()
    elif part == "Фото-модель":
        TM.tab_photo()
    else:
        TM.tab_models({"Реестр": "Реестр", "Подключить модель": "Подключить свою модель", "Проверить модель": "Проверить модель", "Перенос": "Перенос на другую машину"}[part])
