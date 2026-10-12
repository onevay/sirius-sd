"""Вкладка «Модели»: профили конфигураций (просмотр, правка, копия, удаление), реестр весов, подключение и проверка своих моделей, перенос пакетов, фото-модель."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sd import catalog as CAT
from sd import profiles as PR
from sd.ui import common as UC
from sd.ui import nav
from sd.ui import tabs_models as TM

SECTIONS = ["Профили", "Реестр", "Подключить модель", "Проверить модель", "Перенос", "Фото-модель"]


def _profiles() -> None:
    names = PR.list_profiles()
    c = CAT.summary()
    st.caption(f"Доступно: поза {c['pose']} · детекторы {c['detector']} · классификаторы цикла {c['cycle']} · фото {c['photo']} · VLM {c['vlm']}")
    if not names:
        st.info("Профилей нет: установите решатель («Выпуск») или положите YAML в configs/experiments/.")
        return
    rows = []
    for n in names:
        try:
            p = PR.load_named(n)
            rows.append(dict(профиль=n, отпечаток=p.fingerprint(), **p.describe(), описание=p.description[:120]))
        except Exception as e:      # noqa: BLE001
            rows.append(dict(профиль=n, pose=f"ошибка: {e}"))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    c = st.columns([2, 2, 2], vertical_alignment="bottom")
    name = c[0].selectbox("Профиль", names, key="mp_pick")
    if c[1].button("Открыть в «Конструкторе»", type="primary", help="схема конвейера, правка этапов с ценой правки, оценка и сохранение"):
        nav.go("designer", ds_base_pick=name)
    with c[2]:
        UC.confirm("Удалить профиль", "mp_del", f"Профиль «{name}» будет удалён (файл configs/experiments/{name}.yaml). Журнал оценок останется.",
                   lambda: PR.delete(name) and f"удалён {name}")


def render() -> None:
    UC.style()
    UC.header("Модели", "Профили конфигураций, реестр весов с лицензиями, подключение и проверка своих моделей, перенос пакетов.")
    part = UC.sections(SECTIONS, "models_section")
    if part == "Профили":
        _profiles()
    elif part == "Фото-модель":
        TM.tab_photo()
    else:
        TM.tab_models({"Реестр": "Реестр", "Подключить модель": "Подключить свою модель", "Проверить модель": "Проверить модель", "Перенос": "Перенос на другую машину"}[part])
