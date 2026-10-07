"""Веб-интерфейс. Запуск:  sd.cmd ui   (или streamlit run src/sd/ui/app.py).

Страницы идут в порядке рабочего процесса:
  Оценка    — выбрать папки и модели → получить F1, P, R, FP/час, интервал, бюджет ошибок до цели, кривую порога, ошибки
  Просмотр  — видео с эталоном, событиями, рамками людей; переход по ошибкам
  Разметка  — эталон событий на видео (начало, конец, человек, метка)
  Модели    — профили, реестр, подключение и проверка своих моделей
  Этапы     — разбор пайплайна по шагам (отладка, обучение)
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import streamlit as st

from sd.ui import nav, page_eval, page_label, page_models, page_stages, page_view

st.set_page_config(page_title="Детекция курения", layout="wide")
nav.PAGES.update(
    eval=st.Page(page_eval.render, title="Оценка", url_path="eval", default=True),
    view=st.Page(page_view.render, title="Просмотр", url_path="view"),
    label=st.Page(page_label.render, title="Разметка", url_path="label"),
    models=st.Page(page_models.render, title="Модели", url_path="models"),
    stages=st.Page(page_stages.render, title="Этапы", url_path="stages"),
)
st.navigation(list(nav.PAGES.values()), position="top").run()
