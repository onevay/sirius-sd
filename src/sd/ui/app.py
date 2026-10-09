"""Веб-интерфейс. Запуск:  sd.cmd ui   (или streamlit run src/sd/ui/app.py).

Страницы идут в порядке рабочего процесса:
  Оценка    — выбрать папки и модели → получить F1, P, R, FP/час, интервал, бюджет ошибок до цели, кривую порога, ошибки
  Журнал    — все прогоны, парное сравнение, серия вариантов профиля, метки циклов из эталона
  Просмотр  — видео с эталоном, событиями, рамками людей; переход по ошибкам
  Разметка  — эталон событий на видео (начало, конец, человек, метка)
  Анализ    — метки жестов (правка), оценки циклов и порог цикла, кэш и перезапуск прогонов
  Классификатор — признаки, члены ансамбля, регуляризация, диагностика переобучения, обучение пакета
  Реальное время — имитация потока по видео: тревоги, F1, задержка, пропускная способность (rt_factor); дашборд оператора — `sd app`
  Модели    — профили, реестр, подключение и проверка своих моделей
  Задачи    — запуск длинных операций (enrich, оценки, лестница, загрузка весов) из браузера, с журналом
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import streamlit as st

from sd.ui import (nav, page_analysis, page_classifier, page_eval, page_experiments, page_label, page_models, page_stream, page_tasks, page_view)

st.set_page_config(page_title="Детекция курения", layout="wide")
nav.PAGES.update(
    eval=st.Page(page_eval.render, title="Оценка", url_path="eval", default=True),
    experiments=st.Page(page_experiments.render, title="Журнал", url_path="experiments"),
    view=st.Page(page_view.render, title="Просмотр", url_path="view"),
    label=st.Page(page_label.render, title="Разметка", url_path="label"),
    analysis=st.Page(page_analysis.render, title="Анализ", url_path="analysis"),
    classifier=st.Page(page_classifier.render, title="Классификатор", url_path="classifier"),
    stream=st.Page(page_stream.render, title="Реальное время", url_path="stream"),
    models=st.Page(page_models.render, title="Модели", url_path="models"),
    tasks=st.Page(page_tasks.render, title="Задачи", url_path="tasks"),
)
st.navigation(list(nav.PAGES.values()), position="top").run()
