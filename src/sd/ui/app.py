"""Веб-интерфейс. Запуск:  sd.cmd ui   (или streamlit run src/sd/ui/app.py).

Страницы идут в порядке рабочего процесса:
  Оценка    — выбрать папки и модели → получить F1, P, R, FP/час, интервал, бюджет ошибок до цели, кривую порога, ошибки
  Эксперименты — журнал, парное сравнение, серия вариантов профиля, метки циклов из эталона
  Просмотр  — видео с эталоном, событиями, рамками людей; переход по ошибкам
  Разметка  — эталон событий на видео (начало, конец, человек, метка)
  Анализ    — метки жестов (правка), оценки циклов и порог цикла, кэш и перезапуск прогонов
  Классификатор — признаки, члены ансамбля, регуляризация, диагностика переобучения, обучение пакета
  Мониторинг — камеры-папки, воркер, перенос решений оператора в эталон (дашборд оператора — `sd app`)
  Модели    — профили, реестр, подключение и проверка своих моделей
  Задачи    — запуск длинных операций (enrich, оценки, лестница, загрузка весов) из браузера, с журналом
  Этапы     — разбор пайплайна по шагам (отладка, обучение)
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import streamlit as st

from sd.ui import (nav, page_analysis, page_classifier, page_eval, page_experiments, page_label, page_models, page_monitor, page_stages, page_stream, page_tasks, page_view)

st.set_page_config(page_title="Детекция курения", layout="wide")
nav.PAGES.update(
    eval=st.Page(page_eval.render, title="Оценка", url_path="eval", default=True),
    experiments=st.Page(page_experiments.render, title="Эксперименты", url_path="experiments"),
    view=st.Page(page_view.render, title="Просмотр", url_path="view"),
    label=st.Page(page_label.render, title="Разметка", url_path="label"),
    analysis=st.Page(page_analysis.render, title="Анализ", url_path="analysis"),
    classifier=st.Page(page_classifier.render, title="Классификатор", url_path="classifier"),
    monitor=st.Page(page_monitor.render, title="Мониторинг", url_path="monitor"),
    stream=st.Page(page_stream.render, title="Поток", url_path="stream"),
    models=st.Page(page_models.render, title="Модели", url_path="models"),
    tasks=st.Page(page_tasks.render, title="Задачи", url_path="tasks"),
    stages=st.Page(page_stages.render, title="Этапы", url_path="stages"),
)
st.navigation(list(nav.PAGES.values()), position="top").run()
