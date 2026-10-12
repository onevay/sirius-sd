"""Dev Studio — интерфейс разработчика. Запуск:  sd.cmd ui   (или streamlit run src/sd/ui/app.py).

Вкладки идут в порядке работы над моделью (docs/DEV_STUDIO.md):
  Валидация      — проверить конфигурацию на размеченных папках, отчёт и PDF, анализ кэша (где и почему ошибки), сравнение прогонов, поток
  Задачи         — разметка событий и жестов, признаки, обучение классификатора (диагностика, пакет), результаты, очередь фоновых задач
  Данные         — каталог видео, проблемы данных, версии разметки с откатом, решения оператора → эталон
  Эксперименты   — журнал прогонов, варианты профиля, метки циклов из эталона
  Модели         — профили, реестр весов, свои модели, перенос
  Профилирование — скорость по кэшу, лестница вычислений, машина
  Выпуск         — ворота качества → решатель + PDF + манифест; установка решателя
Тяжёлые операции — только фоновыми задачами (sd.tasks): закрытие вкладки их не прерывает, вторая тяжёлая задача ждёт в очереди.
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import streamlit as st

from sd.ui import nav, p_data, p_experiments, p_models, p_profiling, p_release, p_tasks, p_validation

st.set_page_config(page_title="Dev Studio · детекция курения", layout="wide", page_icon="🧪")
nav.PAGES.update(
    validation=st.Page(p_validation.render, title="Валидация", url_path="validation", default=True),
    tasks=st.Page(p_tasks.render, title="Задачи", url_path="tasks"),
    data=st.Page(p_data.render, title="Данные", url_path="data"),
    experiments=st.Page(p_experiments.render, title="Эксперименты", url_path="experiments"),
    models=st.Page(p_models.render, title="Модели", url_path="models"),
    profiling=st.Page(p_profiling.render, title="Профилирование", url_path="profiling"),
    release=st.Page(p_release.render, title="Выпуск", url_path="release"),
)


def _sidebar() -> None:
    from sd import gt as GT
    from sd import labels_history as LH
    from sd import tasks as T
    from sd.ui import common as UC

    with st.sidebar:
        st.markdown("### 🧪 Dev Studio")
        st.text_input("Кто работает", key="who", placeholder="имя или инициалы", help="пишется в разметку, версии разметки и отчёты")

        @st.fragment(run_every=5)
        def _status() -> None:
            rows = T.list_tasks(limit=20)
            run = [r for r in rows if r["state"] == "идёт"]
            wait = [r for r in rows if r["state"] == "в очереди"]
            bad = [r for r in rows[:5] if r["state"] == "ошибка"]
            st.markdown("**Фоновые задачи**")
            if not rows:
                st.caption("задач не было")
            for r in run:
                p = r.get("progress")
                st.caption(f"⏳ {r['title']}" + (f" — {p['done']}/{p['total']}" if p else ""))
                if p:
                    st.progress(p["frac"])
            if wait:
                st.caption(f"🕓 в очереди: {len(wait)}")
            for r in bad:
                st.caption(f"⛔ {r['title']} — ошибка (журнал: «Задачи → Очередь задач»)")
            if rows and not run and not wait:
                st.caption("сейчас ничего не выполняется")

        _status()
        gt = GT.load()
        v = LH.versions("events")
        st.markdown("**Разметка**")
        st.caption(f"эталон: {len(gt)} строк, {gt.clip_id.nunique() if len(gt) else 0} клипов · версий {len(v)}" + (f" · последняя {v.iloc[0].ts[:16]}" if len(v) else ""))
        if not UC.who():
            st.caption("⚠️ укажите имя — без него правки разметки не подписаны")
        st.caption("Руководство: docs/DEV_STUDIO.md")


_sidebar()
st.navigation(list(nav.PAGES.values()), position="top").run()
