"""Dev Studio — интерфейс разработчика. Запуск:  sd.cmd ui   (или streamlit run src/sd/ui/app.py). Описание — docs/DEV_STUDIO.md.

Вкладки в порядке работы над моделью:
  Валидация      — проверить конфигурацию на размеченных папках: запуск с проверкой до старта, отчёт и PDF, разбор слабых мест по кэшу, поток
  Аналитика      — сравнение подходов по журналу: метрики и скорость, клип × подход, ошибки и порог, влияние параметров, классификаторы
  Конструктор    — схема конвейера и правка любых его параметров с ценой правки; оценка без сохранения, сохранение профиля, серия вариантов
  Задачи         — разметка событий и жестов, признаки, обучение классификатора (диагностика, пакет), результаты, очередь фоновых задач
  Данные         — каталог видео, проблемы данных, версии разметки с откатом, метки циклов, решения оператора → эталон
  Модели         — профили, реестр весов, свои модели, перенос
  Профилирование — скорость по кэшу, лестница вычислений, машина
  Выпуск         — ворота качества → решатель + PDF + манифест; установка решателя
Тяжёлые операции — только фоновыми задачами (sd.tasks): закрытие вкладки их не прерывает, вторая тяжёлая задача ждёт в очереди.
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import streamlit as st

from sd.ui import nav, p_analytics, p_data, p_designer, p_models, p_profiling, p_release, p_tasks, p_validation

st.set_page_config(page_title="Dev Studio · детекция курения", layout="wide", page_icon="🧪")
nav.PAGES.update(
    validation=st.Page(p_validation.render, title="Валидация", url_path="validation", default=True, icon=":material/fact_check:"),
    analytics=st.Page(p_analytics.render, title="Аналитика", url_path="analytics", icon=":material/insights:"),
    designer=st.Page(p_designer.render, title="Конструктор", url_path="designer", icon=":material/account_tree:"),
    tasks=st.Page(p_tasks.render, title="Задачи", url_path="tasks", icon=":material/construction:"),
    data=st.Page(p_data.render, title="Данные", url_path="data", icon=":material/database:"),
    models=st.Page(p_models.render, title="Модели", url_path="models", icon=":material/deployed_code:"),
    profiling=st.Page(p_profiling.render, title="Профилирование", url_path="profiling", icon=":material/speed:"),
    release=st.Page(p_release.render, title="Выпуск", url_path="release", icon=":material/rocket_launch:"),
)


def _sidebar() -> None:
    from sd import gt as GT
    from sd import labels_history as LH
    from sd import tasks as T
    from sd.ui import common as UC

    with st.sidebar:
        st.markdown("### 🧪 Dev Studio")
        st.caption("Детекция курения · интерфейс разработчика")
        st.text_input("Кто работает", key="who", placeholder="имя или инициалы", help="пишется в разметку, версии разметки и отчёты")
        if not UC.who():
            st.caption("⚠️ укажите имя — без него правки разметки не подписаны")

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
        st.markdown("**Состояние**")
        runs = UC.runs_df()
        if len(runs):
            r = runs.iloc[0]
            st.caption(f"последний прогон: {r['name']} · F1 {r['f1']:.3f} · {str(r['created'])[5:16]}")
        gt = GT.load()
        v = LH.versions("events")
        st.caption(f"эталон: {len(gt)} строк, {gt.clip_id.nunique() if len(gt) else 0} клипов · версий {len(v)}" + (f" · последняя {v.iloc[0].ts[5:16]}" if len(v) else ""))
        st.caption("Руководство: docs/DEV_STUDIO.md")


_sidebar()
st.navigation(list(nav.PAGES.values()), position="top").run()
