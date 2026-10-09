"""Страница «Задачи»: запуск длинных операций проекта из браузера (проверка окружения, загрузка весов, признаки циклов, оценки, лестница вычислений) с журналом и остановкой.

Каждая задача — отдельный процесс `python -m sd …` (см. `sd.tasks`): закрытие вкладки или перезапуск интерфейса её не прерывает; результат и вывод хранятся в outputs/tasks/.
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import time

import pandas as pd
import streamlit as st

from sd import tasks as T


def _form(t: T.TaskType) -> dict:
    vals: dict = {}
    for f in t.fields:
        key = f"task_{t.id}_{f.key}"
        if f.kind == "bool":
            vals[f.key] = st.checkbox(f.label, bool(f.default), key=key, help=f.help or None)
        elif f.kind == "int":
            vals[f.key] = int(st.number_input(f.label, 0, 100000, int(f.default), key=key, help=f.help or None))
        elif f.kind == "float":
            vals[f.key] = float(st.number_input(f.label, 0.0, 1000.0, float(f.default), key=key, help=f.help or None))
        elif f.kind == "choice":
            vals[f.key] = st.selectbox(f.label, list(f.choices), index=list(f.choices).index(f.default) if f.default in f.choices else 0, key=key, help=f.help or None)
        elif f.kind == "lines":
            vals[f.key] = st.text_area(f.label, str(f.default), key=key, help=f.help or None, height=80)
        else:
            vals[f.key] = st.text_input(f.label, str(f.default), key=key, help=f.help or None)
    return vals


@st.fragment(run_every=3)
def _monitor() -> None:
    rows = T.list_tasks()
    if not rows:
        st.info("Задач ещё не было.")
        return
    df = pd.DataFrame([dict(id=r["id"], задача=r["title"], состояние=r["state"], секунд=int(r["seconds"]), код=r.get("returncode")) for r in rows])
    sel = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="single-row", key="task_tbl")
    idx = sel["selection"]["rows"] if sel else []
    pick = rows[idx[0]] if idx else rows[0]
    st.markdown(f"**{pick['title']}** · {pick['state']} · `sd {' '.join(pick['args'])}`")
    st.code(T.tail(pick["id"], 60) or "(пока пусто)", language=None)
    if pick["state"] == "идёт" and st.button("Остановить эту задачу", key=f"stop_{pick['id']}"):
        T.stop(pick["id"])
        st.rerun()


def render() -> None:
    st.title("Задачи")
    st.caption("Всё, что раньше делалось командой `sd …`, запускается здесь: на новой машине достаточно Docker и браузера. Результаты — на страницах «Оценка», «Анализ», «Классификатор».")
    ids = list(T.CATALOG)
    tid = st.selectbox("Задача", ids, format_func=lambda i: T.CATALOG[i].title, key="task_pick")
    t = T.CATALOG[tid]
    st.write(t.description)
    if t.needs_net:
        st.warning("Нужна сеть (загрузка весов). Данные площадки при этом никуда не отправляются.")
    vals = _form(t)
    st.code("sd " + " ".join(T.command(tid, vals)), language="bash")
    running = [r for r in T.list_tasks(limit=10) if r["state"] == "идёт"]
    if running:
        st.info(f"сейчас выполняется: {', '.join(r['title'] for r in running)}. Тяжёлые задачи лучше запускать по одной (память и процессор общие).")
    if st.button("Запустить", type="primary"):
        st.session_state["task_last"] = T.start(tid, vals)
        time.sleep(0.5)
    st.divider()
    _monitor()
