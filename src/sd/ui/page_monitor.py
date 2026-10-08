"""Страница «Мониторинг» (для разработчика): как кормить приложение оператора — камеры-папки, воркер, перенос решений оператора в эталон. Сам дашборд оператора — `sd app`."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sd.paths import STREAMS
from sd.realtime import feedback as FB
from sd.realtime import sources as S
from sd.realtime.demo import seed_demo
from sd.realtime.store import AlertStore


def render() -> None:
    store = AlertStore()
    st.subheader("Камеры (папки-имитация)")
    st.caption(f"Каталог: `{STREAMS}` (переменная SD_STREAMS). Имя папки: `<район>-<индекс>-<время начала>`, например `gatchina-03-20261009T143000`.")
    streams = S.scan_streams(STREAMS)
    if streams:
        st.dataframe(pd.DataFrame([dict(папка=s.path.name, район=s.district, камера=s.index, начало=s.start.isoformat(" "), фрагментов=len(S.list_chunks(s.path)), имя_разобрано="да" if s.parsed else "нет")
                                   for s in streams]), hide_index=True)
        if any(not s.parsed for s in streams):
            st.warning("Есть папки с именем не по шаблону: район = имя папки, время = дата файла.")
    else:
        st.info("Камер нет. Положите папки с видео в этот каталог.")
    st.code("sd.cmd monitor --watch --speed 1      # следить за папками, как в реальном времени\nsd.cmd monitor --profile final           # разово обработать всё, что есть\nsd.cmd app                               # приложение оператора", language="bash")
    cams = store.cameras()
    if len(cams):
        st.dataframe(cams[["camera_id", "status", "processed_sec", "alerts", "pending", "last_seen"]], hide_index=True)
    st.subheader("Решения оператора → эталон")
    s = store.stats()
    st.write(f"Подтверждено {s['confirmed']}, ложных {s['false']}, ждут решения {s['new']}.")
    c = st.columns(3)
    if c[0].button("Перенести решения в эталон"):
        st.success(str(FB.export_reviewed(store)))
    if c[1].button("Создать демо-тревоги"):
        st.success(f"создано: {seed_demo(store)}")
    if c[2].button("Удалить демо-тревоги"):
        st.success(f"удалено: {store.delete_demo()}")
