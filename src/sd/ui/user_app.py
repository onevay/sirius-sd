"""Интерфейс оператора (MVP-каркас). Запуск:  sd app   (http://localhost:8502).

Дашборд мониторинга: что требует решения человека сейчас, что решено, как работают камеры. Данные берутся из хранилища (`outputs/monitor/monitor.db`), которое наполняет воркер
`sd monitor` — по папкам-камерам <район>-<индекс>-<время начала>. Основной сценарий: пришла тревога → оператор смотрит миниатюру/клип → «Подтвердить» или «Ложная».
Решения возвращаются в эталон и обучение (`sd feedback-export`). Инструменты разработчика — отдельное приложение `sd ui`.
"""
from __future__ import annotations

import sd._env  # noqa: F401,E402

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from sd.paths import from_portable
from sd.realtime.store import DB_PATH, AlertStore

st.set_page_config(page_title="Мониторинг", layout="wide")

STATUS_RU = {"new": "ждёт решения", "confirmed": "подтверждена", "false": "ложная", "unsure": "не уверен"}
SECTIONS = ["Тревоги", "Камеры", "Журнал"]


@st.cache_resource
def _store(path: str) -> AlertStore:
    return AlertStore(path)


def store() -> AlertStore:
    return _store(os.environ.get("SD_MONITOR_DB") or str(DB_PATH))


def fmt_t(iso: str | None) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m %H:%M:%S")
    except (TypeError, ValueError):
        return "—"


def decide(alert_id: int, status: str) -> None:
    store().review(alert_id, status, st.session_state.get("operator", ""), st.session_state.get(f"note_{alert_id}", ""))


def open_alert(alert_id: int | None) -> None:
    st.session_state["open_alert"] = alert_id


def badge(a: dict) -> str:
    color = {"new": "red", "confirmed": "green", "false": "gray", "unsure": "orange"}[a["status"]]
    s = f":{color}-badge[{STATUS_RU[a['status']]}]"
    return s + (" :orange-badge[ДЕМО]" if a.get("demo") else "")


def thumb(a: dict, width: int | None = None) -> None:
    p = from_portable(a.get("thumb"))
    if p and Path(p).exists():
        st.image(str(p), width=width or "stretch")
    else:
        st.caption("нет кадра")


def card(a: dict) -> None:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.3, 3, 1.3])
        with c1:
            thumb(a)
        with c2:
            st.markdown(f"**{a['district'].capitalize()} · камера {a['cam_index']}**  {badge(a)}")
            st.caption(f"{fmt_t(a['t_abs'])} · {a['explain']} · длительность {max(a['end_sec'] - a['start_sec'], 0):.0f} с")
            st.progress(min(max(float(a["confidence"]), 0.0), 1.0), text=f"уверенность {a['confidence']:.0%}")
        with c3:
            pending = a["status"] == "new"
            # кнопки живут внутри автообновляемого фрагмента: полная перерисовка нужна, чтобы обновились счётчики сверху и открылась карточка
            if st.button("Подтвердить", key=f"ok{a['id']}", type="primary" if pending else "secondary", width="stretch"):
                decide(a["id"], "confirmed")
                st.rerun()
            if st.button("Ложная", key=f"no{a['id']}", width="stretch"):
                decide(a["id"], "false")
                st.rerun()
            if st.button("Открыть", key=f"op{a['id']}", width="stretch"):
                open_alert(a["id"])
                st.rerun()


def detail(alert_id: int) -> None:
    a = store().get(alert_id)
    if not a:
        st.warning("Тревога не найдена.")
        st.button("К списку", on_click=open_alert, args=(None,))
        return
    st.button("← К списку", on_click=open_alert, args=(None,))
    st.subheader(f"Тревога #{a['id']} · {a['district'].capitalize()}, камера {a['cam_index']}")
    st.markdown(badge(a))
    c1, c2 = st.columns([3, 2])
    with c1:
        clip = from_portable(a.get("clip"))
        if clip and Path(clip).exists():
            st.video(str(clip))
        else:
            thumb(a)
            st.caption("клип недоступен")
    with c2:
        st.metric("Уверенность", f"{a['confidence']:.0%}")
        st.write(f"**Когда:** {fmt_t(a['t_abs'])}  \n**Что сработало:** {a['explain']}  \n**Правило:** {a['rule']}  \n**Человек (ID трека):** {a['tid']}")
        st.text_input("Заметка", key=f"note_{a['id']}", value=a.get("note") or "")
        b = st.columns(3)
        b[0].button("Подтвердить", type="primary", key=f"dok{a['id']}", on_click=decide, args=(a["id"], "confirmed"), width="stretch")
        b[1].button("Ложная", key=f"dno{a['id']}", on_click=decide, args=(a["id"], "false"), width="stretch")
        b[2].button("Не уверен", key=f"dun{a['id']}", on_click=decide, args=(a["id"], "unsure"), width="stretch")
        if a.get("reviewed_at"):
            st.caption(f"решение: {STATUS_RU[a['status']]} · {a.get('reviewer') or 'оператор'} · {fmt_t(a['reviewed_at'])}")
    near = store().list_alerts(cameras=[a["camera_id"]], limit=8)
    near = near[near["id"] != a["id"]]
    if len(near):
        st.markdown("**Другие тревоги этой камеры**")
        near = near.assign(status=near["status"].map(STATUS_RU))
        st.dataframe(near[["id", "t_abs", "confidence", "status", "explain"]].rename(columns={"t_abs": "время", "confidence": "уверенность", "status": "статус", "explain": "что сработало"}),
                     hide_index=True)
    with st.expander("Для разбора разработчиком"):
        st.code(f"фрагмент: {a.get('chunk')}\nначало в фрагменте: {max(a['start_sec'] - (a.get('chunk_offset') or 0), 0):.1f} с\nключ события: {a['key']}\nпрофиль (отпечаток): {a.get('fingerprint')}")


def filters() -> dict:
    sb = st.sidebar
    sb.text_input("Оператор", key="operator", placeholder="имя или инициалы")
    cams = store().cameras()
    sb.subheader("Фильтры")
    status = sb.multiselect("Статус", list(STATUS_RU), default=["new"], format_func=STATUS_RU.get, key="f_status", placeholder="Любой")
    districts = sb.multiselect("Район", sorted(cams["district"].unique()) if len(cams) else [], key="f_district", placeholder="Все")
    pool = cams[cams["district"].isin(districts)] if districts and len(cams) else cams
    cameras = sb.multiselect("Камера", sorted(pool["camera_id"]) if len(pool) else [], key="f_camera", placeholder="Все")
    period = sb.radio("Период", ["Сегодня", "24 часа", "Всё время"], key="f_period", horizontal=True)
    minc = sb.slider("Уверенность не ниже", 0.0, 1.0, 0.0, 0.05, key="f_conf")
    refresh = sb.toggle("Автообновление", value=True, key="f_refresh", help="раз в 5 секунд, пока не открыта карточка")
    since = {"Сегодня": datetime.now().strftime("%Y-%m-%d"), "24 часа": (datetime.now() - timedelta(hours=24)).isoformat(timespec="seconds"), "Всё время": None}[period]
    return dict(status=status or None, districts=districts or None, cameras=cameras or None, since=since, min_conf=minc or None), refresh


def kpis() -> None:
    s = store().stats()
    c = st.columns(5)
    c[0].metric("Ждут решения", s["new"], help="тревоги без решения оператора")
    c[1].metric("Подтверждено сегодня", s["today"].get("confirmed", 0))
    c[2].metric("Ложных сегодня", s["today"].get("false", 0))
    c[3].metric("Точность тревог", "—" if s["precision"] is None else f"{s['precision']:.0%}", help="подтверждено / (подтверждено + ложных) за всё время")
    c[4].metric("Камер", s["cameras"])


def feed(flt: dict) -> None:
    s = store()
    seen = st.session_state.get("seen_max_id")
    mx = s.max_id()
    if seen is not None and mx > seen:
        st.toast(f"Новых тревог: {mx - seen}")
    st.session_state["seen_max_id"] = mx
    df = s.list_alerts(limit=st.session_state.get("limit", 20), **flt)
    if df.empty:
        st.info("Тревог по выбранным фильтрам нет." if s.stats()["total"] else "Тревог пока нет. Запустите воркер: `sd monitor --streams <папка с камерами> --watch` (или `sd monitor-demo` для демо-данных).")
        return
    for a in df.to_dict("records"):
        card(a)
    if len(df) >= st.session_state.get("limit", 20):
        st.button("Показать ещё", on_click=lambda: st.session_state.update(limit=st.session_state.get("limit", 20) + 20))


def cameras_view() -> None:
    cams = store().cameras()
    if cams.empty:
        st.info("Камер пока нет.")
        return
    now = datetime.now()

    def state(r) -> str:
        try:
            age = (now - datetime.fromisoformat(r["last_seen"])).total_seconds()
        except (TypeError, ValueError):
            return "нет данных"
        return "обработка" if r["status"] == "processing" and age < 120 else ("ожидание" if age < 3600 else "нет данных")

    cams = cams.assign(состояние=cams.apply(state, axis=1), обработано_мин=(cams["processed_sec"] / 60).round(1))
    st.dataframe(cams[["district", "cam_index", "состояние", "обработано_мин", "alerts", "pending", "last_seen"]].rename(columns={
        "district": "район", "cam_index": "камера", "alerts": "тревог", "pending": "ждут решения", "last_seen": "последняя активность", "обработано_мин": "обработано, мин"}), hide_index=True)


def journal_view(flt: dict) -> None:
    df = store().list_alerts(limit=2000, **{**flt, "status": ["confirmed", "false", "unsure"]})
    if df.empty:
        st.info("Решений пока нет.")
        return
    view = df[["id", "t_abs", "district", "cam_index", "confidence", "status", "reviewer", "reviewed_at", "note", "demo"]]
    st.dataframe(view.rename(columns={"t_abs": "время события", "district": "район", "cam_index": "камера", "confidence": "уверенность", "status": "решение", "reviewer": "оператор",
                                      "reviewed_at": "решение принято", "note": "заметка", "demo": "демо"}), hide_index=True)
    st.download_button("Скачать CSV", view.to_csv(index=False).encode("utf-8-sig"), file_name="decisions.csv")


def main() -> None:
    flt, refresh = filters()
    st.title("Мониторинг")
    kpis()
    sec = st.radio("Раздел", SECTIONS, horizontal=True, label_visibility="collapsed", key="section")
    opened = st.session_state.get("open_alert")
    if sec == "Тревоги" and opened:
        detail(opened)
    elif sec == "Тревоги":
        st.fragment(run_every=5 if refresh else None)(feed)(flt)
    elif sec == "Камеры":
        cameras_view()
    else:
        journal_view(flt)


main()
