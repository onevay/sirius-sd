"""Общие элементы Dev Studio: запуск и наблюдение за фоновыми задачами, чек-лист перед запуском, выбор прогона/профиля/классификатора, подтверждение опасных действий.

Правила интерфейса (чтобы не ошибаться):
  * тяжёлое — только фоновой задачей (`task_button`): закрытая вкладка не теряет работу, вторая тяжёлая задача встаёт в очередь;
  * перед запуском — чек-лист (`checklist`): ошибки блокируют кнопку, предупреждения видны рядом с ней;
  * опасное (перезапись, удаление, откат) — через подтверждение (`confirm`);
  * в состоянии сессии — только выбор пользователя; данные читаются с диска (кэш Streamlit по ключу файла).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from sd import experiments as XP
from sd import tasks as T

LEVEL_ICON = {"error": "⛔", "warning": "⚠️", "info": "ℹ️", "ok": "✅"}


def who() -> str:
    """Имя работающего (боковая панель) — пишется в версии разметки, отчёты и задачи."""
    return str(st.session_state.get("who") or "").strip()


# ---------------------------------------------------------------------------------------------- чек-лист
def checklist(items: list[tuple[str, str]], title: str | None = None) -> bool:
    """Показать (уровень, текст) и вернуть True, если нет ошибок. Уровни: error | warning | info | ok."""
    if title:
        st.markdown(f"**{title}**")
    has_err = False
    for lvl, text in items:
        has_err |= lvl == "error"
        st.markdown(f"{LEVEL_ICON.get(lvl, '•')} {text}")
    return not has_err


# ---------------------------------------------------------------------------------------------- задачи
def task_button(task_id: str, values: dict, label: str, *, key: str, disabled: bool = False, help: str | None = None, primary: bool = True) -> str | None:
    """Кнопка «поставить задачу». Возвращает id поставленной задачи (и запоминает её для панели `task_panel`)."""
    if st.button(label, key=key, type="primary" if primary else "secondary", disabled=disabled, help=help):
        try:
            tid = T.start(task_id, values)
        except Exception as e:      # noqa: BLE001
            st.error(f"задача не запущена: {e}")
            return None
        st.session_state.setdefault("my_tasks", []).insert(0, tid)
        state = T.status(tid)["state"]
        st.toast(f"{T.CATALOG[task_id].title}: {state}" + (" — дождётся окончания текущей тяжёлой задачи" if state == "в очереди" else ""), icon="⏳")
        return tid
    return None


def _task_row(m: dict, key: str, show_log: bool) -> None:
    icon = {"идёт": "⏳", "в очереди": "🕓", "готово": "✅", "ошибка": "⛔", "остановлена": "⏹", "прервана": "⚠️"}.get(m["state"], "•")
    c = st.columns([5, 2, 1.2, 1.2])
    c[0].markdown(f"{icon} **{m['title']}** · `{m['id']}`")
    c[1].caption(f"{m['state']} · {int(m['seconds'])} с" if m["state"] != "в очереди" else "в очереди")
    if m["state"] in ("идёт", "в очереди"):
        if c[2].button("Стоп", key=f"{key}_stop_{m['id']}"):
            T.stop(m["id"])
            st.rerun()
    elif c[2].button("Повторить", key=f"{key}_re_{m['id']}", help="поставить ту же задачу с теми же значениями"):
        T.rerun(m["id"])
        st.rerun()
    if m.get("progress"):
        p = m["progress"]
        st.progress(p["frac"], text=f"{p['done']}/{p['total']} · {p['message']}")
    if show_log:
        with st.expander("журнал", expanded=m["state"] == "ошибка"):
            st.code(T.tail(m["id"], 40) or "(пусто)", language=None)
    if m["state"] == "ошибка":
        st.caption("Задача завершилась с ошибкой — последние строки журнала выше.")


def task_panel(kinds: tuple[str, ...] | None = None, key: str = "tp", limit: int = 5, show_log: bool = True, title: str | None = "Задачи") -> None:
    """Последние задачи указанных типов с прогрессом (обновляется каждые 3 с, не перерисовывая страницу)."""

    @st.fragment(run_every=3)
    def _panel() -> None:
        rows = [r for r in T.list_tasks(limit=40) if kinds is None or r["task"] in kinds][:limit]
        if title:
            st.markdown(f"**{title}**")
        if not rows:
            st.caption("задач этого вида ещё не было")
            return
        for m in rows:
            _task_row(m, key, show_log)

    _panel()


def finished_since(task_id: str, after: float) -> dict | None:
    """Последняя завершённая задача типа task_id, начатая после момента `after` (для «показать результат, как только готово»)."""
    for r in T.list_tasks(limit=40):
        if r["task"] == task_id and r["state"] == "готово" and (r.get("created") or 0) >= after:
            return r
    return None


# ---------------------------------------------------------------------------------------------- выбор
@st.cache_data(ttl=10, show_spinner=False)
def _runs(stamp: float) -> pd.DataFrame:
    return XP.list_runs()


def runs_df() -> pd.DataFrame:
    stamp = XP.EXP_DIR.stat().st_mtime if XP.EXP_DIR.exists() else 0.0
    return _runs(stamp)


def run_label(r) -> str:
    f1 = "—" if pd.isna(r.f1) else f"{r.f1:.3f}"
    return f"{r.created} · {r.name} · F1 {f1} · {r.clips} клипов · {r.cycle_model or '—'}" + (f" · сбоев {r.failed}" if r.failed else "")


def run_select(label: str, key: str, mode: str | None = "events", allow_none: bool = False, prefer: str | None = None) -> str | None:
    """Выбор прогона из журнала (новые сверху). `prefer` — id, который выбрать, если он есть (переходы между вкладками)."""
    runs = runs_df()
    if mode and len(runs):
        runs = runs[runs["mode"] == mode]
    if runs.empty:
        st.info("Прогонов в журнале нет: запустите оценку (вкладка «Валидация → Запуск»).")
        return None
    ids = runs.id.tolist()
    opts = ([None] if allow_none else []) + ids
    want = prefer or st.session_state.get(key)
    idx = opts.index(want) if want in opts else 0
    lab = {r.id: run_label(r) for r in runs.itertuples()}
    return st.selectbox(label, opts, index=idx, key=key, format_func=lambda i: "— не выбран —" if i is None else lab.get(i, i))


def confirm(label: str, key: str, warning: str, action, *, disabled: bool = False, help: str | None = None) -> None:
    """Кнопка опасного действия: открывает диалог с предупреждением; действие выполняется только после явного «Да»."""
    @st.dialog("Подтвердите действие")
    def _dlg() -> None:
        st.warning(warning)
        c = st.columns(2)
        if c[0].button("Да, выполнить", type="primary", key=key + "_yes"):
            try:
                res = action()
                st.session_state[key + "_result"] = ("ok", res)
            except Exception as e:      # noqa: BLE001
                st.session_state[key + "_result"] = ("error", f"{type(e).__name__}: {e}")
            st.rerun()
        if c[1].button("Отмена", key=key + "_no"):
            st.rerun()

    if st.button(label, key=key, disabled=disabled, help=help):
        _dlg()
    res = st.session_state.pop(key + "_result", None)
    if res:
        (st.success if res[0] == "ok" else st.error)(str(res[1]) if res[1] is not None else "готово")


def file_mtime(p: str | Path) -> float:
    p = Path(p)
    return p.stat().st_mtime if p.exists() else 0.0


def ago(ts: float | None) -> str:
    if not ts:
        return "—"
    s = time.time() - ts
    return f"{int(s)} с назад" if s < 90 else (f"{int(s / 60)} мин назад" if s < 5400 else f"{s / 3600:.1f} ч назад")
