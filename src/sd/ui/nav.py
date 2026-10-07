"""Страницы приложения и переходы между ними (контекст передаётся через `st.session_state`)."""
from __future__ import annotations

import streamlit as st

PAGES: dict[str, "st.Page"] = {}


def go(page: str, **state) -> None:
    """Перейти на страницу, предварительно положив значения в session_state (например, какой клип открыть в просмотре)."""
    st.session_state.update(state)
    st.switch_page(PAGES[page])
