"""Дымовая проверка интерфейса: каждая страница выполняется без исключений на пустом проекте (AppTest, без браузера и моделей)."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_starts_without_exceptions():
    at = st_testing.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src" / "sd" / "ui" / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["page_eval", "page_label", "page_view", "page_models"])
def test_page_renders_standalone(page):
    code = f"import sd._env\nfrom sd.ui import {page}\n{page}.render()\n"
    at = st_testing.AppTest.from_string(code, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
