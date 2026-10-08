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


@pytest.mark.parametrize("page", ["page_experiments", "page_monitor"])
def test_new_dev_pages_render(page, tmp_path, monkeypatch):
    monkeypatch.setenv("SD_MONITOR_DB", str(tmp_path / "out" / "monitor.db"))      # база мониторинга — во временный каталог, не в проект
    code = f"import sd._env\nfrom sd.ui import {page}\n{page}.render()\n"
    at = st_testing.AppTest.from_string(code, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_user_app_renders_empty_and_with_demo_data(tmp_path, monkeypatch):
    monkeypatch.setenv("SD_MONITOR_DB", str(tmp_path / "out" / "monitor.db"))
    path = str(Path(__file__).resolve().parents[1] / "src" / "sd" / "ui" / "user_app.py")
    at = st_testing.AppTest.from_file(path, default_timeout=60).run()
    assert not at.exception and any("Тревог пока нет" in i.value for i in at.info)
    from sd.realtime.demo import seed_demo
    from sd.realtime.store import AlertStore

    st = AlertStore()
    assert seed_demo(st, 4, root=tmp_path / "art", with_clips=False) == 4
    at = st_testing.AppTest.from_file(path, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len([b for b in at.button if b.label == "Подтвердить"]) == 4
