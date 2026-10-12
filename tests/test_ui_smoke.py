"""Дымовая проверка Dev Studio: приложение и каждый раздел каждой вкладки выполняются без исключений (AppTest, без браузера; данные — проекта как есть)."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

SECTIONS = {
    "p_validation": ("val_section", ["Запуск", "Результат и PDF", "Анализ кэша", "Поток"]),
    "p_analytics": ("an_section", ["Журнал", "Сравнение подходов", "По клипам", "Ошибки и порог", "Влияние параметров", "Классификаторы"]),
    "p_designer": (None, [None]),
    "p_tasks": ("tasks_section", ["Разметка событий", "Метки жестов", "Признаки", "Обучение", "Результаты", "Очередь задач"]),
    "p_data": ("data_section", ["Каталог", "Проблемы", "Версии разметки", "Метки циклов", "Обратная связь оператора"]),
    "p_models": ("models_section", ["Профили", "Реестр", "Подключить модель", "Проверить модель", "Перенос", "Фото-модель"]),
    "p_profiling": ("prof_section", ["Скорость по кэшу", "Лестница вычислений", "Машина"]),
    "p_release": ("rel_section", ["Новый выпуск", "Выпуски", "Установить решатель"]),
}


def test_app_starts_without_exceptions():
    at = st_testing.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src" / "sd" / "ui" / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page,section", [(p, s) for p, (_, secs) in SECTIONS.items() for s in secs])
def test_every_section_renders(page, section, tmp_path, monkeypatch):
    monkeypatch.setenv("SD_MONITOR_DB", str(tmp_path / "out" / "monitor.db"))      # база мониторинга — во временный каталог, не в проект
    at = st_testing.AppTest.from_string(f"import sd._env\nfrom sd.ui import {page}\n{page}.render()\n", default_timeout=240)
    if SECTIONS[page][0]:
        at.session_state[SECTIONS[page][0]] = section
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_page_stream_renders():
    at = st_testing.AppTest.from_string("import sd._env\nfrom sd.ui import page_stream\npage_stream.render()\n", default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
