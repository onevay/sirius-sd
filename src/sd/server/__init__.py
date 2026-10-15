"""Платформа (FastAPI + SQLAlchemy + Alembic): единый API и веб-интерфейс `/app` для оператора и разработчика, внешних интеграций и воркера.

Слои (зависимости только сверху вниз):
    static/       веб-интерфейс (vanilla JS, без сборки), раздаётся по адресу /app (docs/WEB_UI.md)
    routers/      HTTP: разбор запроса, проверка роли, схемы ответов (`schemas.py`); бизнес-логики нет
    services/     прикладная логика: пользователи, тревоги и решения, камеры, отчёты, аудит; видео и разбор моделью (`videos`); профили и задачи разработчика (`devtools`)
    store.py      `SqlAlertStore` — единственное хранилище тревог: в него пишет воркер (`sd monitor`), его читают API, интерфейс и Dev Studio
    models.py     схема БД; изменения схемы — только через миграции (`migrations/`, команда `sd db`)
    settings.py   конфигурация из переменных окружения (`SD_DB_URL`, `SD_SECRET_KEY`, `SD_AUTH`, …)
Распознавание, обучение и Dev Studio остаются в модулях верхнего уровня (`sd.pipeline`, `sd.tasks`, `sd.experiments` …): API лишь даёт к ним доступ с проверкой прав.
Описание — docs/API.md.
"""
from __future__ import annotations

API_PREFIX = "/api/v1"
__version__ = "1.0.0"
