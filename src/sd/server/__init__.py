"""Бэкенд платформы (FastAPI + SQLAlchemy + Alembic): единый API для веб-приложения оператора, Dev Studio и внешних интеграций.

Слои (зависимости только сверху вниз):
    routers/      HTTP: разбор запроса, проверка роли, схемы ответов (`schemas.py`); бизнес-логики нет
    services/     прикладная логика над сессией БД: пользователи, тревоги и решения, камеры, аудит
    store.py      `SqlAlertStore` — тот же интерфейс, что у `realtime.store.AlertStore`, поэтому воркер и `WebApp` работают с новой БД без правок
    models.py     схема БД; изменения схемы — только через миграции (`migrations/`, команда `sd db`)
    settings.py   конфигурация из переменных окружения (`SD_DB_URL`, `SD_SECRET_KEY`, `SD_AUTH`, …)
Распознавание, обучение и Dev Studio остаются в прежних модулях (`sd.pipeline`, `sd.tasks`, `sd.experiments`, `sd.web.api.WebApp` …): API лишь даёт к ним доступ с проверкой прав.
Описание — docs/API.md.
"""
from __future__ import annotations

API_PREFIX = "/api/v1"
__version__ = "1.0.0"
