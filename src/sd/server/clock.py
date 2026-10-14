"""Единый источник времени бэкенда. Все времена в БД — локальное «наивное» время машины (как у камер-папок и прежней SQLite-базы мониторинга).
Переход на UTC — это замена одной функции и миграция данных: больше нигде `datetime.now()` не вызывается."""
from __future__ import annotations

from datetime import datetime

ISO = "%Y-%m-%dT%H:%M:%S"


def now() -> datetime:
    return datetime.now().replace(microsecond=0)


def parse(value: str | datetime | None) -> datetime | None:
    """ISO-строка (`YYYY-MM-DDTHH:MM:SS`, допускается пробел и дробные секунды) → datetime; None и пустое → None. Ошибка формата — ValueError."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None, microsecond=0)
    return datetime.fromisoformat(str(value).strip().replace("Z", "")).replace(tzinfo=None, microsecond=0)


def fmt(value: datetime | None) -> str | None:
    return value.strftime(ISO) if value else None
