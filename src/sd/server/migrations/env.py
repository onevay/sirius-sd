"""Окружение Alembic. URL берётся из опции `sqlalchemy.url` или готового соединения `config.attributes["connection"]` (их задаёт `sd.server.migrate`), а не из alembic.ini: ini-файла у проекта нет."""
from __future__ import annotations

from alembic import context

from sd.server import models  # noqa: F401  — регистрирует таблицы в Base.metadata
from sd.server.db import Base, make_engine

config = context.config
target_metadata = Base.metadata

OPTS = dict(target_metadata=target_metadata, compare_type=True, render_as_batch=True)      # batch — чтобы ALTER работал и в SQLite


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True, dialect_opts={"paramstyle": "named"}, **OPTS)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")        # готовое соединение (тесты, встроенный запуск при старте приложения)
    if connection is not None:
        context.configure(connection=connection, **OPTS)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = make_engine(config.get_main_option("sqlalchemy.url"))
    try:
        with engine.connect() as conn:
            context.configure(connection=conn, **OPTS)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
