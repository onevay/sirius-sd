"""Подключение к БД: движок, фабрика сессий, соглашение об именах ограничений (нужно миграциям и пакетной перестройке таблиц SQLite)."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import MetaData, create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

NAMING = {"ix": "ix_%(column_0_label)s", "uq": "uq_%(table_name)s_%(column_0_name)s", "ck": "ck_%(table_name)s_%(constraint_name)s",
          "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s", "pk": "pk_%(table_name)s"}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


def make_engine(url: str) -> Engine:
    """Движок. SQLite: внешние ключи включены, WAL и ожидание блокировки (воркер и API — разные процессы); в памяти — одно общее соединение."""
    u = make_url(url)
    kw: dict = dict(pool_pre_ping=True)
    sqlite = u.get_backend_name() == "sqlite"
    memory = sqlite and u.database in (None, "", ":memory:")
    if sqlite:
        kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if memory:
            kw["poolclass"] = StaticPool
        else:
            Path(u.database).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, **kw)
    if sqlite:
        @event.listens_for(eng, "connect")
        def _pragmas(dbapi, _rec) -> None:      # noqa: ANN001
            cur = dbapi.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            if not memory:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return eng


class Database:
    """Движок + фабрика сессий. Схема создаётся только миграциями (`sd db upgrade`, `migrate.upgrade`), не `create_all`."""

    def __init__(self, url: str):
        self.url = url
        self.engine = make_engine(url)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False, class_=Session)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Сессия с фиксацией при успехе и откатом при исключении."""
        s = self.sessions()
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
        finally:
            s.close()

    def dispose(self) -> None:
        self.engine.dispose()
