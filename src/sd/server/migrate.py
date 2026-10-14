"""Управление схемой БД: миграции Alembic без alembic.ini (путь к скриптам и URL задаются программно). Команды — `sd db …`.

Правила работы со схемой:
  * схему меняют только миграцией: правка `models.py` → `sd db revision -m "что изменилось"` (автогенерация сравнивает БД с моделями) → проверить файл → `sd db upgrade`;
  * миграции не редактируют после того, как они попали на чужую машину — добавляют новую;
  * у каждой миграции есть `downgrade`; тест `test_migrations_*` проверяет подъём, откат и отсутствие расхождения с моделями.
"""
from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

SCRIPTS = Path(__file__).parent / "migrations"


def config(url: str | None = None, connection: Connection | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(SCRIPTS))
    cfg.set_main_option("path_separator", "os")
    if url:
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def heads() -> list[str]:
    return list(ScriptDirectory.from_config(config()).get_heads())


def history() -> list[dict]:
    """Цепочка миграций от первой к последней: {revision, down, doc}."""
    sd = ScriptDirectory.from_config(config())
    return [dict(revision=r.revision, down=r.down_revision, doc=(r.doc or "").splitlines()[0]) for r in reversed(list(sd.walk_revisions()))]


def current(url: str) -> list[str]:
    """Версия схемы в БД (пусто — миграции не применялись)."""
    eng = create_engine(url)
    try:
        with eng.connect() as c:
            return list(MigrationContext.configure(c).get_current_heads())
    finally:
        eng.dispose()


def is_current(url: str) -> bool:
    return sorted(current(url)) == sorted(heads())


def upgrade(url: str, revision: str = "head", connection: Connection | None = None) -> None:
    command.upgrade(config(url, connection), revision)


def downgrade(url: str, revision: str = "-1", connection: Connection | None = None) -> None:
    command.downgrade(config(url, connection), revision)


def stamp(url: str, revision: str = "head") -> None:
    command.stamp(config(url), revision)


def new_revision(url: str, message: str, autogenerate: bool = True) -> Path:
    """Создаёт файл миграции (автогенерация сравнивает БД по `url` с `models.py`; БД должна быть на актуальной версии)."""
    rev = command.revision(config(url), message=message, autogenerate=autogenerate)
    return Path(rev.path)  # type: ignore[union-attr]
