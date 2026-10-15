"""Конфигурация бэкенда из переменных окружения (и `.env`, который подхватывает `sd._env`).

    SD_DB_URL         строка подключения SQLAlchemy; по умолчанию SQLite-файл `outputs/platform/platform.db`. PostgreSQL — `postgresql+psycopg://user:pass@host/db`
                      (драйвер ставится отдельно; код и миграции от СУБД не зависят)
    SD_SECRET_KEY     секрет подписи токенов доступа; без него создаётся и запоминается `outputs/platform/secret.key` (годится для одной машины)
    SD_AUTH           `off` — без входа (локальная работа: все запросы выполняются от имени администратора). Любое другое значение — вход обязателен
    SD_TOKEN_TTL_MIN  срок жизни токена доступа после входа, минут (по умолчанию 720)
    SD_CORS           разрешённые источники для браузера через запятую (`https://ops.example.com`); пусто — CORS выключен
    SD_API_DOCS       `0` — скрыть /docs и /openapi.json
    SD_UI             `0` — не раздавать веб-интерфейс (/app); сервер остаётся чистым API
    SD_ADMIN_USER / SD_ADMIN_PASSWORD   создать администратора при первом запуске, если пользователей ещё нет
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path

from sqlalchemy.engine import URL

from ..paths import OUTPUTS

PLATFORM_DIR = OUTPUTS / "platform"


def sqlite_url(path: str | Path) -> str:
    return URL.create("sqlite", database=str(path)).render_as_string(hide_password=False)


def _secret_from_file() -> str:
    f = PLATFORM_DIR / "secret.key"
    if f.exists() and f.read_text(encoding="utf-8").strip():
        return f.read_text(encoding="utf-8").strip()
    f.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(48)
    f.write_text(key, encoding="utf-8")
    try:
        os.chmod(f, 0o600)
    except OSError:      # Windows: права файла задаются иначе
        pass
    return key


@dataclass(frozen=True)
class Settings:
    db_url: str
    secret_key: str
    auth_required: bool = True
    token_ttl_sec: int = 12 * 3600
    cors_origins: tuple[str, ...] = ()
    expose_docs: bool = True
    serve_ui: bool = True                  # веб-интерфейс из `sd/server/static` по адресу /app
    allow_fs: bool = False                 # загрузка файлов и подключение папок: только при привязке к локальному адресу
    min_password_len: int = 8
    bootstrap_admin: tuple[str, str] | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, **override) -> "Settings":
        e = os.environ
        admin = (e["SD_ADMIN_USER"], e["SD_ADMIN_PASSWORD"]) if e.get("SD_ADMIN_USER") and e.get("SD_ADMIN_PASSWORD") else None
        s = cls(db_url=e.get("SD_DB_URL") or sqlite_url(PLATFORM_DIR / "platform.db"), secret_key=e.get("SD_SECRET_KEY") or _secret_from_file(),
                auth_required=e.get("SD_AUTH", "on").strip().lower() not in ("off", "0", "false", "no"), token_ttl_sec=int(float(e.get("SD_TOKEN_TTL_MIN", "720")) * 60),
                cors_origins=tuple(x.strip() for x in e.get("SD_CORS", "").split(",") if x.strip()), expose_docs=e.get("SD_API_DOCS", "1") != "0", serve_ui=e.get("SD_UI", "1") != "0", bootstrap_admin=admin)
        return replace(s, **override) if override else s

    @property
    def is_sqlite(self) -> bool:
        return self.db_url.startswith("sqlite")
