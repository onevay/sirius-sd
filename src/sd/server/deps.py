"""Зависимости FastAPI: контекст приложения, сессия БД, текущий пользователь и проверка роли.

Как определяется пользователь (по порядку):
  1. заголовок `Authorization: Bearer <токен доступа | API-токен sdt_…>`;
  2. cookie `sd_token` — только для безопасных методов (GET/HEAD), чтобы не открывать CSRF: изменяющие запросы всегда с заголовком;
  3. параметр `?access_token=` — только для GET-маршрутов `/media/*`, `/alerts/{id}/thumb|clip` и `/events` (теги <video>, <img> и EventSource не умеют заголовков).
При `SD_AUTH=off` вход не требуется: запрос выполняется от имени локального администратора `local` (id=None — в аудите и решениях пользователь не указывается).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from . import security
from .db import Database
from .models import ROLES, User
from .services import audit as AU
from .services import users as US
from .settings import Settings

COOKIE = "sd_token"
SAFE = ("GET", "HEAD", "OPTIONS")
QUERY_TOKEN_PATHS = ("/media/", "/events", "/thumb", "/clip")


@dataclass
class Ctx:
    """Объекты, живущие столько же, сколько приложение."""

    settings: Settings
    db: Database
    legacy_factory: Callable[[], object]          # создаёт прежний WebApp (анализ видео, профили, задачи) — лениво: он тянет тяжёлые модули
    _legacy: object | None = None
    login_failures: dict = field(default_factory=dict)

    @property
    def legacy(self):
        if self._legacy is None:
            legacy = self.legacy_factory()
            if hasattr(legacy, "allow_fs"):
                legacy.allow_fs = self.settings.allow_fs      # загрузка файлов и подключение папок — только по настройке, что бы ни вернула фабрика
            self._legacy = legacy
        return self._legacy


def get_ctx(request: Request) -> Ctx:
    return request.app.state.ctx


def get_session(ctx: Ctx = Depends(get_ctx)) -> Iterator[Session]:
    s = ctx.db.sessions()
    try:
        yield s
        s.commit()
    except BaseException:
        s.rollback()
        raise
    finally:
        s.close()


def client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def _token_from(request: Request) -> tuple[str | None, str]:
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer ") and h[7:].strip():
        return h[7:].strip(), "header"
    if request.method in SAFE and (c := request.cookies.get(COOKIE)):
        return c, "cookie"
    q = request.query_params.get("access_token")
    if q and request.method == "GET" and any(p in request.url.path for p in QUERY_TOKEN_PATHS):
        return q, "query"
    return None, ""


LOCAL_ADMIN = dict(username="local", display_name="Локальный администратор", role="admin")


def current_user(request: Request, ctx: Ctx = Depends(get_ctx), s: Session = Depends(get_session)) -> User:
    if not ctx.settings.auth_required:
        return User(id=None, is_active=True, token_version=0, password_hash="", **LOCAL_ADMIN)
    token, _src = _token_from(request)
    if not token:
        raise HTTPException(401, "требуется вход", headers={"WWW-Authenticate": "Bearer"})
    user: User | None = None
    if security.is_api_token(token):
        user = US.user_by_api_token(s, token)
    else:
        got = security.read_access_token(ctx.settings.secret_key, token, ctx.settings.token_ttl_sec)
        if got:
            u = s.get(User, got[0])
            user = u if u is not None and u.is_active and u.token_version == got[1] else None
    if user is None:
        raise HTTPException(401, "токен недействителен или истёк", headers={"WWW-Authenticate": "Bearer"})
    return user


def require(role: str) -> Callable[..., User]:
    """Зависимость: пользователь с ролью не ниже `role`."""
    if role not in ROLES:
        raise ValueError(role)

    def dep(user: User = Depends(current_user)) -> User:
        if not security.has_role(user.role, role):
            raise HTTPException(403, f"нужна роль {role} или выше")
        return user

    dep.__name__ = f"require_{role}"
    return dep


viewer, operator, developer, admin = (require(r) for r in ROLES)


def audit(s: Session, request: Request, user: User | None, action: str, entity: str = "", entity_id="", **details) -> None:
    AU.record(s, user, action, entity, entity_id, details or None, client_ip(request))


__all__ = ["Ctx", "COOKIE", "get_ctx", "get_session", "current_user", "require", "viewer", "operator", "developer", "admin", "audit", "client_ip"]
