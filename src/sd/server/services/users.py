"""Пользователи, вход и API-токены."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import clock, security
from ..models import ROLES, ApiToken, User

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")


class UserError(ValueError):
    """Ошибка данных пользователя (имя занято, слабый пароль, неизвестная роль)."""


def norm(username: str) -> str:
    return username.strip().lower()


def get_by_username(s: Session, username: str) -> User | None:
    return s.scalar(select(User).where(User.username == norm(username)))


def count(s: Session) -> int:
    return int(s.scalar(select(func.count()).select_from(User)) or 0)


def create(s: Session, username: str, password: str, role: str = "viewer", display_name: str = "", *, min_password_len: int = 8) -> User:
    name = norm(username)
    if not USERNAME_RE.match(name):
        raise UserError("имя: 2–63 символа — латинские буквы, цифры, точка, дефис, подчёркивание; начинается с буквы или цифры")
    if role not in ROLES:
        raise UserError(f"роль ∈ {', '.join(ROLES)}")
    if (p := security.password_problem(password, min_password_len)) is not None:
        raise UserError(p)
    if get_by_username(s, name):
        raise UserError(f"пользователь {name} уже есть")
    u = User(username=name, display_name=display_name.strip() or name, password_hash=security.hash_password(password), role=role, is_active=True, token_version=1, created_at=clock.now())
    s.add(u)
    s.flush()
    return u


def authenticate(s: Session, username: str, password: str) -> User | None:
    """Пользователь при верном пароле и активной учётной записи, иначе None (время ответа не зависит от того, есть ли такое имя)."""
    u = get_by_username(s, username)
    ok = security.verify_password(password, u.password_hash if u else security.DUMMY_HASH)
    if not (u and ok and u.is_active):
        return None
    u.last_login_at = clock.now()
    return u


def update(s: Session, u: User, *, role: str | None = None, display_name: str | None = None, is_active: bool | None = None, password: str | None = None,
           min_password_len: int = 8) -> User:
    if role is not None:
        if role not in ROLES:
            raise UserError(f"роль ∈ {', '.join(ROLES)}")
        if u.role == "admin" and role != "admin" and _admins(s) <= 1 and u.is_active:
            raise UserError("нельзя понизить последнего администратора")
        u.role = role
    if is_active is not None:
        if not is_active and u.role == "admin" and u.is_active and _admins(s) <= 1:
            raise UserError("нельзя заблокировать последнего администратора")
        u.is_active = is_active
    if display_name is not None:
        u.display_name = display_name.strip() or u.username
    if password is not None:
        if (p := security.password_problem(password, min_password_len)) is not None:
            raise UserError(p)
        u.password_hash = security.hash_password(password)
    if role is not None or is_active is not None or password is not None:
        u.token_version += 1             # прежние токены доступа перестают действовать
    return u


def delete(s: Session, u: User) -> None:
    if u.role == "admin" and u.is_active and _admins(s) <= 1:
        raise UserError("нельзя удалить последнего администратора")
    s.delete(u)


def _admins(s: Session) -> int:
    return int(s.scalar(select(func.count()).select_from(User).where(User.role == "admin", User.is_active.is_(True))) or 0)


def list_users(s: Session) -> list[User]:
    return list(s.scalars(select(User).order_by(User.username)))


def bootstrap_admin(s: Session, username: str, password: str, min_password_len: int = 8) -> User | None:
    """Создаёт первого администратора, если пользователей ещё нет (иначе None)."""
    if count(s):
        return None
    return create(s, username, password, "admin", "Администратор", min_password_len=min_password_len)


# ---------------------------------------------------------------------------------------------- API-токены
def create_api_token(s: Session, u: User, name: str, ttl_days: int | None = None) -> tuple[ApiToken, str]:
    tok, prefix, h = security.new_api_token()
    exp = clock.now() + timedelta(days=ttl_days) if ttl_days else None
    row = ApiToken(user_id=u.id, name=name.strip()[:120] or "token", prefix=prefix, token_hash=h, created_at=clock.now(), expires_at=exp)
    s.add(row)
    s.flush()
    return row, tok


def user_by_api_token(s: Session, token: str) -> User | None:
    """Владелец действующего API-токена (не отозван, не истёк, учётная запись активна); время последнего использования обновляется не чаще раза в минуту."""
    row = s.scalar(select(ApiToken).where(ApiToken.token_hash == security.hash_api_token(token)))
    now = clock.now()
    if row is None or row.revoked_at is not None or (row.expires_at is not None and row.expires_at <= now):
        return None
    u = s.get(User, row.user_id)
    if u is None or not u.is_active:
        return None
    if row.last_used_at is None or (now - row.last_used_at).total_seconds() > 60:
        row.last_used_at = now
    return u


def list_api_tokens(s: Session, u: User | None = None) -> list[ApiToken]:
    q = select(ApiToken).order_by(ApiToken.id.desc())
    if u is not None:
        q = q.where(ApiToken.user_id == u.id)
    return list(s.scalars(q))


def revoke_api_token(s: Session, token_id: int, owner: User | None = None) -> ApiToken | None:
    """Отзывает токен; `owner` — если задан, чужой токен не отзывается (None)."""
    row = s.get(ApiToken, token_id)
    if row is None or (owner is not None and row.user_id != owner.id):
        return None
    if row.revoked_at is None:
        row.revoked_at = clock.now()
    return row


def purge_expired(s: Session, older_than_days: int = 30) -> int:
    """Удаляет отозванные и давно истёкшие токены."""
    cut: datetime = clock.now() - timedelta(days=older_than_days)
    n = 0
    for t in s.scalars(select(ApiToken).where((ApiToken.revoked_at < cut) | (ApiToken.expires_at < cut))):
        s.delete(t)
        n += 1
    return n
