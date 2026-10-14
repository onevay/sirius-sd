"""Пароли, токены и роли. Только стандартная библиотека и `itsdangerous` (уже есть в окружении Streamlit).

  * пароль — scrypt с индивидуальной солью (`scrypt$n$r$p$соль$хэш`), сравнение за постоянное время;
  * токен доступа — подписанная строка с `uid` и `ver` (версия токенов пользователя) и сроком жизни: смена пароля или блокировка (`users.token_version`) отзывает все выданные;
  * API-токен — случайная строка `sdt_…`; в БД лежит только sha256.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .models import ROLES

_N, _R, _P = 2 ** 14, 8, 1
API_TOKEN_PREFIX = "sdt_"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def hash_password(password: str, *, n: int = _N) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=_R, p=_P, dklen=32, maxmem=128 * 1024 * 1024)
    return f"scrypt${n}${_R}${_P}${_b64(salt)}${_b64(h)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, h = stored.split("$")
        if scheme != "scrypt":
            return False
        calc = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=len(base64.b64decode(h)), maxmem=128 * 1024 * 1024)
        return hmac.compare_digest(calc, base64.b64decode(h))
    except (ValueError, TypeError):
        return False


DUMMY_HASH = hash_password("dummy-password-for-timing")        # проверка несуществующего пользователя тратит столько же времени, сколько настоящего


def password_problem(password: str, min_len: int = 8) -> str | None:
    if len(password) < min_len:
        return f"пароль короче {min_len} символов"
    if password.lower() == password or password.upper() == password or not any(c.isdigit() for c in password):
        return "пароль должен содержать строчные и заглавные буквы и цифру"
    return None


# ---------------------------------------------------------------------------------------------- токен доступа
def _ser(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt="sd-access-v1")


def make_access_token(secret: str, user_id: int, version: int) -> str:
    return _ser(secret).dumps({"uid": int(user_id), "ver": int(version)})


def read_access_token(secret: str, token: str, ttl_sec: int) -> tuple[int, int] | None:
    """(uid, ver) или None, если подпись неверна или срок истёк."""
    try:
        d = _ser(secret).loads(token, max_age=ttl_sec)
        return int(d["uid"]), int(d["ver"])
    except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------- API-токен
def new_api_token() -> tuple[str, str, str]:
    """(токен, префикс для списка, sha256 для БД)."""
    tok = API_TOKEN_PREFIX + secrets.token_urlsafe(32)
    return tok, tok[:12], hash_api_token(tok)


def hash_api_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_api_token(token: str) -> bool:
    return token.startswith(API_TOKEN_PREFIX)


# ---------------------------------------------------------------------------------------------- роли
def role_rank(role: str) -> int:
    return ROLES.index(role) if role in ROLES else -1


def has_role(role: str, needed: str) -> bool:
    """Роль `role` включает права роли `needed` (viewer < operator < developer < admin)."""
    return role_rank(role) >= role_rank(needed) >= 0
