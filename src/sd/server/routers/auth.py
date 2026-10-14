"""Вход, текущий пользователь, смена пароля, API-токены."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from .. import deps, security
from ..models import User
from ..schemas import ApiTokenCreate, ApiTokenCreated, ApiTokenOut, LoginIn, Message, PasswordChange, TokenOut, UserOut
from ..services import users as US

router = APIRouter(prefix="/auth", tags=["вход"])

MAX_FAILURES, LOCK_SEC = 5, 60


def _locked(ctx: deps.Ctx, key: tuple) -> int:
    n, until = ctx.login_failures.get(key, (0, 0.0))
    return int(until - time.time()) + 1 if n >= MAX_FAILURES and until > time.time() else 0


def _fail(ctx: deps.Ctx, key: tuple) -> None:
    n, _ = ctx.login_failures.get(key, (0, 0.0))
    ctx.login_failures[key] = (n + 1, time.time() + LOCK_SEC)
    if len(ctx.login_failures) > 10000:      # защита памяти: забываем самые старые записи
        for k in sorted(ctx.login_failures, key=lambda k: ctx.login_failures[k][1])[:5000]:
            ctx.login_failures.pop(k, None)


@router.post("/login", response_model=TokenOut, summary="Вход по имени и паролю")
def login(body: LoginIn, request: Request, response: Response, ctx: deps.Ctx = Depends(deps.get_ctx), s: Session = Depends(deps.get_session)) -> TokenOut:
    key = (deps.client_ip(request), US.norm(body.username))
    if wait := _locked(ctx, key):
        raise HTTPException(429, f"слишком много неудачных попыток; повторите через {wait} с", headers={"Retry-After": str(wait)})
    u = US.authenticate(s, body.username, body.password)
    if u is None:
        _fail(ctx, key)
        deps.audit(s, request, None, "auth.login_failed", "user", key[1])
        s.commit()          # запись о неудачном входе сохраняется, несмотря на ошибку ответа
        raise HTTPException(401, "неверное имя или пароль")
    ctx.login_failures.pop(key, None)
    tok = security.make_access_token(ctx.settings.secret_key, u.id, u.token_version)
    deps.audit(s, request, u, "auth.login", "user", u.id)
    if body.cookie:
        response.set_cookie(deps.COOKIE, tok, httponly=True, samesite="lax", max_age=ctx.settings.token_ttl_sec, secure=request.url.scheme == "https", path="/")
    return TokenOut(access_token=tok, expires_in=ctx.settings.token_ttl_sec, user=UserOut.model_validate(u))


@router.post("/logout", response_model=Message, summary="Выход (удаляет cookie)")
def logout(response: Response) -> Message:
    response.delete_cookie(deps.COOKIE, path="/")
    return Message()


@router.post("/logout-all", response_model=Message, summary="Отозвать все выданные токены доступа этого пользователя")
def logout_all(request: Request, response: Response, user: User = Depends(deps.current_user), s: Session = Depends(deps.get_session)) -> Message:
    if user.id is None:
        return Message(detail="вход отключён: отзывать нечего")
    user.token_version += 1
    deps.audit(s, request, user, "auth.logout_all", "user", user.id)
    response.delete_cookie(deps.COOKIE, path="/")
    return Message()


@router.get("/me", response_model=UserOut, summary="Текущий пользователь")
def me(user: User = Depends(deps.current_user)) -> UserOut:
    return UserOut.model_validate(user)


@router.post("/password", response_model=Message, summary="Смена своего пароля (прежние токены доступа перестают действовать)")
def change_password(body: PasswordChange, request: Request, ctx: deps.Ctx = Depends(deps.get_ctx), user: User = Depends(deps.current_user), s: Session = Depends(deps.get_session)) -> Message:
    if user.id is None:
        raise HTTPException(400, "вход отключён: менять нечего")
    if not security.verify_password(body.old_password, user.password_hash):
        raise HTTPException(403, "неверный текущий пароль")
    US.update(s, user, password=body.new_password, min_password_len=ctx.settings.min_password_len)
    deps.audit(s, request, user, "auth.password", "user", user.id)
    return Message(detail="пароль изменён; войдите заново")


@router.get("/tokens", response_model=list[ApiTokenOut], summary="Мои API-токены")
def tokens(user: User = Depends(deps.current_user), s: Session = Depends(deps.get_session)) -> list:
    return US.list_api_tokens(s, user) if user.id is not None else []


@router.post("/tokens", response_model=ApiTokenCreated, status_code=201, summary="Создать API-токен (показывается один раз)")
def create_token(body: ApiTokenCreate, request: Request, user: User = Depends(deps.current_user), s: Session = Depends(deps.get_session)) -> ApiTokenCreated:
    if user.id is None:
        raise HTTPException(400, "вход отключён: токены не нужны")
    row, tok = US.create_api_token(s, user, body.name, body.ttl_days)
    deps.audit(s, request, user, "token.create", "api_token", row.id, name=row.name)
    return ApiTokenCreated(**ApiTokenOut.model_validate(row).model_dump(), token=tok)


@router.delete("/tokens/{token_id}", response_model=Message, summary="Отозвать свой API-токен")
def revoke_token(token_id: int, request: Request, user: User = Depends(deps.current_user), s: Session = Depends(deps.get_session)) -> Message:
    if user.id is None or US.revoke_api_token(s, token_id, owner=user) is None:
        raise HTTPException(404, "токен не найден")
    deps.audit(s, request, user, "token.revoke", "api_token", token_id)
    return Message()
