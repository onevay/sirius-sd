"""Администрирование: пользователи и журнал действий (роль admin)."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from .. import deps
from ..models import User
from ..schemas import AuditOut, Message, Page, UserCreate, UserOut, UserUpdate
from ..services import audit as AU
from ..services import users as US

router = APIRouter(tags=["администрирование"])


def _target(s: Session, user_id: int) -> User:
    u = s.get(User, user_id)
    if u is None:
        raise HTTPException(404, "пользователь не найден")
    return u


@router.get("/users", response_model=list[UserOut], summary="Пользователи")
def list_users(_: User = Depends(deps.admin), s: Session = Depends(deps.get_session)) -> list:
    return US.list_users(s)


@router.post("/users", response_model=UserOut, status_code=201, summary="Создать пользователя")
def create_user(body: UserCreate, request: Request, ctx: deps.Ctx = Depends(deps.get_ctx), me: User = Depends(deps.admin), s: Session = Depends(deps.get_session)) -> User:
    u = US.create(s, body.username, body.password, body.role, body.display_name, min_password_len=ctx.settings.min_password_len)
    deps.audit(s, request, me, "user.create", "user", u.id, username=u.username, role=u.role)
    return u


@router.get("/users/{user_id}", response_model=UserOut, summary="Пользователь")
def get_user(user_id: int, _: User = Depends(deps.admin), s: Session = Depends(deps.get_session)) -> User:
    return _target(s, user_id)


@router.patch("/users/{user_id}", response_model=UserOut, summary="Изменить роль, имя, блокировку или пароль")
def patch_user(user_id: int, body: UserUpdate, request: Request, ctx: deps.Ctx = Depends(deps.get_ctx), me: User = Depends(deps.admin), s: Session = Depends(deps.get_session)) -> User:
    u = _target(s, user_id)
    US.update(s, u, role=body.role, display_name=body.display_name, is_active=body.is_active, password=body.password, min_password_len=ctx.settings.min_password_len)
    deps.audit(s, request, me, "user.update", "user", u.id, **{k: v for k, v in body.model_dump(exclude_none=True).items() if k != "password"}, password_changed=body.password is not None)
    return u


@router.delete("/users/{user_id}", response_model=Message, summary="Удалить пользователя (история решений сохраняется с его именем)")
def delete_user(user_id: int, request: Request, me: User = Depends(deps.admin), s: Session = Depends(deps.get_session)) -> Message:
    u = _target(s, user_id)
    US.delete(s, u)
    deps.audit(s, request, me, "user.delete", "user", user_id, username=u.username)
    return Message()


@router.get("/audit", response_model=Page[AuditOut], summary="Журнал действий")
def audit_log(action: str | None = None, entity: str | None = None, entity_id: str | None = None, username: str | None = None, since: datetime | None = None, until: datetime | None = None,
              limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0), _: User = Depends(deps.admin), s: Session = Depends(deps.get_session)) -> Page[AuditOut]:
    rows, total = AU.query(s, action=action, entity=entity, entity_id=entity_id, username=username, since=since, until=until, limit=limit, offset=offset)
    return Page(items=[AuditOut.model_validate(r) for r in rows], total=total, limit=limit, offset=offset)
