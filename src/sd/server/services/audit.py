"""Журнал действий: запись (только добавление) и чтение с фильтрами."""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import clock
from ..models import AuditLog, User

MAX_DETAILS = 4000      # знаков JSON в деталях: журнал не должен разрастаться из-за одного большого запроса


def record(s: Session, user: User | None, action: str, entity: str = "", entity_id: Any = "", details: dict | None = None, ip: str = "") -> AuditLog:
    d = details
    if d is not None and len(str(d)) > MAX_DETAILS:
        d = {"truncated": str(d)[:MAX_DETAILS]}
    row = AuditLog(ts=clock.now(), user_id=user.id if user and user.id else None, username=user.username if user else "", action=action[:64], entity=entity[:32],
                   entity_id=str(entity_id)[:128], details=d, ip=ip[:64])
    s.add(row)
    return row


def query(s: Session, *, action: str | None = None, entity: str | None = None, entity_id: str | None = None, username: str | None = None, since=None, until=None,
          limit: int = 100, offset: int = 0) -> tuple[list[AuditLog], int]:
    q = select(AuditLog)
    for col, v in ((AuditLog.action, action), (AuditLog.entity, entity), (AuditLog.entity_id, entity_id), (AuditLog.username, username)):
        if v:
            q = q.where(col == v)
    if since:
        q = q.where(AuditLog.ts >= since)
    if until:
        q = q.where(AuditLog.ts <= until)
    total = s.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = list(s.scalars(q.order_by(AuditLog.id.desc()).limit(limit).offset(offset)))
    return rows, int(total)
