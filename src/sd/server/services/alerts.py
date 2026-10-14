"""Тревоги: выборки со страницами, решения оператора с историей, статистика, поток обновлений."""
from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median
from typing import Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from .. import clock
from ..models import ALERT_STATUSES, Alert, AlertReview, Camera, User

#: справочник причин для решения «ложная» (расширяется без миграции: колонка — строка)
REASONS = {"drink": "питьё", "phone": "телефон", "eat": "еда", "touch_face": "касание лица", "other_gesture": "другой жест", "no_person": "нет человека",
           "duplicate": "дубль тревоги", "other": "другое"}
MAX_PAGE = 500


class AlertError(ValueError):
    """Недопустимое решение или параметры выборки."""


def filtered(q: Select, *, status: Sequence[str] | None = None, cameras: Sequence[str] | None = None, districts: Sequence[str] | None = None, city: str | None = None,
              min_conf: float | None = None, since: datetime | None = None, until: datetime | None = None, demo: bool | None = None, state: str | None = None) -> Select:
    if status:
        bad = [x for x in status if x not in ALERT_STATUSES]
        if bad:
            raise AlertError(f"status ∈ {', '.join(ALERT_STATUSES)}")
        q = q.where(Alert.status.in_(list(status)))
    if cameras:
        q = q.where(Alert.camera_id.in_(list(cameras)))
    if districts:
        q = q.where(Alert.district.in_(list(districts)))
    if city:
        q = q.where(Alert.camera_id.in_(select(Camera.camera_id).where(Camera.city == city)))
    if min_conf is not None:
        q = q.where(Alert.confidence >= min_conf)
    if since is not None:
        q = q.where(Alert.t_abs >= since)
    if until is not None:
        q = q.where(Alert.t_abs <= until)
    if demo is not None:
        q = q.where(Alert.demo.is_(demo))
    if state:
        q = q.where(Alert.state == state)
    return q


def list_page(s: Session, *, limit: int = 50, offset: int = 0, newest_first: bool = True, **filters) -> tuple[list[Alert], int]:
    """Страница тревог и общее число под фильтр."""
    if limit < 1 or offset < 0:
        raise AlertError("limit ≥ 1, offset ≥ 0")
    limit = min(limit, MAX_PAGE)
    base = filtered(select(Alert), **filters)
    total = int(s.scalar(select(func.count()).select_from(base.subquery())) or 0)
    order = (Alert.t_abs.desc(), Alert.id.desc()) if newest_first else (Alert.t_abs.asc(), Alert.id.asc())
    return list(s.scalars(base.order_by(*order).limit(limit).offset(offset))), total


def get(s: Session, alert_id: int) -> Alert | None:
    return s.get(Alert, alert_id)


def review(s: Session, alert: Alert, status: str, user: User | None = None, *, note: str = "", reason: str | None = None, reviewer_name: str = "") -> AlertReview:
    """Решение оператора: строка истории + обновление текущего статуса. Продление события воркером решение не сбрасывает."""
    if status not in ALERT_STATUSES:
        raise AlertError(f"status ∈ {', '.join(ALERT_STATUSES)}")
    if reason is not None and reason not in REASONS:
        raise AlertError(f"причина ∈ {', '.join(REASONS)}")
    name = (user.display_name or user.username) if user else reviewer_name
    now = clock.now()
    row = AlertReview(alert_id=alert.id, status=status, reason=reason, note=note or None, reviewer_id=user.id if user else None, reviewer_name=name, created_at=now)
    s.add(row)
    alert.status = status
    alert.reviewer = name or None
    alert.reviewer_id = user.id if user else None
    alert.reviewed_at = None if status == "new" else now
    alert.note = note or None
    alert.updated_at = now
    s.flush()
    return row


def bulk_review(s: Session, ids: Sequence[int], status: str, user: User | None, *, note: str = "", reason: str | None = None) -> tuple[list[int], list[int]]:
    """Одно решение для нескольких тревог: (применено, не найдено)."""
    done, missing = [], []
    for i in dict.fromkeys(ids):
        a = s.get(Alert, i)
        if a is None:
            missing.append(i)
            continue
        review(s, a, status, user, note=note, reason=reason)
        done.append(i)
    return done, missing


def history(s: Session, alert_id: int) -> list[AlertReview]:
    return list(s.scalars(select(AlertReview).where(AlertReview.alert_id == alert_id).order_by(AlertReview.id)))


def stats(s: Session, day: str | None = None) -> dict:
    """KPI: по статусам, за день, точность решений, медианное время реакции, число камер. Время реакции считается по последним 5000 решённым тревогам."""
    day = day or clock.now().strftime("%Y-%m-%d")
    d0 = datetime.fromisoformat(day)
    by = dict(s.execute(select(Alert.status, func.count()).group_by(Alert.status)).all())
    today = dict(s.execute(select(Alert.status, func.count()).where(Alert.t_abs >= d0, Alert.t_abs < d0 + timedelta(days=1)).group_by(Alert.status)).all())
    pairs = s.execute(select(Alert.reviewed_at, Alert.t_raised).where(Alert.reviewed_at.is_not(None), Alert.t_raised.is_not(None), Alert.demo.is_(False)).order_by(Alert.id.desc()).limit(5000)).all()
    lat = [(a - b).total_seconds() for a, b in pairs]
    decided = by.get("confirmed", 0) + by.get("false", 0)
    return dict(new=by.get("new", 0), confirmed=by.get("confirmed", 0), false=by.get("false", 0), unsure=by.get("unsure", 0), total=sum(by.values()), today=today,
                cameras=int(s.scalar(select(func.count()).select_from(Camera)) or 0), precision=(by.get("confirmed", 0) / decided) if decided else None,
                median_review_sec=float(median(lat)) if lat else None, max_id=int(s.scalar(select(func.coalesce(func.max(Alert.id), 0))) or 0))


def journal(s: Session, limit: int = 5000) -> list[Alert]:
    """Решённые тревоги для выгрузки журнала решений."""
    return list(s.scalars(select(Alert).where(Alert.status.in_(("confirmed", "false", "unsure"))).order_by(Alert.t_abs.desc(), Alert.id.desc()).limit(limit)))


def events_after(s: Session, last_alert_id: int, last_review_id: int, limit: int = 200) -> tuple[list[Alert], list[AlertReview]]:
    """Новые тревоги и новые решения после курсоров — основа потока обновлений (SSE) и опроса."""
    new = list(s.scalars(select(Alert).where(Alert.id > last_alert_id).order_by(Alert.id).limit(limit)))
    rev = list(s.scalars(select(AlertReview).where(AlertReview.id > last_review_id).order_by(AlertReview.id).limit(limit)))
    return new, rev


def cursors(s: Session) -> tuple[int, int]:
    """Текущие максимумы (id тревоги, id решения): с них начинается поток «только новое»."""
    return (int(s.scalar(select(func.coalesce(func.max(Alert.id), 0))) or 0), int(s.scalar(select(func.coalesce(func.max(AlertReview.id), 0))) or 0))
