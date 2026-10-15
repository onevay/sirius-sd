"""Тревоги: список со страницами, карточка, решения оператора с историей, статистика, журнал CSV, файлы, поток обновлений (SSE), демо-данные."""
from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session

from .. import deps
from ..models import ALERT_STATUSES, Alert, User
from ..schemas import AlertOut, AlertStatus, BulkReviewIn, BulkReviewOut, DemoIn, Page, ReviewIn, ReviewOut, StatsOut
from ..services import alerts as AS
from ..services import artifacts as AF
from ..services import reports as RS

router = APIRouter(tags=["тревоги"])

STATUS_RU = {"new": "ждёт решения", "confirmed": "подтверждена", "false": "ложная", "unsure": "не уверен"}


def to_out(a: Alert) -> AlertOut:
    out = AlertOut.model_validate(a)
    out.has_thumb, out.has_clip = AF.flags(a).values()
    return out


def _get(s: Session, alert_id: int) -> Alert:
    a = AS.get(s, alert_id)
    if a is None:
        raise HTTPException(404, "тревога не найдена")
    return a


@router.get("/alerts", response_model=Page[AlertOut], summary="Список тревог (новые сверху)")
def list_alerts(status: list[AlertStatus] | None = Query(None, description="можно несколько: ?status=new&status=unsure"), camera: list[str] | None = Query(None), district: list[str] | None = Query(None),
                city: str | None = None, min_conf: float | None = Query(None, ge=0, le=1), since: datetime | None = None, until: datetime | None = None, demo: bool | None = None,
                state: str | None = Query(None, pattern="^(open|closed)$"), oldest_first: bool = False, limit: int = Query(50, ge=1, le=AS.MAX_PAGE), offset: int = Query(0, ge=0),
                _: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> Page[AlertOut]:
    rows, total = AS.list_page(s, limit=limit, offset=offset, newest_first=not oldest_first, status=status, cameras=camera, districts=district, city=city, min_conf=min_conf, since=since,
                               until=until, demo=demo, state=state)
    return Page(items=[to_out(a) for a in rows], total=total, limit=limit, offset=offset)


@router.get("/alerts/{alert_id}", response_model=AlertOut, summary="Карточка тревоги")
def get_alert(alert_id: int, _: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> AlertOut:
    return to_out(_get(s, alert_id))


@router.post("/alerts/{alert_id}/review", response_model=AlertOut, summary="Решение оператора (подтвердить / ложная / не уверен / вернуть в очередь)")
def review_alert(alert_id: int, body: ReviewIn, request: Request, me: User = Depends(deps.operator), s: Session = Depends(deps.get_session)) -> AlertOut:
    a = _get(s, alert_id)
    AS.review(s, a, body.status, me, note=body.note, reason=body.reason)
    deps.audit(s, request, me, "alert.review", "alert", alert_id, status=body.status, reason=body.reason)
    return to_out(a)


@router.post("/alerts/review", response_model=BulkReviewOut, summary="Одно решение для нескольких тревог")
def bulk_review(body: BulkReviewIn, request: Request, me: User = Depends(deps.operator), s: Session = Depends(deps.get_session)) -> BulkReviewOut:
    done, missing = AS.bulk_review(s, body.ids, body.status, me, note=body.note, reason=body.reason)
    deps.audit(s, request, me, "alert.review_bulk", "alert", ",".join(map(str, done[:20])), status=body.status, n=len(done))
    return BulkReviewOut(done=done, missing=missing)


@router.get("/alerts/{alert_id}/reviews", response_model=list[ReviewOut], summary="История решений по тревоге")
def alert_reviews(alert_id: int, _: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> list:
    _get(s, alert_id)
    return AS.history(s, alert_id)


@router.get("/alerts/{alert_id}/{kind}", summary="Миниатюра или клип тревоги (поддерживает Range)", responses={404: {"description": "файла нет"}})
def alert_file(alert_id: int, kind: str, _: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> FileResponse:
    if kind not in ("thumb", "clip"):
        raise HTTPException(404, "kind ∈ thumb, clip")
    p = AF.file_for(_get(s, alert_id), kind)
    if p is None:
        raise HTTPException(404, "файла нет")
    return FileResponse(p, headers={"Cache-Control": "private, max-age=300"})


@router.get("/stats", response_model=StatsOut, summary="KPI дашборда")
def stats(day: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"), _: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> dict:
    return AS.stats(s, day)


@router.get("/reports/summary", summary="Сводка за период: ряд по дням/часам, причины «ложных», камеры, суточный профиль")
def reports_summary(days: int = Query(7, ge=1, le=RS.MAX_DAYS), bucket: str = Query("day", pattern="^(day|hour)$"), camera: str | None = None, district: str | None = None,
                    include_demo: bool = True, _: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> dict:
    return RS.summary(s, days=days, bucket=bucket, camera=camera, district=district, include_demo=include_demo)


@router.get("/review-reasons", summary="Справочник причин для решения «ложная»")
def reasons(_: User = Depends(deps.viewer)) -> dict[str, str]:
    return AS.REASONS


@router.get("/journal.csv", summary="Журнал решений (CSV)", response_class=PlainTextResponse)
def journal_csv(_: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> PlainTextResponse:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "t_abs", "district", "cam_index", "confidence", "status", "reviewer", "reviewed_at", "note", "demo"])
    for a in AS.journal(s):
        w.writerow([a.id, a.t_abs, a.district, a.cam_index, a.confidence, a.status, a.reviewer, a.reviewed_at, a.note, int(a.demo)])
    return PlainTextResponse("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": "attachment; filename=decisions.csv"})


def _iso(o):
    return o.isoformat() if hasattr(o, "isoformat") else str(o)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=_iso)}\n\n"


@router.get("/events", summary="Поток обновлений (Server-Sent Events): новые тревоги и решения", response_class=StreamingResponse)
def events(request: Request, since_alert: int | None = Query(None, ge=0, description="id последней известной тревоги; по умолчанию — только новые"),
           since_review: int | None = Query(None, ge=0), timeout: float = Query(60.0, ge=1, le=3600, description="сколько секунд держать соединение"),
           interval: float = Query(1.0, ge=0.05, le=30), ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)) -> StreamingResponse:
    """Соединение закрывается через `timeout` секунд — клиент (EventSource) переподключается сам и передаёт курсоры; между событиями каждые ~15 с приходит комментарий-«пульс»."""
    with ctx.db.session() as s:
        a0, r0 = AS.cursors(s)
    last_a = a0 if since_alert is None else since_alert
    last_r = r0 if since_review is None else since_review

    def gen() -> Iterator[str]:
        nonlocal last_a, last_r
        end, beat = time.monotonic() + timeout, time.monotonic()
        yield _sse("ready", dict(alert=last_a, review=last_r))
        while time.monotonic() < end:
            with ctx.db.session() as s:
                new, rev = AS.events_after(s, last_a, last_r)
                payload = [("alert.new", dict(id=a.id, camera_id=a.camera_id, confidence=a.confidence, t_abs=a.t_abs, status=a.status)) for a in new] + \
                          [("alert.review", dict(id=r.id, alert_id=r.alert_id, status=r.status, reviewer=r.reviewer_name, reason=r.reason)) for r in rev]
                if new:
                    last_a = new[-1].id
                if rev:
                    last_r = rev[-1].id
            for name, d in payload:
                yield _sse(name, {**d, "cursor": dict(alert=last_a, review=last_r)})
            if payload:
                beat = time.monotonic()
            elif time.monotonic() - beat > 15:
                yield ": ping\n\n"
                beat = time.monotonic()
            time.sleep(interval)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.post("/demo", summary="Демонстрационные тревоги (помечены demo) или их удаление", tags=["тревоги", "администрирование"])
def demo(body: DemoIn, request: Request, ctx: deps.Ctx = Depends(deps.get_ctx), me: User = Depends(deps.admin), s: Session = Depends(deps.get_session)) -> dict:
    from ...realtime.demo import seed_demo

    store = ctx.legacy.store
    n = store.delete_demo() if body.clear else seed_demo(store, body.n)
    deps.audit(s, request, me, "demo.clear" if body.clear else "demo.seed", "alert", "", n=n)
    return dict(n=n)


__all__ = ["router", "ALERT_STATUSES"]
