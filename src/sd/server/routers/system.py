"""Состояние сервиса: проверка живости и версия схемы БД (без входа — для балансировщика и мониторинга)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text

from .. import __version__
from .. import migrate as M
from ..deps import Ctx, get_ctx
from ..schemas import HealthOut

router = APIRouter(tags=["система"])


@router.get("/health", response_model=HealthOut, summary="Проверка живости")
def health(ctx: Ctx = Depends(get_ctx)) -> HealthOut:
    db_ok, rev = True, []
    try:
        with ctx.db.engine.connect() as c:
            c.execute(text("SELECT 1"))
        rev = M.current(ctx.db.url)
    except Exception:      # noqa: BLE001 — любая ошибка соединения = БД недоступна
        db_ok = False
    return HealthOut(ok=db_ok, version=__version__, db=db_ok, schema_current=db_ok and sorted(rev) == sorted(M.heads()), schema_revision=rev, auth_required=ctx.settings.auth_required)
