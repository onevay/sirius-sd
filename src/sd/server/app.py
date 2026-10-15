"""Сборка приложения FastAPI: настройки, БД с миграциями, маршруты `/api/v1`, обработка ошибок, CORS, заголовки безопасности.

    app = create_app()                     # настройки из окружения; запуск:  uvicorn --factory sd.server.app:create_app   или   sd serve
    app = create_app(settings, db=db)      # свои настройки и БД (тесты, встраивание)

Схему БД приложение само доводит до актуальной версии при старте (`ensure_schema`): первый запуск создаёт таблицы, обновление кода применяет новые миграции.
"""
from __future__ import annotations

import logging
import mimetypes
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import API_PREFIX, __version__
from .db import Database
from .deps import Ctx
from .routers import alerts, analysis, auth, cameras, dev, system, users

from .services.alerts import AlertError
from .services.cameras import CameraError
from .services.users import UserError
from .settings import Settings
from .store import SqlAlertStore, ensure_schema

log = logging.getLogger("sd.server")

STATIC_DIR = Path(__file__).parent / "static"
UI_PREFIX = "/app"
#: политика для интерфейса: только свои файлы, без внешних запросов; встроенные стили нужны для атрибутов style, которые ставит DOM-построитель
UI_CSP = ("default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self'; "
          "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
# Windows берёт типы из реестра, где .js может оказаться text/plain — браузер тогда отказывается выполнять модули
for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"), (".woff2", "font/woff2"), (".svg", "image/svg+xml"), (".json", "application/json")):
    mimetypes.add_type(_type, _ext)

DESCRIPTION = """
API платформы детекции курения: тревоги и решения операторов, камеры, разбор видео, профили и задачи разработчика.

**Вход.** `POST /api/v1/auth/login` → `access_token`; дальше заголовок `Authorization: Bearer <токен>`. Для скриптов — долгоживущие API-токены (`POST /api/v1/auth/tokens`).
**Роли** (каждая включает предыдущие): `viewer` — просмотр; `operator` — решения по тревогам и разбор видео; `developer` — профили, задачи, журнал прогонов; `admin` — пользователи, камеры, аудит.
**Ошибки** — JSON `{"detail": "..."}`; коды: 401 нет входа, 403 мало прав, 404 нет объекта, 409 конфликт, 422 неверные данные, 429 слишком много попыток входа.
"""

TAGS = [dict(name="система", description="Проверка живости и версия схемы."), dict(name="вход", description="Токены доступа, текущий пользователь, API-токены."),
        dict(name="администрирование", description="Пользователи и журнал действий (admin)."), dict(name="камеры", description="Справочник камер."),
        dict(name="тревоги", description="Тревоги, решения оператора, статистика, поток обновлений."), dict(name="разбор видео", description="Источники, воспроизведение, очередь разбора."),
        dict(name="разработчик", description="Профили, журнал прогонов, задачи, разметка (developer).")]


def _videos_factory(db: Database) -> Callable[[], object]:
    def make():
        from .services.videos import VideoService

        return VideoService(store=SqlAlertStore(db, migrate=False))

    return make


def create_app(settings: Settings | None = None, *, db: Database | None = None, videos_factory: Callable[[], object] | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = db or Database(settings.db_url)
    ctx = Ctx(settings=settings, db=db, videos_factory=videos_factory or _videos_factory(db))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ensure_schema(db)
        if settings.bootstrap_admin:
            from .services import users as US

            with db.session() as s:
                if US.bootstrap_admin(s, *settings.bootstrap_admin, min_password_len=settings.min_password_len):
                    log.warning("создан администратор %s (SD_ADMIN_USER); смените пароль после входа", settings.bootstrap_admin[0])
        if settings.auth_required:
            from .services import users as US

            with db.session() as s:
                if not US.count(s):
                    log.warning("пользователей нет: создайте администратора командой `sd user create <имя> --role admin` или переменными SD_ADMIN_USER / SD_ADMIN_PASSWORD")
        yield
        db.dispose()

    docs = settings.expose_docs
    app = FastAPI(title="Платформа детекции курения — API", version=__version__, description=DESCRIPTION, openapi_tags=TAGS, lifespan=lifespan,
                  docs_url="/docs" if docs else None, redoc_url="/redoc" if docs else None, openapi_url=f"{API_PREFIX}/openapi.json" if docs else None)
    app.state.ctx = ctx

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=True, allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                           allow_headers=["Authorization", "Content-Type"], max_age=600)

    @app.middleware("http")
    async def hardening(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        resp = await call_next(request)
        resp.headers.setdefault("X-Request-ID", rid)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        path = request.url.path
        if path.startswith(API_PREFIX) and "cache-control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-store"
        elif path == UI_PREFIX or path.startswith(UI_PREFIX + "/"):
            resp.headers.setdefault("Content-Security-Policy", UI_CSP)
            resp.headers["Cache-Control"] = "public, max-age=604800" if path.endswith(".woff2") else "no-cache"        # файлы интерфейса проверяются по ETag при каждом открытии
        return resp

    # ---- ошибки прежних модулей и сервисов → JSON с понятным кодом
    def _err(code: int):
        return lambda request, exc: JSONResponse({"detail": str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc)}, status_code=code)

    from .services.videos import NotFound

    for exc, code in ((NotFound, 404), (FileNotFoundError, 404), (PermissionError, 403), (FileExistsError, 409), (UserError, 422), (AlertError, 422), (CameraError, 422), (ValueError, 400),
                      (KeyError, 400)):
        app.add_exception_handler(exc, _err(code))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):      # noqa: ARG001
        log.exception("необработанная ошибка: %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "внутренняя ошибка сервера"}, status_code=500)

    for r in (system.router, auth.router, users.router, cameras.router, alerts.router, analysis.router, dev.router):
        app.include_router(r, prefix=API_PREFIX)

    ui = settings.serve_ui and (STATIC_DIR / "index.html").exists()
    if ui:
        app.mount(UI_PREFIX, StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(f"{UI_PREFIX}/" if ui else "/docs" if docs else f"{API_PREFIX}/health")

    return app


def ensure_ready(settings: Settings | None = None) -> Database:
    """База с актуальной схемой (для CLI и воркера)."""
    settings = settings or Settings.from_env()
    db = Database(settings.db_url)
    ensure_schema(db)
    return db
