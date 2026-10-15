"""Команды платформы: `sd app` (приложение для одного компьютера), `sd serve` (сервер), `sd db …` (миграции и перенос данных), `sd user …` (пользователи и токены)."""
from __future__ import annotations

from .. import _env  # noqa: F401

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

db_app = typer.Typer(no_args_is_help=True, help="Схема БД платформы: миграции Alembic и перенос данных из прежней базы мониторинга.")
user_app = typer.Typer(no_args_is_help=True, help="Пользователи и API-токены платформы.")
console = Console(width=140)
LOCAL = ("127.0.0.1", "localhost", "::1")


def _settings(**kw):
    from .settings import Settings

    return Settings.from_env(**kw)


def _banner(st, host: str, port: int) -> None:
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    ui = f"интерфейс: [bold]http://{shown}:{port}/app/[/] · " if st.serve_ui else ""
    console.print(f"{ui}API: http://{shown}:{port}/docs · БД: {st.db_url.split('@')[-1]} · вход: {'нужен' if st.auth_required else 'отключён (локальный режим)'}  (Ctrl+C — остановить)")


def serve_cmd(host: Annotated[str, typer.Option(help="адрес привязки; для доступа из сети — 0.0.0.0 (вход обязателен)")] = "127.0.0.1", port: int = 8503,
              no_auth: Annotated[bool, typer.Option("--no-auth", help="без входа (все запросы — администратор local); только для локальной привязки")] = False,
              reload: Annotated[bool, typer.Option(help="перезапуск при правках кода (разработка)")] = False, log_level: str = "info") -> None:
    """Запускает сервер платформы — API и веб-интерфейс (FastAPI + uvicorn): интерфейс — http://<адрес>:<порт>/app/, документация API — /docs. Вход обязателен."""
    import uvicorn

    local = host in LOCAL
    if no_auth and not local:
        console.print("[red]--no-auth разрешён только при привязке к 127.0.0.1: вход отключается целиком[/]")
        raise typer.Exit(2)
    st = _settings(**({"auth_required": False} if no_auth else {}), allow_fs=local)
    if not st.auth_required and not local:          # и флаг --no-auth, и SD_AUTH=off
        console.print("[red]вход отключён (SD_AUTH=off): при привязке не к 127.0.0.1 запуск запрещён[/]")
        raise typer.Exit(2)
    _banner(st, host, port)
    if reload:
        import os

        os.environ["SD_AUTH"] = "off" if no_auth else os.environ.get("SD_AUTH", "on")
        uvicorn.run("sd.server.app:create_app", factory=True, host=host, port=port, reload=True, log_level=log_level)
        return
    from .app import create_app

    uvicorn.run(create_app(st), host=host, port=port, log_level=log_level)


def app_cmd(host: Annotated[str, typer.Option(help="адрес привязки; в контейнере и для доступа из сети — 0.0.0.0 (тогда вход включается сам)")] = "127.0.0.1", port: int = 8502,
            login: Annotated[bool, typer.Option("--login/--no-login", help="требовать вход (пользователей создаёт `sd user create`); по умолчанию на этом компьютере вход отключён")] = False,
            open_browser: Annotated[bool, typer.Option("--open/--no-open", help="открыть браузер")] = False, log_level: str = "warning") -> None:
    """Веб-приложение для работы на одном компьютере: интерфейс оператора и разработчика (`/app/`), без входа, видео и папки — прямо с этого компьютера.
    Тот же сервер, что `sd serve`, но с настройками «для себя»: порт 8502, вход отключён. Не на локальном адресе вход включается автоматически."""
    import threading

    import uvicorn

    local = host in LOCAL
    st = _settings(auth_required=bool(login or not local), allow_fs=local)
    if not st.serve_ui:
        console.print("[red]интерфейс отключён переменной SD_UI=0: запустите `sd serve` (только API) или уберите её[/]")
        raise typer.Exit(2)
    _banner(st, host, port)
    if not local and not login:
        console.print("[yellow]доступ из сети: вход включён автоматически, пользователя создаёт `sd user create` или SD_ADMIN_USER / SD_ADMIN_PASSWORD[/]")
    if open_browser:
        import webbrowser

        threading.Timer(1.5, webbrowser.open, [f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/app/"]).start()
    from .app import create_app

    uvicorn.run(create_app(st), host=host, port=port, log_level=log_level)


# ---------------------------------------------------------------------------------------------- sd db
@db_app.command("upgrade", help="Применить миграции (по умолчанию до последней).")
def db_upgrade(revision: Annotated[str, typer.Argument()] = "head") -> None:
    from . import migrate as M

    url = _settings().db_url
    M.upgrade(url, revision)
    console.print(f"схема: {', '.join(M.current(url)) or '—'}")


@db_app.command("downgrade", help="Откатить миграции (по умолчанию на одну назад). `base` — удалить все таблицы платформы.")
def db_downgrade(revision: Annotated[str, typer.Argument()] = "-1", yes: Annotated[bool, typer.Option("--yes", help="не спрашивать подтверждение")] = False) -> None:
    from . import migrate as M

    url = _settings().db_url
    if not yes and not typer.confirm(f"Откат до «{revision}» может удалить данные таблиц. Продолжить?"):
        raise typer.Exit(1)
    M.downgrade(url, revision)
    console.print(f"схема: {', '.join(M.current(url)) or '—'}")


@db_app.command("current", help="Версия схемы в БД и последняя доступная миграция.")
def db_current() -> None:
    from . import migrate as M

    url = _settings().db_url
    cur = M.current(url)
    console.print(f"в БД: {', '.join(cur) or 'миграции не применялись'} · последняя: {', '.join(M.heads())} · {'актуальна' if M.is_current(url) else '[yellow]нужен sd db upgrade[/]'}")


@db_app.command("history", help="Цепочка миграций.")
def db_history() -> None:
    from . import migrate as M

    t = Table("версия", "предыдущая", "описание")
    for r in M.history():
        t.add_row(r["revision"], str(r["down"] or "—"), r["doc"])
    console.print(t)


@db_app.command("revision", help="Создать миграцию: сравнивает БД с models.py (БД должна быть на актуальной версии). Результат проверьте вручную.")
def db_revision(message: Annotated[str, typer.Option("-m", "--message", help="что меняется")], empty: Annotated[bool, typer.Option("--empty", help="пустая заготовка без автогенерации")] = False) -> None:
    from . import migrate as M

    url = _settings().db_url
    if not empty and not M.is_current(url):
        console.print("[red]БД не на последней версии: сначала `sd db upgrade`[/]")
        raise typer.Exit(2)
    f = M.new_revision(url, message, autogenerate=not empty)
    console.print(f"создано: {f}\nпроверьте файл (особенно типы и имена ограничений) и примените: sd db upgrade")


@db_app.command("stamp", help="Пометить БД версией без выполнения миграций (БД создана вручную или из бэкапа).")
def db_stamp(revision: Annotated[str, typer.Argument()] = "head") -> None:
    from . import migrate as M

    M.stamp(_settings().db_url, revision)


@db_app.command("import-legacy", help="Перенести камеры, тревоги и решения из прежней SQLite-базы мониторинга (по умолчанию outputs/monitor/monitor.db).")
def db_import_legacy(path: Annotated[Optional[Path], typer.Argument(help="файл monitor.db")] = None) -> None:
    from .app import ensure_ready
    from .legacy import LEGACY_DB, import_legacy

    src = path or LEGACY_DB
    db = ensure_ready(_settings())
    try:
        console.print(import_legacy(db, src))
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    finally:
        db.dispose()


# ---------------------------------------------------------------------------------------------- sd user
def _db():
    from .app import ensure_ready

    return ensure_ready(_settings())


@user_app.command("create", help="Создать пользователя. Пароль спрашивается без отображения (или --password).")
def user_create(username: str, role: Annotated[str, typer.Option(help="viewer | operator | developer | admin")] = "viewer", display_name: str = "",
                password: Annotated[Optional[str], typer.Option(help="не рекомендуется: останется в истории оболочки")] = None) -> None:
    from .services import users as US

    pw = password or typer.prompt("Пароль", hide_input=True, confirmation_prompt=True)
    db = _db()
    try:
        with db.session() as s:
            u = US.create(s, username, pw, role, display_name, min_password_len=_settings().min_password_len)
            console.print(f"создан: {u.username} ({u.role})")
    except US.UserError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    finally:
        db.dispose()


@user_app.command("list", help="Список пользователей.")
def user_list() -> None:
    from .services import users as US

    db = _db()
    try:
        with db.session() as s:
            t = Table("id", "имя", "роль", "активен", "последний вход")
            for u in US.list_users(s):
                t.add_row(str(u.id), u.username, u.role, "да" if u.is_active else "нет", str(u.last_login_at or "—"))
            console.print(t)
    finally:
        db.dispose()


@user_app.command("set", help="Изменить роль, блокировку или пароль пользователя (смена пароля и блокировка отзывают его токены).")
def user_set(username: str, role: Optional[str] = None, active: Annotated[Optional[bool], typer.Option("--active/--inactive")] = None,
             password: Annotated[bool, typer.Option("--password", help="спросить новый пароль")] = False) -> None:
    from .services import users as US

    pw = typer.prompt("Новый пароль", hide_input=True, confirmation_prompt=True) if password else None
    db = _db()
    try:
        with db.session() as s:
            u = US.get_by_username(s, username)
            if u is None:
                console.print("[red]пользователь не найден[/]")
                raise typer.Exit(2)
            US.update(s, u, role=role, is_active=active, password=pw, min_password_len=_settings().min_password_len)
            console.print(f"ок: {u.username} · {u.role} · {'активен' if u.is_active else 'заблокирован'}")
    except US.UserError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from None
    finally:
        db.dispose()


@user_app.command("token", help="Создать API-токен пользователя (показывается один раз).")
def user_token(username: str, name: str = "cli", ttl_days: Annotated[Optional[int], typer.Option(help="срок жизни, дней; пусто — бессрочный")] = None) -> None:
    from .services import users as US

    db = _db()
    try:
        with db.session() as s:
            u = US.get_by_username(s, username)
            if u is None:
                console.print("[red]пользователь не найден[/]")
                raise typer.Exit(2)
            _, tok = US.create_api_token(s, u, name, ttl_days)
            console.print(tok)
    finally:
        db.dispose()
