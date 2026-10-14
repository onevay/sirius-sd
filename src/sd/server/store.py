"""`SqlAlertStore` — хранилище мониторинга на SQLAlchemy с тем же интерфейсом, что у `realtime.store.AlertStore` (SQLite без ORM).

Воркер (`sd monitor`), демо-данные, обратная связь и прежний `WebApp` принимают любой объект с этим интерфейсом, поэтому переход на новую БД — замена одной строки:
`store = open_store()` (см. `realtime.store.open_store`: при заданном `SD_DB_URL` вернёт этот класс, иначе прежний). Схему создают и обновляют миграции — конструктор сам доводит БД
до актуальной версии (`ensure_schema`).

Отличия от прежнего хранилища: типизированные времена (наружу отдаются теми же строками `YYYY-MM-DDTHH:MM:SS`), история решений (`alert_reviews`), камеры с городом и координатами.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from alembic import command
from alembic.runtime.migration import MigrationContext
from sqlalchemy import func, select

from . import clock
from . import migrate as M
from .db import Database
from .models import ALERT_STATUSES, Alert, Camera, Chunk
from .services import alerts as AS
from .settings import Settings, sqlite_url

STATUSES = ALERT_STATUSES
ALERT_COLUMNS = ("id", "key", "camera_id", "district", "cam_index", "chunk", "chunk_offset", "start_sec", "end_sec", "peak_sec", "t_abs", "t_raised", "confidence", "tid", "box", "rule",
                 "explain", "n_cycles", "state", "status", "reviewer", "reviewed_at", "note", "thumb", "clip", "demo", "fingerprint", "created_at", "updated_at")


def ensure_schema(db: Database) -> None:
    """Применяет недостающие миграции (безопасно вызывать при каждом старте; для БД на актуальной версии ничего не делает)."""
    with db.engine.begin() as conn:
        if sorted(MigrationContext.configure(conn).get_current_heads()) == sorted(M.heads()):
            return
        command.upgrade(M.config(db.url, conn), "head")


def alert_dict(a: Alert) -> dict:
    """Тревога в формате прежнего хранилища: времена — строки, `demo` — 0/1, `box` — список."""
    d: dict[str, Any] = {c: getattr(a, c) for c in ALERT_COLUMNS}
    for k in ("t_abs", "t_raised", "reviewed_at", "created_at", "updated_at"):
        d[k] = clock.fmt(d[k])
    d["demo"] = int(bool(a.demo))
    return d


class SqlAlertStore:
    def __init__(self, target: Database | str | Path | None = None, *, migrate: bool = True):
        if isinstance(target, Database):
            self.db = target
        else:
            url = str(target) if isinstance(target, str) and "://" in str(target) else sqlite_url(target) if target else Settings.from_env().db_url
            self.db = Database(url)
        self.path = Path(self.db.engine.url.database) if self.db.engine.url.get_backend_name() == "sqlite" and self.db.engine.url.database else None
        if migrate:
            ensure_schema(self.db)

    # ------------------------------------------------------------------ камеры и фрагменты
    def upsert_camera(self, camera_id: str, district: str, cam_index: str, folder: str, start_at: str, status: str = "idle") -> None:
        with self.db.session() as s:
            cam = s.get(Camera, camera_id)
            if cam is None:
                cam = Camera(camera_id=camera_id, city="", title="", street="", status=status, enabled=True, processed_sec=0.0, created_at=clock.now())
                s.add(cam)
            cam.district, cam.cam_index, cam.folder, cam.start_at = district, cam_index, folder, clock.parse(start_at)
            cam.last_seen = clock.now()

    def touch_camera(self, camera_id: str, status: str, processed_sec: float | None = None) -> None:
        with self.db.session() as s:
            cam = s.get(Camera, camera_id)
            if cam is None:
                return
            cam.status, cam.last_seen = status, clock.now()
            if processed_sec is not None:
                cam.processed_sec = processed_sec

    def chunks_done(self, camera_id: str) -> dict[str, tuple[float, float]]:
        with self.db.session() as s:
            return {c.name: (c.duration, c.offset_sec) for c in s.scalars(select(Chunk).where(Chunk.camera_id == camera_id).order_by(Chunk.offset_sec))}

    def mark_chunk(self, camera_id: str, name: str, duration: float, offset: float) -> None:
        with self.db.session() as s:
            row = s.get(Chunk, (camera_id, name))
            if row is None:
                s.add(Chunk(camera_id=camera_id, name=name, duration=duration, offset_sec=offset, done_at=clock.now()))
            else:
                row.duration, row.offset_sec, row.done_at = duration, offset, clock.now()

    # ------------------------------------------------------------------ тревоги
    def apply_update(self, u, meta: dict) -> tuple[int, str]:
        """Применяет `AlertUpdate` движка (open / update / close). Решение оператора при продлении события сохраняется. Возвращает (id, kind)."""
        now = clock.now()
        with self.db.session() as s:
            a = s.scalar(select(Alert).where(Alert.key == u.key))
            if a is None:
                if u.kind == "close":
                    return -1, "close"
                a = Alert(key=u.key, camera_id=u.camera_id, district=meta.get("district"), cam_index=meta.get("cam_index"), chunk=meta.get("chunk"), chunk_offset=meta.get("chunk_offset", 0.0),
                          start_sec=u.start, end_sec=u.end, peak_sec=u.peak, t_abs=clock.parse(meta.get("t_abs")), t_raised=now, confidence=u.confidence, tid=u.tid,
                          box=list(u.box) if u.box else None, rule=u.rule, explain=u.explain, n_cycles=u.n_cycles, state="open", status="new", fingerprint=meta.get("fingerprint"),
                          demo=bool(meta.get("demo", 0)), created_at=now, updated_at=now)
                s.add(a)
                s.flush()
                return int(a.id), "open"
            if u.kind == "close":
                a.state = "closed"
            else:
                a.end_sec, a.peak_sec, a.confidence, a.n_cycles, a.explain = u.end, u.peak, u.confidence, u.n_cycles, u.explain
                if u.box:
                    a.box = list(u.box)
            a.updated_at = now
            return int(a.id), u.kind

    def set_artifacts(self, alert_id: int, thumb: str | None = None, clip: str | None = None) -> None:
        with self.db.session() as s:
            a = s.get(Alert, alert_id)
            if a is not None:
                a.thumb, a.clip = thumb or a.thumb, clip or a.clip

    def review(self, alert_id: int, status: str, reviewer: str = "", note: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"status ∈ {STATUSES}")
        with self.db.session() as s:
            a = s.get(Alert, alert_id)
            if a is not None:
                AS.review(s, a, status, None, note=note, reviewer_name=reviewer)

    def get(self, alert_id: int) -> dict | None:
        with self.db.session() as s:
            a = s.get(Alert, alert_id)
            return alert_dict(a) if a else None

    def list_alerts(self, *, status: list[str] | None = None, districts: list[str] | None = None, cameras: list[str] | None = None, since: str | None = None, until: str | None = None,
                    min_conf: float | None = None, demo: bool | None = None, limit: int = 300, newest_first: bool = True) -> pd.DataFrame:
        """Выборка для воркера, выгрузок и прежнего веб-приложения; размер не ограничен страницей API (журнал и экспорт в эталон читают десятки тысяч строк)."""
        q = AS.filtered(select(Alert), status=status, districts=districts, cameras=cameras, since=clock.parse(since), until=clock.parse(until), min_conf=min_conf, demo=demo)
        order = (Alert.t_abs.desc(), Alert.id.desc()) if newest_first else (Alert.t_abs.asc(), Alert.id.asc())
        with self.db.session() as s:
            return pd.DataFrame([alert_dict(a) for a in s.scalars(q.order_by(*order).limit(limit))])

    def max_id(self) -> int:
        with self.db.session() as s:
            return int(s.scalar(select(func.coalesce(func.max(Alert.id), 0))) or 0)

    def cameras(self) -> pd.DataFrame:
        from .services import cameras as CS

        with self.db.session() as s:
            rows = []
            for r in CS.list_with_metrics(s):
                c = r["camera"]
                rows.append(dict(camera_id=c.camera_id, district=c.district, cam_index=c.cam_index, folder=c.folder, start_at=clock.fmt(c.start_at), status=c.status, last_seen=clock.fmt(c.last_seen),
                                 processed_sec=c.processed_sec, city=c.city, alerts=r["alerts"], pending=r["pending"]))
            return pd.DataFrame(rows)

    def stats(self, day: str | None = None) -> dict:
        with self.db.session() as s:
            st = AS.stats(s, day)
        st.pop("max_id", None)
        return st

    def delete_demo(self) -> int:
        with self.db.session() as s:
            demo = list(s.scalars(select(Alert).where(Alert.demo.is_(True))))
            for a in demo:
                s.delete(a)
            for c in s.scalars(select(Camera).where(Camera.folder.like("demo:%"))):
                s.delete(c)
            return len(demo)

