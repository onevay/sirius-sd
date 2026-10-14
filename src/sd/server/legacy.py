"""Перенос данных из прежней SQLite-базы мониторинга (`outputs/monitor/monitor.db`, `realtime.store.AlertStore`) в БД платформы.

Идентификаторы тревог сохраняются (на них ссылаются каталоги миниатюр и клипов `outputs/monitor/alerts/<id>`); по ключу тревоги повторный перенос ничего не дублирует.
Решённые тревоги получают первую строку в истории решений (кто и когда решил — по данным прежней базы).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sqlalchemy import select, text

from . import clock
from .db import Database
from .models import ALERT_STATUSES, CAMERA_STATUSES, Alert, AlertReview, Camera, Chunk


def _rows(con: sqlite3.Connection, table: str) -> list[dict]:
    try:
        return [dict(r) for r in con.execute(f"SELECT * FROM {table}")]
    except sqlite3.OperationalError:      # такой таблицы в старой базе нет
        return []


def _dt(v):
    try:
        return clock.parse(v)
    except (ValueError, TypeError):
        return None


def import_legacy(db: Database, path: str | Path) -> dict[str, int]:
    """Переносит камеры, фрагменты и тревоги. Возвращает числа перенесённых и пропущенных (уже есть) записей."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"база не найдена: {p}")
    con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = dict(cameras=0, chunks=0, alerts=0, reviews=0, skipped=0)
    try:
        cams, chunks, alerts = _rows(con, "cameras"), _rows(con, "chunks"), _rows(con, "alerts")
    finally:
        con.close()
    now = clock.now()
    with db.session() as s:
        have_cam = set(s.scalars(select(Camera.camera_id)))
        for c in cams:
            if c["camera_id"] in have_cam:
                out["skipped"] += 1
                continue
            s.add(Camera(camera_id=c["camera_id"], city="", district=c.get("district"), cam_index=c.get("cam_index"), title="", street="", folder=c.get("folder"), start_at=_dt(c.get("start_at")),
                         status=c.get("status") if c.get("status") in CAMERA_STATUSES else "idle", enabled=True, last_seen=_dt(c.get("last_seen")), processed_sec=float(c.get("processed_sec") or 0),
                         created_at=now))
            have_cam.add(c["camera_id"])
            out["cameras"] += 1
        s.flush()
        for ch in chunks:
            if ch["camera_id"] in have_cam and s.get(Chunk, (ch["camera_id"], ch["name"])) is None:
                s.add(Chunk(camera_id=ch["camera_id"], name=ch["name"], duration=float(ch["duration"] or 0), offset_sec=float(ch["offset_sec"] or 0), done_at=_dt(ch.get("done_at")) or now))
                out["chunks"] += 1
        have_key = set(s.scalars(select(Alert.key)))
        for a in alerts:
            if a["key"] in have_key:
                out["skipped"] += 1
                continue
            status = a.get("status") if a.get("status") in ALERT_STATUSES else "new"
            created = _dt(a.get("created_at")) or now
            try:
                box = json.loads(a["box"]) if a.get("box") else None
            except (TypeError, json.JSONDecodeError):
                box = None
            row = Alert(id=a["id"], key=a["key"], camera_id=a["camera_id"], district=a.get("district"), cam_index=a.get("cam_index"), chunk=a.get("chunk"), chunk_offset=a.get("chunk_offset") or 0.0,
                        start_sec=a["start_sec"], end_sec=a["end_sec"], peak_sec=a["peak_sec"], t_abs=_dt(a.get("t_abs")), t_raised=_dt(a.get("t_raised")), confidence=a["confidence"], tid=a.get("tid"),
                        box=box, rule=a.get("rule"), explain=a.get("explain"), n_cycles=a.get("n_cycles"), state="closed" if a.get("state") == "closed" else "open", status=status,
                        reviewer=a.get("reviewer"), reviewed_at=_dt(a.get("reviewed_at")), note=a.get("note"), thumb=a.get("thumb"), clip=a.get("clip"), demo=bool(a.get("demo")),
                        fingerprint=a.get("fingerprint"), created_at=created, updated_at=_dt(a.get("updated_at")) or created)
            s.add(row)
            out["alerts"] += 1
            if status != "new":
                s.add(AlertReview(alert=row, status=status, note=a.get("note"), reviewer_name=a.get("reviewer") or "", created_at=_dt(a.get("reviewed_at")) or created))
                out["reviews"] += 1
        s.flush()
        if db.engine.dialect.name == "postgresql":      # после вставки с явными id счётчик последовательности надо подвинуть
            s.execute(text("SELECT setval(pg_get_serial_sequence('alerts', 'id'), COALESCE((SELECT MAX(id) FROM alerts), 1))"))
    return out
