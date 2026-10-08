"""Хранилище мониторинга (SQLite, один файл `outputs/monitor/monitor.db`): камеры, обработанные фрагменты, тревоги и решения оператора.

Пишет воркер, читает приложение оператора — разные процессы, поэтому WAL-режим и короткие соединения. Времена — локальные, ISO-строки `YYYY-MM-DDTHH:MM:SS`.
Статус тревоги: `new` (ждёт оператора) → `confirmed` | `false` | `unsure`. Продление события воркером не сбрасывает решение оператора.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from ..paths import OUTPUTS

DB_PATH = OUTPUTS / "monitor" / "monitor.db"
STATUSES = ("new", "confirmed", "false", "unsure")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras(camera_id TEXT PRIMARY KEY, district TEXT, cam_index TEXT, folder TEXT, start_at TEXT, status TEXT, last_seen TEXT, processed_sec REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS chunks(camera_id TEXT, name TEXT, duration REAL, offset_sec REAL, done_at TEXT, PRIMARY KEY(camera_id, name));
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, camera_id TEXT, district TEXT, cam_index TEXT, chunk TEXT, chunk_offset REAL,
  start_sec REAL, end_sec REAL, peak_sec REAL, t_abs TEXT, t_raised TEXT, confidence REAL, tid INTEGER, box TEXT, rule TEXT, explain TEXT, n_cycles INTEGER,
  state TEXT DEFAULT 'open', status TEXT DEFAULT 'new', reviewer TEXT, reviewed_at TEXT, note TEXT, thumb TEXT, clip TEXT, demo INTEGER DEFAULT 0, fingerprint TEXT,
  created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts(status, t_abs);
CREATE INDEX IF NOT EXISTS ix_alerts_cam ON alerts(camera_id, t_abs);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class AlertStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path(os.environ.get("SD_MONITOR_DB") or DB_PATH)   # SD_MONITOR_DB — база в другом месте (общий диск, тесты)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _c(self):
        con = sqlite3.connect(str(self.path), timeout=30)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ камеры и фрагменты
    def upsert_camera(self, camera_id: str, district: str, cam_index: str, folder: str, start_at: str, status: str = "idle") -> None:
        with self._c() as c:
            c.execute("INSERT INTO cameras(camera_id,district,cam_index,folder,start_at,status,last_seen) VALUES(?,?,?,?,?,?,?) "
                      "ON CONFLICT(camera_id) DO UPDATE SET district=excluded.district, cam_index=excluded.cam_index, folder=excluded.folder, start_at=excluded.start_at",
                      (camera_id, district, cam_index, folder, start_at, status, now()))

    def touch_camera(self, camera_id: str, status: str, processed_sec: float | None = None) -> None:
        with self._c() as c:
            c.execute("UPDATE cameras SET status=?, last_seen=?, processed_sec=COALESCE(?, processed_sec) WHERE camera_id=?", (status, now(), processed_sec, camera_id))

    def chunks_done(self, camera_id: str) -> dict[str, tuple[float, float]]:
        """{имя фрагмента: (длительность, смещение от начала потока)}."""
        with self._c() as c:
            return {r["name"]: (r["duration"], r["offset_sec"]) for r in c.execute("SELECT * FROM chunks WHERE camera_id=? ORDER BY offset_sec", (camera_id,))}

    def mark_chunk(self, camera_id: str, name: str, duration: float, offset: float) -> None:
        with self._c() as c:
            c.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?)", (camera_id, name, duration, offset, now()))

    # ------------------------------------------------------------------ тревоги
    def apply_update(self, u, meta: dict) -> tuple[int, str]:
        """Применяет `AlertUpdate` движка. `meta`: district, cam_index, chunk, chunk_offset, t_abs, fingerprint, demo. Возвращает (id, 'open' | 'update' | 'close')."""
        with self._c() as c:
            row = c.execute("SELECT id FROM alerts WHERE key=?", (u.key,)).fetchone()
            if row is None:
                if u.kind == "close":
                    return -1, "close"
                cur = c.execute(
                    "INSERT INTO alerts(key,camera_id,district,cam_index,chunk,chunk_offset,start_sec,end_sec,peak_sec,t_abs,t_raised,confidence,tid,box,rule,explain,n_cycles,fingerprint,demo,created_at,updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (u.key, u.camera_id, meta.get("district"), meta.get("cam_index"), meta.get("chunk"), meta.get("chunk_offset", 0.0), u.start, u.end, u.peak, meta.get("t_abs"), now(),
                     u.confidence, u.tid, json.dumps(list(u.box) if u.box else None), u.rule, u.explain, u.n_cycles, meta.get("fingerprint"), int(meta.get("demo", 0)), now(), now()))
                return int(cur.lastrowid), "open"
            if u.kind == "close":
                c.execute("UPDATE alerts SET state='closed', updated_at=? WHERE id=?", (now(), row["id"]))
            else:
                c.execute("UPDATE alerts SET end_sec=?, peak_sec=?, confidence=?, n_cycles=?, explain=?, box=COALESCE(?, box), updated_at=? WHERE id=?",
                          (u.end, u.peak, u.confidence, u.n_cycles, u.explain, json.dumps(list(u.box)) if u.box else None, now(), row["id"]))
            return int(row["id"]), u.kind

    def set_artifacts(self, alert_id: int, thumb: str | None = None, clip: str | None = None) -> None:
        with self._c() as c:
            c.execute("UPDATE alerts SET thumb=COALESCE(?, thumb), clip=COALESCE(?, clip) WHERE id=?", (thumb, clip, alert_id))

    def review(self, alert_id: int, status: str, reviewer: str = "", note: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"status ∈ {STATUSES}")
        with self._c() as c:
            c.execute("UPDATE alerts SET status=?, reviewer=?, reviewed_at=?, note=?, updated_at=? WHERE id=?", (status, reviewer, None if status == "new" else now(), note, now(), alert_id))

    def get(self, alert_id: int) -> dict | None:
        with self._c() as c:
            r = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return self._row(r) if r else None

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["box"] = json.loads(d["box"]) if d.get("box") else None
        return d

    def list_alerts(self, *, status: list[str] | None = None, districts: list[str] | None = None, cameras: list[str] | None = None, since: str | None = None, until: str | None = None,
                    min_conf: float | None = None, demo: bool | None = None, limit: int = 300, newest_first: bool = True) -> pd.DataFrame:
        q, a = "SELECT * FROM alerts WHERE 1=1", []
        for col, vals in (("status", status), ("district", districts), ("camera_id", cameras)):
            if vals:
                q += f" AND {col} IN ({','.join('?' * len(vals))})"
                a += list(vals)
        if since:
            q, a = q + " AND t_abs>=?", a + [since]
        if until:
            q, a = q + " AND t_abs<=?", a + [until]
        if min_conf is not None:
            q, a = q + " AND confidence>=?", a + [min_conf]
        if demo is not None:
            q, a = q + " AND demo=?", a + [int(demo)]
        q += f" ORDER BY t_abs {'DESC' if newest_first else 'ASC'}, id DESC LIMIT ?"
        with self._c() as c:
            rows = [self._row(r) for r in c.execute(q, a + [limit])]
        return pd.DataFrame(rows)

    def max_id(self) -> int:
        with self._c() as c:
            return int(c.execute("SELECT COALESCE(MAX(id),0) FROM alerts").fetchone()[0])

    def cameras(self) -> pd.DataFrame:
        with self._c() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM alerts a WHERE a.camera_id=c.camera_id) AS alerts, "
                "(SELECT COUNT(*) FROM alerts a WHERE a.camera_id=c.camera_id AND a.status='new') AS pending FROM cameras c ORDER BY district, cam_index")]
        return pd.DataFrame(rows)

    def stats(self, day: str | None = None) -> dict:
        """KPI дашборда. `day` — 'YYYY-MM-DD' (по умолчанию сегодня)."""
        day = day or time.strftime("%Y-%m-%d")
        with self._c() as c:
            by = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM alerts GROUP BY status")}
            today = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM alerts WHERE t_abs LIKE ? GROUP BY status", (day + "%",))}
            lat = [r[0] for r in c.execute("SELECT (julianday(reviewed_at)-julianday(t_raised))*86400 FROM alerts WHERE reviewed_at IS NOT NULL AND demo=0")]
            cams = c.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        decided = by.get("confirmed", 0) + by.get("false", 0)
        return dict(new=by.get("new", 0), confirmed=by.get("confirmed", 0), false=by.get("false", 0), unsure=by.get("unsure", 0), total=sum(by.values()), today=today, cameras=cams,
                    precision=(by.get("confirmed", 0) / decided) if decided else None, median_review_sec=float(pd.Series(lat).median()) if lat else None)

    def delete_demo(self) -> int:
        with self._c() as c:
            n = c.execute("DELETE FROM alerts WHERE demo=1").rowcount
            c.execute("DELETE FROM cameras WHERE camera_id IN (SELECT camera_id FROM cameras WHERE folder LIKE 'demo:%')")
        return n
