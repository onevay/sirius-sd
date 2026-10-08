"""HTTP-сервер приложения: JSON API + статические файлы + видео с поддержкой Range (перемотка). Только стандартная библиотека.

Маршруты (все GET, если не указано иное):
  /api/health  /api/stats  /api/cameras  /api/camera/<id>  /api/folders  /api/videos?folder=  /api/profiles
  /api/alerts?status=&camera=&district=&min_conf=&limit=   /api/alert/<id>   POST /api/alert/<id>/review {status, reviewer, note}
  /media/video/<fid>   /media/alert/<id>/thumb|clip   /api/media-status?id=<fid>
  POST /api/analyze {video, profile}   /api/job/<id>   /api/analysis?video=<fid>[&profile=]
  /api/journal.csv   POST /api/demo {clear}
"""
from __future__ import annotations

import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import media
from .api import NotFound, WebApp

STATIC = Path(__file__).parent / "static"


def make_handler(app: WebApp):
    class H(BaseHTTPRequestHandler):
        server_version = "sd-web"

        def log_message(self, *a) -> None:      # тишина в консоли
            pass

        # ------------------------------------------------------------------ ответы
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

        def _file(self, path: Path, ctype: str | None = None) -> None:
            size = path.stat().st_size
            ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end, code = 0, size - 1, 200
            if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
                a, b = m.groups()
                if a == "" and b:
                    start = max(0, size - int(b))
                else:
                    start = int(a or 0)
                    end = int(b) if b else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
                code = 206
            n = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(n))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    left = n
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ------------------------------------------------------------------ маршрутизация
        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            p = unquote(u.path)
            try:
                if p.startswith("/api/") or p.startswith("/media/"):
                    return self._api_get(p, q)
                name = "index.html" if p in ("/", "") else p.lstrip("/")
                f = (STATIC / name).resolve()
                f.relative_to(STATIC.resolve())
                if not f.is_file():
                    f = STATIC / "index.html"
                return self._file(f)
            except NotFound as e:
                self._json(dict(error=str(e)), 404)
            except (ValueError, KeyError) as e:
                self._json(dict(error=str(e)), 400)
            except Exception as e:
                self._json(dict(error=f"{type(e).__name__}: {e}"), 500)

        def _api_get(self, p: str, q: dict) -> None:
            if p == "/api/health":
                return self._json(dict(ok=True, roots={k: str(v) for k, v in app.roots.items()}))
            if p == "/api/stats":
                return self._json(app.stats())
            if p == "/api/cameras":
                return self._json(app.cameras())
            if m := re.fullmatch(r"/api/camera/(.+)", p):
                return self._json(app.camera(m.group(1)))
            if p == "/api/folders":
                return self._json(app.folders())
            if p == "/api/videos":
                return self._json(app.videos(q["folder"]))
            if p == "/api/folder-of":
                return self._json(dict(folder=app.folder_of(q["video"])))
            if p == "/api/profiles":
                return self._json(app.profiles())
            if p == "/api/alerts":
                st = [s for s in q.get("status", "").split(",") if s]
                return self._json(app.alerts(st, q.get("camera"), q.get("district"), float(q["min_conf"]) if q.get("min_conf") else None, int(q.get("limit", 100)), q.get("since")))
            if m := re.fullmatch(r"/api/alert/(\d+)", p):
                return self._json(app.alert(int(m.group(1))))
            if p == "/api/journal.csv":
                return self._send(200, ("﻿" + app.journal_csv()).encode("utf-8"), "text/csv; charset=utf-8", {"Content-Disposition": "attachment; filename=decisions.csv"})
            if m := re.fullmatch(r"/api/job/(\w+)", p):
                return self._json(app.job(m.group(1)).to_dict())
            if p == "/api/analysis":
                r = app.latest_analysis(q["video"], q.get("profile"))
                return self._json(r or dict(trace=None, meta=None))
            if p == "/api/media-status":
                state, _, err = app.media_for(q["id"])
                return self._json(dict(state=state, error=err))
            if m := re.fullmatch(r"/media/video/(.+)", p):
                state, path, err = app.media_for(m.group(1))
                if path is None:
                    return self._json(dict(state=state, error=err), 202 if state == "preparing" else 500)
                return self._file(path, "video/mp4" if path.suffix.lower() in (".mp4", ".m4v") else None)
            if m := re.fullmatch(r"/media/alert/(\d+)/(thumb|clip)", p):
                return self._file(app.artifact(int(m.group(1)), m.group(2)))
            raise NotFound(p)

        def do_POST(self) -> None:
            u = urlparse(self.path)
            p = unquote(u.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if m := re.fullmatch(r"/api/alert/(\d+)/review", p):
                    st = body.get("status")
                    if st not in ("new", "confirmed", "false", "unsure"):
                        raise ValueError("status ∈ new, confirmed, false, unsure")
                    return self._json(app.review(int(m.group(1)), st, str(body.get("reviewer", ""))[:60], str(body.get("note", ""))[:500]))
                if p == "/api/analyze":
                    return self._json(app.submit(body["video"], body["profile"]).to_dict(), 202)
                if p == "/api/demo":
                    return self._json(dict(n=app.demo(bool(body.get("clear")))))
                raise NotFound(p)
            except NotFound as e:
                self._json(dict(error=str(e)), 404)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self._json(dict(error=f"неверный запрос: {e}"), 400)
            except Exception as e:
                self._json(dict(error=f"{type(e).__name__}: {e}"), 500)

    return H


def serve(app: WebApp | None = None, host: str = "127.0.0.1", port: int = 8502, block: bool = True) -> ThreadingHTTPServer:
    app = app or WebApp()
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    if block:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()
    else:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
