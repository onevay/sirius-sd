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
import os
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import shutil
import zipfile

from . import admin, media
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
                if p.startswith(("/api/", "/media/", "/export/")):
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
                return self._json(app.cameras(float(q.get("hours", 0) or 0)))
            if p == "/api/tune/table":
                return self._json(app.tuning.table_info(q.get("enriched", "1") != "0", q.get("refresh") == "1"))
            if m := re.fullmatch(r"/api/tune/bundle/([^/]+)", p):
                return self._json(app.tuning.bundle_analysis(m.group(1), q.get("member", "oof_ensemble")))
            if p == "/api/tune/experiments":
                return self._json(app.tuning.experiments())
            if m := re.fullmatch(r"/api/tune/experiment/([^/]+)", p):
                return self._json(app.tuning.experiment_detail(m.group(1)))
            if m := re.fullmatch(r"/api/tune/spec/([^/]+)", p):
                from .. import classifier as CL

                return self._json(CL.load_spec(m.group(1)))
            if m := re.fullmatch(r"/api/op/(\w+)", p):
                return self._json(app.tuning.op(m.group(1)).to_dict())
            if p == "/api/catalog":
                return self._json(admin.catalog())
            if m := re.fullmatch(r"/api/profile/([^/]+)", p):
                return self._json(admin.profile_get(m.group(1)))
            if p == "/api/registry":
                return self._json(admin.registry())
            if p == "/api/tasks/catalog":
                return self._json(admin.tasks_catalog())
            if p == "/api/tasks":
                return self._json(admin.tasks_list())
            if m := re.fullmatch(r"/api/task/(\w+)", p):
                return self._json(admin.task_log(m.group(1), int(q.get("lines", 120))))
            if m := re.fullmatch(r"/export/profile/([^/]+)\.yaml", p):
                return self._send(200, admin.profile_export_yaml(m.group(1)).encode("utf-8"), "text/yaml; charset=utf-8", {"Content-Disposition": f"attachment; filename={m.group(1)}.yaml"})
            if m := re.fullmatch(r"/export/([^/]+)", p):
                f = admin.export_file(m.group(1))
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(f.stat().st_size))
                self.send_header("Content-Disposition", f"attachment; filename={f.name}")
                self.end_headers()
                with open(f, "rb") as fh:
                    shutil.copyfileobj(fh, self.wfile)
                return
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
            if p == "/api/jobs":
                return self._json(app.jobs_list())
            if m := re.fullmatch(r"/api/job/(\w+)/stream", p):
                return self._json(app.stream(m.group(1), int(q.get("since", 0))))
            if m := re.fullmatch(r"/api/job/(\w+)", p):
                return self._json(app.job(m.group(1)).to_dict())
            if p == "/api/analysis":
                r = app.latest_analysis(q["video"], q.get("profile"))
                return self._json(r or dict(trace=None, meta=None))
            if p == "/api/media-status":
                r = app.media_for_alert(int(q["alert"]), q.get("fmt")) if q.get("alert") else app.media_for(q["id"], q.get("fmt"))
                return self._json(dict(state=r["state"], error=r["error"], pct=round(r["pct"], 3)))
            if m := re.fullmatch(r"/media/video/(.+)", p):
                r = app.media_for(m.group(1), q.get("fmt"))
                if r["path"] is None:
                    return self._json(dict(state=r["state"], error=r["error"], pct=round(r["pct"], 3)), 202 if r["state"] == "preparing" else 500)
                return self._file(r["path"], r["mime"])
            if m := re.fullmatch(r"/media/clip/(\d+)", p):
                r = app.media_for_alert(int(m.group(1)), q.get("fmt"))
                if r["path"] is None:
                    return self._json(dict(state=r["state"], error=r["error"], pct=round(r["pct"], 3)), 202 if r["state"] == "preparing" else 500)
                return self._file(r["path"], r["mime"])
            if m := re.fullmatch(r"/media/export/([^/]+)", p):
                f = app.export_path(m.group(1))
                return self._file(f, "video/mp4") if q.get("view") else self._send(200, f.read_bytes(), "video/mp4", {"Content-Disposition": f"attachment; filename={f.name}"})
            if m := re.fullmatch(r"/media/alert/(\d+)/(thumb|clip)", p):
                return self._file(app.artifact(int(m.group(1)), m.group(2)))
            raise NotFound(p)

        def _raw(self, suffix: str):
            n = int(self.headers.get("Content-Length") or 0)
            return admin.tmp_save(self.rfile, n, suffix), n

        def do_POST(self) -> None:
            u = urlparse(self.path)
            p = unquote(u.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                # --- загрузки файлов: тело запроса = файл (потоком)
                if p == "/api/upload":
                    return self._json(app.save_upload(q.get("name", ""), self.rfile, int(self.headers.get("Content-Length") or 0)))
                raw = {"/api/model/add": None, "/api/solver/inspect": ".sdsolver.zip", "/api/solver/install": ".sdsolver.zip", "/api/bundles/import": ".zip", "/api/config/import": ".yaml"}
                if p in raw:
                    return self._raw_post(p, q, raw[p])
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if m := re.fullmatch(r"/api/alert/(\d+)/review", p):
                    st = body.get("status")
                    if st not in ("new", "confirmed", "false", "unsure"):
                        raise ValueError("status ∈ new, confirmed, false, unsure")
                    return self._json(app.review(int(m.group(1)), st, str(body.get("reviewer", ""))[:60], str(body.get("note", ""))[:500]))
                if p == "/api/export-video":
                    return self._json(app.export_video(body["video"], body.get("profile")))
                if p == "/api/analyze":
                    if body.get("folder"):
                        return self._json([j.to_dict() for j in app.submit_folder(body["folder"], body["profile"])], 202)
                    return self._json(app.submit(body["video"], body["profile"], bool(body.get("live"))).to_dict(), 202)
                if p == "/api/demo":
                    return self._json(dict(n=app.demo(bool(body.get("clear")))))
                if p == "/api/tune/spec/default":
                    return self._json(app.tuning.default_spec(body.get("set", "obj_hold_zsd"), body.get("kinds") or [], body.get("enriched", True)))
                if p == "/api/tune/spec/validate":
                    return self._json(dict(problems=app.tuning.validate_spec(body["spec"], body.get("enriched", True))))
                if p == "/api/tune/spec/save":
                    from .. import classifier as CL

                    sp = body["spec"]
                    if not str(sp.get("name", "")).strip():
                        raise ValueError("у спецификации нет имени")
                    return self._json(dict(file=CL.save_spec(sp).name))
                if p == "/api/tune/diagnose":
                    return self._json(app.tuning.diagnose(body["spec"], int(body.get("n_perm", 3)), bool(body.get("importance", True)), body.get("enriched", True)).to_dict(), 202)
                if p == "/api/tune/train":
                    return self._json(app.tuning.train(body["spec"], str(body.get("name", "")).strip(), body.get("attach") or None, body.get("enriched", True)).to_dict(), 202)
                if p == "/api/tune/eval":
                    return self._json(app.tuning.evaluate(body["profile"], body["folders"], body.get("classifier"), body.get("mode", "events"), body.get("policy", "fixed"),
                                                           bool(body.get("only_labeled", True)), float(body["max_sec"]) if body.get("max_sec") else None, int(body.get("n_boot", 200))).to_dict(), 202)
                if p == "/api/root":
                    return self._json(app.add_root(body["path"]))
                if p == "/api/root/remove":
                    return self._json(dict(ok=app.remove_root(body["key"])))
                if m := re.fullmatch(r"/api/profile/([^/]+)", p):
                    return self._json(admin.profile_save(m.group(1), body))
                if m := re.fullmatch(r"/api/profile/([^/]+)/clone", p):
                    return self._json(admin.profile_clone(m.group(1), body["to"]))
                if m := re.fullmatch(r"/api/profile/([^/]+)/delete", p):
                    from .. import profiles as PR

                    return self._json(dict(ok=PR.delete(m.group(1))))
                if p == "/api/model/remove":
                    return self._json(dict(ok=admin.model_remove(body["id"])))
                if p == "/api/model/add":
                    return self._json(admin.model_add(body["id"], body["kind"], body.get("license", "?"), None, None, body.get("hf_repo")))
                if p == "/api/solver/pack":
                    return self._json(admin.solver_pack(body["profile"], body.get("name"), bool(body.get("with_weights")), bool(body.get("with_dataset")), body.get("classifier"), body.get("run_id")))
                if p == "/api/bundles/export":
                    return self._json(admin.bundles_export(list(body["items"])))
                if p == "/api/task/start":
                    return self._json(admin.task_start(body["task"], body.get("values") or {}), 202)
                if m := re.fullmatch(r"/api/task/(\w+)/stop", p):
                    return self._json(dict(ok=admin.task_stop(m.group(1))))
                raise NotFound(p)
            except NotFound as e:
                self._json(dict(error=str(e)), 404)
            except PermissionError as e:
                self._json(dict(error=str(e)), 403)
            except FileExistsError as e:
                self._json(dict(error=str(e)), 409)
            except (ValueError, KeyError, json.JSONDecodeError, RuntimeError, OSError, zipfile.BadZipFile) as e:
                self._json(dict(error=f"{e}"), 400)
            except Exception as e:
                self._json(dict(error=f"{type(e).__name__}: {e}"), 500)

        def _raw_post(self, p: str, q: dict, suffix: str | None) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            if p == "/api/model/add" and n == 0:        # запись без файла (Hugging Face)
                return self._json(admin.model_add(q["id"], q["kind"], q.get("license", "?"), None, None, q.get("hf")))
            ext = suffix or Path(q.get("name", "")).suffix.lower() or ".bin"
            f = admin.tmp_save(self.rfile, n, ext)
            try:
                if p == "/api/model/add":
                    return self._json(admin.model_add(q["id"], q["kind"], q.get("license", "?"), q.get("name"), f, None, q.get("allow_unknown") == "1"))
                if p == "/api/solver/inspect":
                    return self._json(admin.solver_inspect(f))
                if p == "/api/solver/install":
                    return self._json(admin.solver_install(f, q.get("overwrite") == "1"))
                if p == "/api/bundles/import":
                    return self._json(dict(imported=admin.bundles_import(f, q.get("overwrite") == "1")))
                if p == "/api/config/import":
                    return self._json(admin.profile_import_yaml(f.read_bytes(), q.get("name") or None))
            finally:
                shutil.rmtree(f.parent, ignore_errors=True)

    return H


def serve(app: WebApp | None = None, host: str = "127.0.0.1", port: int = 8502, block: bool = True) -> ThreadingHTTPServer:
    app = app or WebApp()
    app.allow_fs = host in ("127.0.0.1", "localhost", "::1")          # загрузка файлов и подключение папок — только на локальном сервере
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    if os.environ.get("SD_WARM_MEDIA", "1") != "0" and block:          # копии для браузера (wmv/mkv/H.265) готовятся заранее, по одной
        threading.Thread(target=app.warm_media, daemon=True, name="warm-media").start()
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
