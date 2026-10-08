"""Логика веб-приложения (без HTTP — тестируется напрямую): камеры и карта, видео из папок, тревоги и решения оператора, фоновый анализ видео моделью."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from .. import paths as P
from ..realtime import sources as S
from ..realtime.store import AlertStore
from . import geo, media

STATUS_RU = {"new": "ждёт решения", "confirmed": "подтверждена", "false": "ложная", "unsure": "не уверен"}


class NotFound(KeyError):
    pass


class Job:
    def __init__(self, video: str, profile: str, name: str):
        self.id, self.video, self.profile, self.name = uuid.uuid4().hex[:10], video, profile, name
        self.state, self.progress, self.error = "queued", 0.0, None
        self.alerts: list[dict] = []
        self.stats: dict | None = None
        self.metrics: dict | None = None
        self.notes: list[str] = []
        self.created = time.time()

    def to_dict(self) -> dict:
        return dict(id=self.id, video=self.video, profile=self.profile, name=self.name, state=self.state, progress=round(self.progress, 3), error=self.error, alerts=self.alerts,
                    stats=self.stats, metrics=self.metrics, notes=self.notes)


class WebApp:
    def __init__(self, store: AlertStore | None = None, streams: Path | None = None, data: Path | None = None, out: Path | None = None):
        self.store = store or AlertStore()
        self.out = Path(out) if out else P.OUTPUTS / "web"
        extra = [Path(x) for x in os.environ.get("SD_WEB_ROOTS", "").split(os.pathsep) if x]
        self.roots: dict[str, Path] = {"streams": Path(streams or P.STREAMS), "data": Path(data or P.DATA), **{f"extra{i}": p for i, p in enumerate(extra)}}
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._dur: dict[tuple, dict] = {}
        self.run_job = self._run_job            # подменяется в тестах

    # ------------------------------------------------------------------ идентификаторы файлов (путь не уходит в браузер; выход за корни невозможен)
    def fid(self, path: Path) -> str:
        p = Path(path).resolve()
        for k, r in self.roots.items():
            try:
                rel = p.relative_to(r.resolve())
            except ValueError:
                continue
            return base64.urlsafe_b64encode(f"{k}|{rel.as_posix()}".encode()).decode().rstrip("=")
        raise NotFound("файл вне разрешённых папок")

    def resolve(self, fid: str) -> Path:
        try:
            k, rel = base64.urlsafe_b64decode(fid + "=" * (-len(fid) % 4)).decode().split("|", 1)
            root = self.roots[k].resolve()
            p = (root / rel).resolve()
            p.relative_to(root)
        except Exception:
            raise NotFound("неверный идентификатор") from None
        if not p.exists():
            raise NotFound("файл не найден")
        return p

    # ------------------------------------------------------------------ видео и папки
    def info(self, path: Path) -> dict:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self._dur:
            try:
                from ..video_io import probe

                i = probe(path)
                self._dur[key] = dict(duration=i.duration, fps=i.fps, width=i.width, height=i.height, codec=i.codec)
            except Exception as e:
                self._dur[key] = dict(duration=None, error=str(e)[:120])
        return {**self._dur[key], "size_mb": round(st.st_size / 1e6, 1)}

    def folders(self) -> list[dict]:
        out = []
        sroot = self.roots["streams"]
        for s in S.scan_streams(sroot):
            out.append(dict(id=self.fid(s.path), name=s.path.name, kind="camera", root="streams", n=len(S.list_chunks(s.path)), camera_id=s.camera_id))
        seen = set()
        for k, r in self.roots.items():
            if k == "streams" or not r.is_dir():
                continue
            for v in P.list_videos(r):
                if v.parent in seen:
                    continue
                seen.add(v.parent)
                out.append(dict(id=self.fid(v.parent), name=str(v.parent.relative_to(r)) if v.parent != r else r.name, kind="data", root=k,
                                n=sum(1 for f in v.parent.iterdir() if f.suffix.lower() in P.VIDEO_EXT)))
        return out

    def videos(self, folder_fid: str) -> list[dict]:
        d = self.resolve(folder_fid)
        files = S.list_chunks(d)
        out = []
        for f in files:
            i = self.info(f)
            out.append(dict(id=self.fid(f), name=f.name, clip_id=P.video_id(f), analyzed=self.latest_analysis(self.fid(f)) is not None, **i))
        return out

    def folder_of(self, vid: str) -> str:
        return self.fid(self.resolve(vid).parent)

    def media_for(self, vid: str) -> tuple[str, Path | None, str | None]:
        return media.playable(self.resolve(vid))

    # ------------------------------------------------------------------ камеры (карта)
    def cameras(self) -> list[dict]:
        ov = geo.load_overrides(self.roots["streams"])
        known = {r["camera_id"]: r for r in self.store.cameras().to_dict("records")}
        out, seen = [], set()
        for s in S.scan_streams(self.roots["streams"]):
            seen.add(s.camera_id)
            ch = S.list_chunks(s.path)
            k = known.get(s.camera_id, {})
            out.append(dict(camera_id=s.camera_id, district=s.district, index=s.index, start=s.start.isoformat(timespec="seconds"), n_chunks=len(ch), folder=self.fid(s.path),
                            status=k.get("status") or "idle", processed_sec=k.get("processed_sec") or 0.0, alerts=k.get("alerts") or 0, pending=k.get("pending") or 0,
                            **geo.locate(s.district, s.index, ov)))
        for cid, k in known.items():          # камеры из базы без папки (демо-тревоги, перенесённые данные)
            if cid not in seen:
                out.append(dict(camera_id=cid, district=k["district"], index=k["cam_index"], start=k["start_at"], n_chunks=0, folder=None, status=k.get("status") or "idle",
                                processed_sec=k.get("processed_sec") or 0.0, alerts=k.get("alerts") or 0, pending=k.get("pending") or 0, **geo.locate(k["district"], k["cam_index"], ov)))
        return out

    def camera(self, camera_id: str) -> dict:
        for c in self.cameras():
            if c["camera_id"] == camera_id:
                c = dict(c)
                c["chunks"] = self.videos(c["folder"]) if c["folder"] else []
                return c
        raise NotFound(camera_id)

    # ------------------------------------------------------------------ тревоги
    def alerts(self, status=None, camera=None, district=None, min_conf=None, limit=100, since=None) -> list[dict]:
        df = self.store.list_alerts(status=status or None, cameras=[camera] if camera else None, districts=[district] if district else None, min_conf=min_conf, limit=limit, since=since)
        return [self._alert(a) for a in df.to_dict("records")] if len(df) else []

    def _alert(self, a: dict) -> dict:
        a = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in a.items()}
        a["status_ru"] = STATUS_RU.get(a["status"], a["status"])
        a["has_thumb"] = bool(self._artifact(a.get("thumb")))
        a["has_clip"] = bool(self._artifact(a.get("clip")))
        return a

    def _artifact(self, portable_path) -> Path | None:
        p = P.from_portable(portable_path) if portable_path else None
        if p and Path(p).exists():
            try:
                Path(p).resolve().relative_to(P.OUTPUTS.resolve())
                return Path(p)
            except ValueError:
                return None
        return None

    def alert(self, alert_id: int) -> dict:
        a = self.store.get(alert_id)
        if not a:
            raise NotFound(f"тревога {alert_id}")
        return self._alert(a)

    def artifact(self, alert_id: int, kind: str) -> Path:
        a = self.store.get(alert_id)
        p = self._artifact((a or {}).get("thumb" if kind == "thumb" else "clip"))
        if not p:
            raise NotFound("нет файла")
        return p

    def review(self, alert_id: int, status: str, reviewer: str = "", note: str = "") -> dict:
        self.alert(alert_id)
        self.store.review(alert_id, status, reviewer, note)
        return self.alert(alert_id)

    def stats(self) -> dict:
        s = self.store.stats()
        s["max_id"] = self.store.max_id()
        return s

    def journal_csv(self) -> str:
        df = self.store.list_alerts(status=["confirmed", "false", "unsure"], limit=5000)
        return df[["id", "t_abs", "district", "cam_index", "confidence", "status", "reviewer", "reviewed_at", "note", "demo"]].to_csv(index=False) if len(df) else "id\n"

    def demo(self, clear: bool = False) -> int:
        from ..realtime.demo import seed_demo

        return self.store.delete_demo() if clear else seed_demo(self.store)

    # ------------------------------------------------------------------ анализ видео моделью
    def profiles(self) -> list[dict]:
        from .. import profiles as PR
        from .. import solver as SV

        out = []
        for n in PR.list_profiles():
            try:
                p = PR.load(n)
                iss = SV.check(p, "replay", devices=True)
                out.append(dict(name=n, kind="profile", ready=not any(i.level == "error" for i in iss), problems=[i.text for i in iss if i.level == "error"], describe=p.describe()))
            except Exception as e:
                out.append(dict(name=n, kind="profile", ready=False, problems=[str(e)], describe={}))
        for s in SV.installed_solvers():
            try:
                p = SV.resolve_profile(s["name"])
                iss = SV.check(p, "replay", devices=True)
                out.append(dict(name=s["name"], kind="solver", ready=not any(i.level == "error" for i in iss), problems=[i.text for i in iss if i.level == "error"], describe=p.describe()))
            except Exception as e:
                out.append(dict(name=s["name"], kind="solver", ready=False, problems=[str(e)], describe={}))
        return out

    def _result_dir(self, vid: str, profile: str) -> Path:
        return self.out / "analysis" / hashlib.sha1(f"{vid}|{profile}".encode()).hexdigest()[:14]

    def latest_analysis(self, vid: str, profile: str | None = None) -> dict | None:
        base = self.out / "analysis"
        cands = [self._result_dir(vid, profile)] if profile else sorted(base.glob("*"), key=lambda d: d.stat().st_mtime, reverse=True) if base.exists() else []
        for d in cands:
            f = d / "trace.json"
            if f.exists():
                try:
                    t = json.loads(f.read_text(encoding="utf-8"))
                    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                except Exception:
                    continue
                if meta.get("video") == vid:
                    return dict(trace=t, meta=meta)
        return None

    def submit(self, vid: str, profile: str) -> Job:
        path = self.resolve(vid)
        if profile not in {p["name"] for p in self.profiles()}:
            raise NotFound(f"профиль {profile}")
        j = Job(vid, profile, path.name)
        self.jobs[j.id] = j
        self._q.put(j)
        if not self._worker or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()
        return j

    def _loop(self) -> None:
        while True:
            try:
                j = self._q.get(timeout=300)
            except queue.Empty:
                return
            self.run_job(j)

    def _run_job(self, j: Job) -> None:
        from .. import gt as GT
        from .. import profiles as PR
        from .. import runner as RN
        from .. import solver as SV
        from ..realtime import replay as RP

        j.state = "running"
        try:
            path = self.resolve(j.video)
            insts = {s["name"] for s in SV.installed_solvers()}
            prof = SV.resolve_profile(j.profile) if j.profile in insts and j.profile not in PR.list_profiles() else PR.load(j.profile)
            errs = SV.errors(prof, "replay", devices=True)
            if errs:
                raise RuntimeError("; ".join(e.text for e in errs))
            dur = float(self.info(path).get("duration") or 0.0)

            def on_alert(u) -> None:
                if u.kind == "open":
                    j.alerts.append(dict(t_now=round(u.t_now, 2), start=round(u.start, 2), tid=u.tid, confidence=round(u.confidence, 3), explain=u.explain))

            res = RP.replay_video(path, prof, speed=0.0, gt=GT.load(), render=False, on_alert=on_alert, save=False,
                                  progress=lambda t, e: setattr(j, "progress", min(0.99, t / dur if dur else 0.0)))
            d = self._result_dir(j.video, j.profile)
            d.mkdir(parents=True, exist_ok=True)
            (d / "trace.json").write_text(json.dumps(res.trace, separators=(",", ":")), encoding="utf-8")
            (d / "meta.json").write_text(json.dumps(dict(video=j.video, profile=j.profile, created=datetime.now().isoformat(timespec="seconds"), stats=res.stats, metrics=res.metrics,
                                                         notes=res.notes), ensure_ascii=False, default=str), encoding="utf-8")
            j.stats, j.metrics, j.notes, j.progress, j.state = res.stats, res.metrics, res.notes, 1.0, "done"
        except Exception as e:
            j.state, j.error = "error", f"{type(e).__name__}: {e}"[:400]

    def job(self, jid: str) -> Job:
        if jid not in self.jobs:
            raise NotFound(jid)
        return self.jobs[jid]
