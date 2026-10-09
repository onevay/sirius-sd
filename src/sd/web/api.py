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
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .. import paths as P
from ..realtime import sources as S
from ..realtime.store import AlertStore
from . import geo, media

STATUS_RU = {"new": "ждёт решения", "confirmed": "подтверждена", "false": "ложная", "unsure": "не уверен"}


class NotFound(KeyError):
    pass


class Job:
    def __init__(self, video: str, profile: str, name: str, live: bool = False):
        self.id, self.video, self.profile, self.name, self.live = uuid.uuid4().hex[:10], video, profile, name, live
        self.state, self.progress, self.error = "queued", 0.0, None
        self.alerts: list[dict] = []
        self.frames: list = []                 # живой режим: кадры [t, [[tid, x1, y1, x2, y2]]] по мере обработки
        self.frame_hw: list | None = None
        self.fps: float = 10.0
        self.gt: list[dict] = []
        self.cycles: list[dict] = []           # оценённые циклы с уверенностью классификатора и сигналами
        self.cycle_th = self.event_th = 0.5
        self.horizon = 0.0                     # до какой секунды видео модель уже посчитала
        self._alert_map: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.stats: dict | None = None
        self.metrics: dict | None = None
        self.notes: list[str] = []
        self.created = time.time()

    def to_dict(self) -> dict:
        return dict(id=self.id, video=self.video, profile=self.profile, name=self.name, live=self.live, state=self.state, progress=round(self.progress, 3), error=self.error, alerts=self.alerts,
                    stats=self.stats, metrics=self.metrics, notes=self.notes)


class WebApp:
    def __init__(self, store: AlertStore | None = None, streams: Path | None = None, data: Path | None = None, out: Path | None = None):
        self.store = store or AlertStore()
        self.out = Path(out) if out else P.OUTPUTS / "web"
        extra = [Path(x) for x in os.environ.get("SD_WEB_ROOTS", "").split(os.pathsep) if x]
        self.roots: dict[str, Path] = {"streams": Path(streams or P.STREAMS), "data": Path(data or P.DATA), **{f"extra{i}": p for i, p in enumerate(extra)}}
        self.allow_fs = True                     # добавление папок и загрузка файлов: только для локального сервера (server.serve выставляет по адресу привязки)
        self._roots_file = self.out / "roots.json"
        for r in self._load_roots():
            self.roots[r["key"]] = Path(r["path"])
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._dur: dict[tuple, dict] = {}
        self.run_job = self._run_job            # подменяется в тестах

    # ------------------------------------------------------------------ источники видео: свои папки и загрузка
    def _load_roots(self) -> list[dict]:
        try:
            return [r for r in json.loads(self._roots_file.read_text(encoding="utf-8")) if Path(r["path"]).is_dir()]
        except Exception:
            return []

    def _save_roots(self) -> None:
        rows = [dict(key=k, path=str(p)) for k, p in self.roots.items() if k.startswith("x")]
        self._roots_file.parent.mkdir(parents=True, exist_ok=True)
        self._roots_file.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def add_root(self, path: str) -> dict:
        """Подключает папку с видео на этом компьютере (рекурсивно). Возвращает {key, path, n}."""
        if not self.allow_fs:
            raise PermissionError("добавление папок отключено: сервер доступен из сети")
        p = Path(str(path).strip().strip('"')).expanduser()
        if not p.is_dir():
            raise ValueError(f"папка не найдена: {p}")
        p = p.resolve()
        key = "x" + hashlib.sha1(str(p).encode()).hexdigest()[:8]
        self.roots[key] = p
        self._save_roots()
        return dict(key=key, path=str(p), n=len(P.list_videos(p)))

    def remove_root(self, key: str) -> bool:
        if key.startswith("x") and key in self.roots:
            del self.roots[key]
            self._save_roots()
            return True
        return False

    def upload_dir(self) -> Path:
        d = self.roots["data"] / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_upload(self, name: str, stream, length: int) -> dict:
        """Принимает видео из браузера потоком (без загрузки в память). Формат любой из `VIDEO_EXT`; имя очищается; существующий файл не затирается."""
        if not self.allow_fs:
            raise PermissionError("загрузка отключена: сервер доступен из сети")
        base = Path(str(name).replace("\\", "/")).name
        ext = Path(base).suffix.lower()
        if ext not in P.VIDEO_EXT:
            raise ValueError(f"формат {ext or '?'} не поддерживается (допустимо: {', '.join(sorted(P.VIDEO_EXT))})")
        limit = float(os.environ.get("SD_MAX_UPLOAD_GB", "8")) * 1e9
        if length <= 0 or length > limit:
            raise ValueError("пустой файл или больше допустимого размера")
        stem = "".join(c if c.isalnum() or c in "-_." or ord(c) > 127 else "_" for c in Path(base).stem)[:80] or "video"
        d = self.upload_dir()
        dst = d / f"{stem}{ext}"
        k = 1
        while dst.exists():
            dst = d / f"{stem}_{k}{ext}"
            k += 1
        tmp = dst.with_name(dst.name + ".part")
        left = length
        try:
            with open(tmp, "wb") as f:
                while left > 0:
                    chunk = stream.read(min(1 << 20, left))
                    if not chunk:
                        raise ValueError("загрузка оборвалась")
                    f.write(chunk)
                    left -= len(chunk)
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)
        return dict(id=self.fid(dst), name=dst.name, folder=self.fid(d), **self.info(dst))

    def media_for_alert(self, alert_id: int, fmt: str | None = None) -> dict:
        p = self.artifact(alert_id, "clip")
        return media.playable(p, fmt, float(self.info(p).get("duration") or 0.0))

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
                out.append(dict(id=self.fid(v.parent), name=str(v.parent.relative_to(r)) if v.parent != r else r.name, kind="data", root=k, removable=k.startswith("x"),
                                n=sum(1 for f in v.parent.iterdir() if f.suffix.lower() in P.VIDEO_EXT)))
        return out

    def videos(self, folder_fid: str) -> list[dict]:
        d = self.resolve(folder_fid)
        files = S.list_chunks(d)
        out = []
        for f in files:
            i = self.info(f)
            out.append(dict(id=self.fid(f), name=f.name, clip_id=P.video_id(f), analysis=self.analysis_summary(self.fid(f)), **i))
        return out

    def folder_of(self, vid: str) -> str:
        return self.fid(self.resolve(vid).parent)

    def media_for(self, vid: str, fmt: str | None = None) -> dict:
        p = self.resolve(vid)
        return media.playable(p, fmt, float(self.info(p).get("duration") or 0.0))

    # ------------------------------------------------------------------ камеры (карта)
    def camera_metrics(self, hours: float = 0.0) -> dict[str, dict]:
        """Показатели по камерам за период (0 — за всё время): тревог, в час, суммарное время курения, средняя уверенность, время реакции оператора, активность относительно самой активной камеры."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds") if hours else None
        df = self.store.list_alerts(since=since, limit=20000)
        out: dict[str, dict] = {}
        if len(df):
            df = df.assign(dur=(df.end_sec - df.start_sec).clip(lower=0))
            span = hours or max(1.0, (pd.to_datetime(df.t_abs).max() - pd.to_datetime(df.t_abs).min()).total_seconds() / 3600)
            for cid, g in df.groupby("camera_id"):
                rv = g[g.reviewed_at.notna()]
                react = (pd.to_datetime(rv.reviewed_at) - pd.to_datetime(rv.t_raised)).dt.total_seconds().median() if len(rv) else None
                out[cid] = dict(alerts=int(len(g)), per_hour=round(len(g) / span, 2), smoke_sec=round(float(g.dur.sum()), 1), avg_conf=round(float(g.confidence.mean()), 3),
                                reaction_sec=None if react is None or react != react else round(float(react), 1), pending=int((g.status == "new").sum()))
            top = max(v["alerts"] for v in out.values())
            for v in out.values():
                v["activity_rel"] = round(v["alerts"] / top, 3)
        return out

    def cameras(self, hours: float = 0.0) -> list[dict]:
        metrics = self.camera_metrics(hours)
        rows = self._cameras()
        for c in rows:
            m = metrics.get(c["camera_id"]) or dict(alerts=0, per_hour=0.0, smoke_sec=0.0, avg_conf=None, reaction_sec=None, pending=0, activity_rel=0.0)
            c["m"] = m
            c["pending_period"] = m["pending"]
            c["notice"] = (dict(title="Нет видео", reason="в папке камеры нет фрагментов") if not c["n_chunks"] and not c["alerts"] else
                           dict(title="Камера ещё не обрабатывалась", reason="воркер не запускался: sd monitor") if c["n_chunks"] and not c["processed_sec"] and not c["alerts"] else
                           dict(title="Низкая уверенность модели", reason=f"средняя {round(m['avg_conf'] * 100)}% при {m['alerts']} тревогах") if m["avg_conf"] is not None and m["alerts"] >= 3 and m["avg_conf"] < 0.55 else None)
        return rows

    def _cameras(self) -> list[dict]:
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
        for c in self._cameras():
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
        return sorted(out, key=lambda p: (not p["ready"], p["kind"] != "solver", p["name"]))

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

    def analysis_summary(self, vid: str) -> dict | None:
        """Краткий итог последнего разбора видео (без чтения трассы): {profile, alerts, f1, rt_factor, created}."""
        base = self.out / "analysis"
        best = None
        for d in base.glob("*/meta.json") if base.exists() else []:
            try:
                m = json.loads(d.read_text(encoding="utf-8"))
            except Exception:
                continue
            if m.get("video") == vid and (best is None or m.get("created", "") > best.get("created", "")):
                best = m
        if not best:
            return None
        return dict(profile=best.get("profile"), alerts=best.get("n_alerts"), f1=(best.get("metrics") or {}).get("f1"), rt_factor=(best.get("stats") or {}).get("rt_factor"), created=best.get("created"))

    def submit_folder(self, folder: str, profile: str) -> list[Job]:
        return [self.submit(v["id"], profile) for v in self.videos(folder)]

    def jobs_list(self, limit: int = 50) -> list[dict]:
        return [{k: v for k, v in j.to_dict().items() if k != "alerts"} | dict(n_alerts=len(j.alerts)) for j in sorted(self.jobs.values(), key=lambda j: -j.created)[:limit]]

    def submit(self, vid: str, profile: str, live: bool = False) -> Job:
        path = self.resolve(vid)
        if profile not in {p["name"] for p in self.profiles()}:
            raise NotFound(f"профиль {profile}")
        j = Job(vid, profile, path.name, live)
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

    def load_profile(self, name: str):
        from .. import profiles as PR
        from .. import solver as SV

        insts = {s["name"] for s in SV.installed_solvers()}
        return SV.resolve_profile(name) if name in insts and name not in PR.list_profiles() else PR.load(name)

    engine_factory = None        # тесты и встраивание: (профиль, clip_id, fps) -> StreamEngine вместо реальных моделей

    def _run_job(self, j: Job) -> None:
        from .. import gt as GT
        from .. import solver as SV
        from ..realtime import replay as RP

        j.state = "running"
        try:
            path = self.resolve(j.video)
            prof = self.load_profile(j.profile)
            if self.engine_factory is None:
                errs = SV.errors(prof, "replay", devices=True)
                if errs:
                    raise RuntimeError("; ".join(e.text for e in errs))
            dur = float(self.info(path).get("duration") or 0.0)
            gt_df = GT.load()
            cid = P.video_id(path)
            sub = gt_df[(gt_df["clip_id"] == cid) & (gt_df["label"] == "POSITIVE")] if len(gt_df) else gt_df
            j.gt = [dict(start=float(r.start_sec), end=float(r.end_sec)) for r in sub.itertuples()]
            fps = float(prof.cfg()["video"]["process_fps"])
            j.fps = fps
            j.cycle_th, j.event_th = float(prof.cfg()["events"]["cycle_th"]), float(prof.threshold)

            def on_alert(u) -> None:
                a = dict(key=u.key, tid=int(u.tid), start=round(float(u.start), 2), end=round(float(u.end), 2), confidence=round(float(u.confidence), 3), explain=u.explain,
                         rule=u.rule, n_cycles=u.n_cycles, peak=round(float(u.peak), 2))
                with j.lock:
                    old = j._alert_map.get(u.key)
                    a["t_open"] = old["t_open"] if old else round(float(u.t_now), 2)
                    a["delay"] = round(a["t_open"] - a["start"], 2)
                    if u.kind == "close":        # у «close» нет причины и рамки: сохраняем то, что уже знаем об этой тревоге
                        if old:
                            old["end"] = max(old["end"], a["end"])
                    else:
                        j._alert_map[u.key] = a
                    j.alerts = list(j._alert_map.values())

            def on_frame(row, hw, eng_) -> None:
                with j.lock:
                    j.frames.append(row)
                    if len(eng_.cycle_log) != len(j.cycles):
                        j.cycles = [dict(c, passed=bool(c["score"] >= j.cycle_th)) for c in eng_.cycle_log]
                    j.horizon = row[0]
                    if hw and not j.frame_hw:
                        j.frame_hw = list(hw)

            eng = self.engine_factory(prof, cid, fps) if self.engine_factory else None
            res = RP.replay_video(path, prof, speed=1.0 if j.live else 0.0, gt=gt_df, render=False, on_alert=on_alert, on_frame=on_frame, save=False, engine=eng,
                                  progress=lambda t, e: setattr(j, "progress", min(0.99, t / dur if dur else 0.0)))
            d = self._result_dir(j.video, j.profile)
            d.mkdir(parents=True, exist_ok=True)
            (d / "trace.json").write_text(json.dumps(res.trace, separators=(",", ":")), encoding="utf-8")
            (d / "meta.json").write_text(json.dumps(dict(video=j.video, profile=j.profile, created=datetime.now().isoformat(timespec="seconds"), n_alerts=len(res.alerts), stats=res.stats,
                                                         metrics=res.metrics, notes=res.notes), ensure_ascii=False, default=str), encoding="utf-8")
            with j.lock:
                j.stats, j.metrics, j.notes, j.progress, j.horizon, j.state = res.stats, res.metrics, res.notes, 1.0, max(j.horizon, dur), "done"
                j.alerts = [dict(a, key=f"{i}") for i, a in enumerate(res.trace["alerts"])] if res.trace else j.alerts
        except Exception as e:
            j.state, j.error = "error", f"{type(e).__name__}: {e}"[:400]

    def stream(self, jid: str, since: int = 0) -> dict:
        """Живой вывод: кадры с номера `since`, все тревоги на сейчас, до какой секунды посчитано. Клиент опрашивает ~3 раза в секунду."""
        j = self.job(jid)
        with j.lock:
            new = j.frames[since:since + 2000]
            return dict(state=j.state, progress=round(j.progress, 3), error=j.error, frames=new, next=since + len(new), alerts=list(j.alerts), horizon=round(j.horizon, 2), frame_hw=j.frame_hw,
                        fps=j.fps, gt=j.gt, cycles=j.cycles, thresholds=dict(cycle=j.cycle_th, event=j.event_th), stats=j.stats if j.state == "done" else None, metrics=j.metrics)

    def job(self, jid: str) -> Job:
        if jid not in self.jobs:
            raise NotFound(jid)
        return self.jobs[jid]
