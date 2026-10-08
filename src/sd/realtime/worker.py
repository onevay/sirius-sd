"""Воркер мониторинга: прогоняет «камеры» (папки район-индекс-время) через `StreamEngine` и пишет тревоги в хранилище.

    sd monitor --streams streams/ --profile final            # обработать всё, что есть, и выйти
    sd monitor --streams streams/ --watch --speed 1          # следить за папками: новые фрагменты подхватываются; speed=1 — как в реальном времени

Состояние потока (буфер, события) держится в памяти на камеру; обработанные фрагменты записываются в базу — после перезапуска воркер продолжает с первого необработанного (контекст жеста на стыке
фрагментов при перезапуске теряется). Сбой артефакта (миниатюра, клип) не останавливает мониторинг: тревога важнее картинки.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..paths import OUTPUTS, portable
from ..profiles import Profile
from . import sources as S
from .engine import BundleScorer, HeuristicScorer, StreamEngine
from .store import AlertStore

ARTIFACTS = OUTPUTS / "monitor" / "alerts"


def lazy_pose_fn(profile: Profile, process_fps: float):
    """`pose_fn`, который создаёт `PoseTracker` по первому кадру (размер кадра известен только тогда). Тяжёлые зависимости импортируются при первом вызове."""
    state: dict = {}

    def fn(img, t):
        if "pt" not in state:
            from ..pose_track import PoseTracker

            state["pt"] = PoseTracker(profile.cfg(), process_fps, (img.shape[0], img.shape[1]))
        return state["pt"].step(img, t)

    return fn


def make_engine(profile: Profile, camera_id: str, process_fps: float | None = None, pose_fn=None, keep_frames: bool = True, *, mode: str = "live",
                source_path: str | Path | None = None) -> StreamEngine:
    """Движок камеры. Классификатор цикла обязателен (`solver.check`): без него или с признаками, которых в этом режиме нет, — `ValueError`, а не тихая эвристика.
    `mode='replay'` + `source_path` — имитация потока по файлу: предмет и фото-модель читаются из исходного файла."""
    from ..solver import errors
    from .scoring import ClassifierScorer

    errs = errors(profile, mode)
    if errs:
        raise ValueError("\n".join(e.text for e in errs))
    cfg = profile.cfg()
    fps = process_fps or float(cfg["video"]["process_fps"])
    o = profile.opts()
    scorer = (ClassifierScorer(o["cycle_bundle"], cfg, camera_id, objects=o["objects"], photo_bundle=o["photo_bundle"], source_path=source_path, backend=o["backend"])
              if o.get("cycle_bundle") else HeuristicScorer())
    return StreamEngine(cfg, camera_id, pose_fn or lazy_pose_fn(profile, fps), scorer, keep_frames=keep_frames, source_path=source_path)


@dataclass
class Span:
    offset: float
    duration: float
    path: Path


def _chunk_for(spans: list[Span], t: float) -> Span | None:
    for sp in reversed(spans):
        if sp.offset - 1e-6 <= t:
            return sp
    return spans[0] if spans else None


def process_stream(stream: S.StreamDir, store: AlertStore, engine: StreamEngine, profile: Profile, *, process_fps: float = 10.0, speed: float = 0.0,
                   frames_fn: Callable = S.iter_chunk_frames, probe_fn: Callable | None = None, stop: threading.Event | None = None, finalize: bool = True,
                   demo: bool = False, artifacts_dir: Path | None = None, log: Callable[[str], None] = print) -> dict:
    """Обрабатывает необработанные фрагменты одной камеры. Возвращает {chunks, alerts, updates, seconds}."""
    from ..video_io import probe as _probe

    probe_fn = probe_fn or _probe
    root = artifacts_dir or ARTIFACTS
    cam = stream.camera_id
    store.upsert_camera(cam, stream.district, stream.index, portable(stream.path) or str(stream.path), stream.start.isoformat(timespec="seconds"))
    done = store.chunks_done(cam)
    spans = [Span(off, dur, stream.path / name) for name, (dur, off) in done.items()]
    offset = max((sp.offset + sp.duration for sp in spans), default=0.0)
    fp = profile.fingerprint()
    res = dict(chunks=0, alerts=0, updates=0, seconds=0.0)
    t_wall = time.perf_counter()

    def handle(ups) -> None:
        for u in ups:
            sp = _chunk_for(spans, u.peak)
            meta = dict(district=stream.district, cam_index=stream.index, chunk=portable(sp.path) if sp else None, chunk_offset=sp.offset if sp else 0.0,
                        t_abs=S.abs_time(stream, u.start).isoformat(timespec="seconds"), fingerprint=fp, demo=int(demo))
            aid, kind = store.apply_update(u, meta)
            if aid < 0:
                continue
            res["alerts" if kind == "open" else "updates"] += 1
            if kind in ("open", "update"):
                d = root / str(aid)
                try:
                    d.mkdir(parents=True, exist_ok=True)
                    jpg = engine.thumbnail(u)
                    thumb = None
                    if jpg:
                        (d / "thumb.jpg").write_bytes(jpg)
                        thumb = portable(d / "thumb.jpg")
                    clip = portable(d / "clip.mp4") if engine.save_clip(u, d / "clip.mp4") else None
                    store.set_artifacts(aid, thumb, clip)
                except Exception as e:     # картинка/клип — не повод ронять мониторинг
                    log(f"[{cam}] артефакты тревоги {aid}: {type(e).__name__}: {e}")
            if kind == "open":
                log(f"[{cam}] ТРЕВОГА #{aid}: {S.abs_time(stream, u.start):%H:%M:%S} · {u.explain} · {u.confidence:.0%}")

    for chunk in S.list_chunks(stream.path):
        if chunk.name in done:
            continue
        if stop is not None and stop.is_set():
            break
        try:
            dur = float(probe_fn(chunk).duration)
        except Exception as e:
            log(f"[{cam}] {chunk.name}: не открывается ({e}); пропущен")
            store.mark_chunk(cam, chunk.name, 0.0, offset)
            continue
        sp = Span(offset, dur, chunk)
        spans.append(sp)
        store.touch_camera(cam, "processing")
        pacer = S.Pacer(speed)
        for img, t in frames_fn(chunk, offset, process_fps):
            if stop is not None and stop.is_set():
                break
            pacer.wait(t)
            handle(engine.step(img, t))
        store.mark_chunk(cam, chunk.name, dur, offset)
        offset += dur
        res["chunks"] += 1
        store.touch_camera(cam, "processing", processed_sec=offset)
    if finalize:
        handle(engine.flush(offset))
    store.touch_camera(cam, "idle", processed_sec=offset)
    res["seconds"] = round(time.perf_counter() - t_wall, 1)
    return res


def run_monitor(streams_root: str | Path, profile: Profile, store: AlertStore | None = None, *, cameras: Iterable[str] | None = None, watch: bool = False, interval: float = 5.0,
                speed: float = 0.0, process_fps: float | None = None, stop: threading.Event | None = None, engine_factory: Callable | None = None, log: Callable[[str], None] = print,
                **kw) -> dict:
    """Основной цикл: найти камеры → обработать новые фрагменты → (в режиме `watch`) подождать и повторить."""
    store = store or AlertStore()
    fps = process_fps or float(profile.cfg()["video"]["process_fps"])
    engines: dict[str, StreamEngine] = {}
    total = dict(chunks=0, alerts=0, updates=0)
    stop = stop or threading.Event()
    while not stop.is_set():
        streams = [s for s in S.scan_streams(streams_root) if not cameras or s.camera_id in set(cameras)]
        if not streams and not watch:
            log(f"в {streams_root} нет папок-камер с видео")
        for st in streams:
            eng = engines.get(st.camera_id) or engines.setdefault(st.camera_id, (engine_factory or make_engine)(profile, st.camera_id, fps))
            r = process_stream(st, store, eng, profile, process_fps=fps, speed=speed, stop=stop, finalize=not watch, log=log, **kw)
            for k in total:
                total[k] += r[k]
        if not watch:
            break
        stop.wait(interval)
    return total
