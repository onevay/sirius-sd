"""Этапы пайплайна с кэшем артефактов — общий код для CLI и веб-интерфейса.

Раскладка артефактов:
  outputs/runs/<video_id>/<run_id>/pose/                  tracks.parquet, keypoints.npy, frame_t.parquet, meta.json
  outputs/runs/<video_id>/<run_id>/features/<hash>/       series.parquet           (ряд признаков каждого трека)
  outputs/runs/<video_id>/<run_id>/cycles/<hash>/         cycles.parquet, rejected.parquet, states.parquet
  outputs/runs/<video_id>/<run_id>/events/<hash>/         events.csv
`run_id` кодирует интервал и параметры позы/трекинга; `hash` — параметры признаков/циклов/событий, поэтому дорогую позу не пересчитываем
при подборе порогов (UI меняет порог -> пересчитываются только дешёвые этапы).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import stable_hash
from .cycles import STATE_ID, Cycle, find_cycles
from .events import Event, ScoredCycle, build_events, events_to_frame, heuristic_cycle_score
from .features import build_series, kp_threshold, track_grid, visibility_fraction
from .paths import OUTPUTS, video_id
from .pose_track import PoseTracker
from .tracks import Tracks, stitch_tracks
from .video_io import choose_stride, iter_frames, probe

Progress = Callable[[int, int], None] | None


def run_id_for(cfg: dict, start: float, end: float | None) -> str:
    p, t = cfg["pose"], cfg["tracking"]
    span = f"{start:g}-{'end' if end is None else format(end, 'g')}s"
    ref = f"_ref-{p['refine'].get('method', 'yolo')}" if p["refine"]["enabled"] else ""
    return (f"{span}_{p['weights']}_{p['runtime']}{'' if p['runtime'] == 'torch' else '-' + p['device'].split(':')[-1]}_{p['imgsz']}{ref}"
            f"_{t['tracker']}-{stable_hash(t, 4)}_{cfg['video']['process_fps']:g}fps")


def run_window(rd: Path) -> tuple[float, float]:
    """Окно запуска в секундах видео: (начало, конец); `end` отсутствует в meta — до конца видео. Нет run.json — (0, 0)."""
    f = rd / "run.json"
    if not f.exists():
        return 0.0, 0.0
    meta = json.loads(f.read_text(encoding="utf-8")).get("meta", {})
    a = float(meta.get("start") or 0.0)
    b = meta.get("end")
    return a, float(b) if b is not None else float(meta.get("video_info", {}).get("duration", 0.0))


def run_created(rd: Path) -> float:
    """Время создания запуска (mtime результата позы): по нему решается, какой из пересекающихся запусков «основной»."""
    for name in ("pose/tracks.parquet", "run.json"):
        f = rd / name
        if f.exists():
            return f.stat().st_mtime
    return float("inf")


def drop_overlapping_runs(runs: list[tuple[str, Path]], min_overlap_sec: float = 1.0) -> list[tuple[str, Path]]:
    """Несколько запусков ОДНОГО видео, окна которых пересекаются (другое окно в UI, другая модель позы, бенчмарк), считать вместе нельзя:
    циклы того же участка записи попадут в датасет и в AUC дважды. Побеждает САМЫЙ РАННИЙ запуск, пересекающиеся с ним более поздние отбрасываются:
    таблицы признаков, метки и оценки VLM привязаны к циклам именно первого запуска (ключ video, tid, start), поэтому ваши пробные запуски в интерфейсе
    не должны подменять основной. Запуски непересекающихся окон того же видео сохраняются. Порядок результата — как у входа."""
    keep: set[Path] = set()
    by_video: dict[str, list[tuple[str, Path]]] = {}
    for v, rd in runs:
        by_video.setdefault(v, []).append((v, rd))
    for group in by_video.values():
        kept: list[tuple[float, float]] = []
        for v, rd in sorted(group, key=lambda r: (run_created(r[1]), r[1].name)):
            a, b = run_window(rd)
            if all(min(b, kb) - max(a, ka) < min_overlap_sec for ka, kb in kept):
                kept.append((a, b))
                keep.add(rd)
    return [(v, rd) for v, rd in runs if rd in keep]


def run_dir(video: str | Path, cfg: dict, start: float = 0.0, end: float | None = None, create: bool = True) -> Path:
    d = OUTPUTS / "runs" / video_id(video) / run_id_for(cfg, start, end)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _feat_hash(cfg: dict) -> str:
    return stable_hash({"f": cfg["features"], "kp": kp_threshold(cfg)})


def _cyc_hash(cfg: dict) -> str:
    return stable_hash({"f": cfg["features"], "kp": kp_threshold(cfg), "c": cfg["cycles"]})


def _ev_hash(cfg: dict) -> str:
    return stable_hash({"c": _cyc_hash(cfg), "e": cfg["events"], "min_h": cfg["video"]["min_person_height_px"]})


# ---------------------------------------------------------------------------------------------- pose + tracking
def stage_pose(video: str | Path, cfg: dict, start: float = 0.0, end: float | None = None, force: bool = False,
               progress: Progress = None, max_frames: int | None = None) -> tuple[Tracks, Path]:
    rd = run_dir(video, cfg, start, end)
    pose_dir = rd / "pose"
    if (pose_dir / "tracks.parquet").exists() and not force:
        return Tracks.load(pose_dir), rd

    info = probe(video)
    stride = choose_stride(info.fps, cfg["video"]["process_fps"])
    proc_fps = info.fps / stride
    pt = PoseTracker(cfg, proc_fps, (info.height, info.width))
    t_end = end if end is not None else info.duration
    n_expected = max(1, int((t_end - start) * proc_fps))
    rows, kps, frame_t, raw = [], [], [], []
    t_wall = time.perf_counter()
    warmup = 0.0
    for i, fr in enumerate(iter_frames(video, start, end, stride, max_frames)):
        frame_t.append((fr.idx, fr.t))
        for d in pt.step(fr.img, fr.t):
            rows.append((fr.idx, fr.t, d.tid, *d.box.tolist(), d.score, float(d.box[3] - d.box[1])))
            kps.append(d.kp)
        for b in pt.last_raw:   # сырые детекции до трекера — для переигрывания трекеров без повторного инференса
            raw.append((fr.idx, fr.t, *b.tolist()))
        if i == 0:
            warmup = time.perf_counter() - t_wall   # первый кадр: компиляция/загрузка ядер — в установившийся fps не входит
        if progress:
            progress(i + 1, n_expected)
    wall = time.perf_counter() - t_wall
    df = pd.DataFrame(rows, columns=["frame", "t", "tid", "x1", "y1", "x2", "y2", "score", "h"])
    kp = np.stack(kps).astype(np.float32) if kps else np.zeros((0, 17, 3), np.float32)
    tr = Tracks(df, kp, pd.DataFrame(frame_t, columns=["frame", "t"]),
                dict(video=str(video), video_info=info.to_dict(), stride=stride, proc_fps=proc_fps, start=start, end=end,
                     pose=pt.describe(), wall_sec=round(wall, 2), frames_processed=len(frame_t), warmup_sec=round(warmup, 1),
                     fps_wall=round(len(frame_t) / wall, 2) if wall > 0 else None,
                     fps_steady=round((len(frame_t) - 1) / (wall - warmup), 2) if wall - warmup > 0 and len(frame_t) > 1 else None,
                     ms=dict(total=_stats(pt.timing["total"]), infer=_stats(pt.timing["infer"]), refine=_stats(pt.timing["refine"])),
                     raw_tracks=int(df.tid.nunique()) if len(df) else 0))
    tr = stitch_tracks(tr, cfg["tracking"]) if len(df) else tr
    tr.save(pose_dir)
    pd.DataFrame(raw, columns=["frame", "t", "x1", "y1", "x2", "y2", "conf"]).to_parquet(pose_dir / "raw_dets.parquet", index=False)
    (rd / "run.json").write_text(json.dumps(dict(cfg=_plain(cfg), meta=tr.meta), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return tr, rd


def _stats(x: list[float]) -> dict:
    if not x:
        return {}
    a = np.asarray(x)
    return dict(mean=round(float(a.mean()), 1), median=round(float(np.median(a)), 1), p95=round(float(np.percentile(a, 95)), 1), n=len(a))


def _plain(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg, default=str))


# ---------------------------------------------------------------------------------------------- features
def stage_features(tr: Tracks, cfg: dict, rd: Path, force: bool = False) -> pd.DataFrame:
    d = rd / "features" / _feat_hash(cfg)
    f = d / "series.parquet"
    if f.exists() and not force:
        cached = pd.read_parquet(f)
        return cached if "tid" in cached.columns else _empty_series()      # кэш запуска без людей, записанный до исправления: пустая таблица без колонок
    parts = []
    for tid in tr.tids:
        rows, kp = tr.of(tid)
        grid = track_grid(tr.frame_t, int(rows.frame.min()), int(rows.frame.max()))
        s = build_series(grid, rows, kp, cfg)
        s.insert(0, "tid", tid)
        parts.append(s)
    out = pd.concat(parts, ignore_index=True) if parts else _empty_series()
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f, index=False)
    return out


def _empty_series() -> pd.DataFrame:
    """Запуск без единого человека (пустой кадр, сцена без людей): таблица без строк, но с ключевыми колонками — этапы ниже работают без ветвлений."""
    return pd.DataFrame({"tid": pd.Series(dtype="int64"), "frame": pd.Series(dtype="int64"), "t": pd.Series(dtype="float64")})


# ---------------------------------------------------------------------------------------------- cycles
def stage_cycles(series: pd.DataFrame, cfg: dict, rd: Path, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = rd / "cycles" / _cyc_hash(cfg)
    if (d / "cycles.parquet").exists() and not force:
        return pd.read_parquet(d / "cycles.parquet"), pd.read_parquet(d / "rejected.parquet"), pd.read_parquet(d / "states.parquet")
    cyc_rows, rej_rows, st_rows = [], [], []
    for tid, s in (series.groupby("tid") if "tid" in series.columns else []):
        s = s.reset_index(drop=True)
        cycles, rejected, states = find_cycles(s, cfg, tid=int(tid))
        for c in cycles:
            cyc_rows.append(c.to_dict())
        for r in rejected:
            rej_rows.append(dict(tid=r.tid, t0=r.t0, t1=r.t1, reason=r.reason, hold=r.hold))
        st_rows.append(pd.DataFrame(dict(tid=int(tid), frame=s.frame.values, t=s.t.values, state=states)))
    cyc = pd.DataFrame(cyc_rows, columns=["tid", "start", "mouth_in", "mouth_out", "end", "hold", "hand", "d_min", "peak_t"])
    rej = pd.DataFrame(rej_rows, columns=["tid", "t0", "t1", "reason", "hold"])
    st = pd.concat(st_rows, ignore_index=True) if st_rows else pd.DataFrame(columns=["tid", "frame", "t", "state"])
    d.mkdir(parents=True, exist_ok=True)
    cyc.to_parquet(d / "cycles.parquet", index=False)
    rej.to_parquet(d / "rejected.parquet", index=False)
    st.to_parquet(d / "states.parquet", index=False)
    return cyc, rej, st


# ---------------------------------------------------------------------------------------------- events
def track_quality(series: pd.DataFrame, tid: int) -> float:
    s = series[series.tid == tid]
    return float(np.clip(visibility_fraction(s) / 0.8, 0, 1)) if len(s) else 0.0


def assemble_events(tr: Tracks, series: pd.DataFrame, cycles: pd.DataFrame, cfg: dict, camera_id: str, clip_id: str,
                    cycle_scores: dict | None = None) -> tuple[pd.DataFrame, list[Event]]:
    """Циклы -> события по регламенту (чистая функция, на диск ничего не пишет). `cycle_scores` {(tid, round(start, 3)): score} от классификатора
    (или от разметки, для проверки сборки на «эталоне из меток»); иначе — эвристика-заглушка по длительности паузы."""
    summ = tr.summary(cfg["video"]["min_person_height_px"])
    small = set(summ[summ.ignore_small].tid.tolist())
    events: list[Event] = []
    for tid, g in cycles.groupby("tid"):
        if int(tid) in small:
            continue  # IGNORE: человек ниже min_person_height_px — события не выдаём
        q = track_quality(series, int(tid))
        sc = [ScoredCycle(Cycle(int(r.tid), r.start, r.mouth_in, r.mouth_out, r.end, r.hold, int(r.hand), r.d_min),
                          score=(cycle_scores or {}).get((int(r.tid), round(float(r.start), 3)),
                                                         heuristic_cycle_score(Cycle(int(r.tid), r.start, r.mouth_in, r.mouth_out, r.end, r.hold, int(r.hand), r.d_min), q)))
              for r in g.itertuples()]
        events += build_events(sc, cfg["events"], tid=int(tid), quality=q)
    boxes = {t: (lambda tt, _t=t: tr.box_at(_t, tt)) for t in tr.tids}
    return events_to_frame(events, camera_id, clip_id, boxes), events


def stage_events(tr: Tracks, series: pd.DataFrame, cycles: pd.DataFrame, cfg: dict, rd: Path, camera_id: str, clip_id: str,
                 cycle_scores: dict | None = None, force: bool = False) -> tuple[pd.DataFrame, list[Event]]:
    """Этап 6: `assemble_events` + запись events.csv в каталог запуска (по хэшу параметров)."""
    d = rd / "events" / _ev_hash(cfg)
    df, events = assemble_events(tr, series, cycles, cfg, camera_id, clip_id, cycle_scores)
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "events.csv", index=False)
    return df, events
