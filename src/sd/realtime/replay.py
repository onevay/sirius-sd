"""Имитация реального времени по видеофайлу: файл идёт через ТОТ ЖЕ потоковый движок, что и камеры, — кадр за кадром, в заданном темпе, с тревогами в момент их появления.

Два режима использования:
  * «просто решение модели»: нет эталона — на выходе тревоги (когда сработали, о ком, с какой уверенностью), видео с рамками и баннером тревоги, пропускная способность;
  * «с разметкой и выводом»: есть эталон событий клипа (`labels/events_gt.csv`) — дополнительно TP/FP/FN, precision/recall/F1 и **задержка тревоги** (сколько секунд прошло от начала курения
    по разметке до тревоги) — метрика, которой нет в offline-оценке, а для мониторинга она важна не меньше F1.

Пропускная способность — главное, что нужно знать до развёртывания: `rt_factor` = секунд видео / секунд ВЫЧИСЛЕНИЙ (без пауз темпа). Больше 1 — камера обрабатывается быстрее реального времени, число одновременных
камер на этом устройстве ≈ целая часть `rt_factor` (при последовательной обработке); меньше 1 — поток отстаёт, снижайте `video.process_fps`, `pose.imgsz`, отключайте уточнение или тяжёлые признаки.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from .. import evaluation as EV
from .. import gt as GT
from ..evaluate import evaluate
from ..paths import OUTPUTS, video_id
from ..profiles import Profile
from .engine import AlertUpdate, StreamEngine
from .sources import Pacer

REPLAY_DIR = OUTPUTS / "replay"
EVENT_COLUMNS = ["camera_id", "clip_id", "event_id", "start_sec", "end_sec", "confidence", "label", "person_track_id", "peak_sec", "x1", "y1", "x2", "y2"]


@dataclass
class ReplayResult:
    video: str
    clip_id: str
    profile: str
    window: tuple[float, float | None]
    alerts: pd.DataFrame                    # по строке на событие: когда сработала тревога и что с ней стало
    events: pd.DataFrame                    # финальные события в формате организаторов
    log: pd.DataFrame                       # каждое обновление движка (open/update/close)
    stats: dict
    metrics: dict | None = None
    report: EV.Report | None = None
    out_dir: str | None = None
    overlay: str | None = None
    notes: list[str] = field(default_factory=list)
    trace: dict | None = None               # рамки людей по времени + тревоги: браузер рисует их поверх исходного видео (без перекодирования)


def _frames(video, start, end, fps) -> Iterator[tuple[np.ndarray, float]]:
    from ..video_io import choose_stride, iter_frames, probe

    info = probe(video)
    for fr in iter_frames(video, start, end, choose_stride(info.fps, fps)):
        yield fr.img, fr.t


def _annotate(img: np.ndarray, dets: list, active: list[AlertUpdate], gt_label: str | None, t: float, max_w: int = 960) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    k = min(1.0, max_w / w)
    out = cv2.resize(img, (int(w * k), int(h * k))) if k < 1.0 else img.copy()
    hot = {u.tid for u in active}
    for tid, b in dets:
        x1, y1, x2, y2 = [int(v * k) for v in b]
        col = (60, 60, 255) if tid in hot else (240, 240, 240)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if tid in hot else 1)
        cv2.putText(out, f"ID {tid}", (x1 + 3, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    if active:
        u = max(active, key=lambda a: a.confidence)
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (30, 30, 200), -1)
        cv2.putText(out, f"TREVOGA  ID {u.tid} - {u.explain} - {u.confidence:.0%}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if gt_label:
        cv2.putText(out, f"GT: {gt_label}", (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 60), 2)
    cv2.putText(out, f"{t:6.1f}s", (out.shape[1] - 90, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def _box_row(u: AlertUpdate, frame_hw) -> tuple:
    b = u.box or (np.nan,) * 4
    if frame_hw and np.isfinite(b).all():
        h, w = frame_hw
        x1, y1 = max(0.0, min(b[0], w - 2)), max(0.0, min(b[1], h - 2))
        x2, y2 = min(float(w), max(b[2], x1 + 1)), min(float(h), max(b[3], y1 + 1))
        return tuple(int(round(v)) for v in (x1, y1, x2, y2))
    return tuple(b)


def replay_video(video: str | Path, profile: Profile, *, start: float = 0.0, end: float | None = None, speed: float = 0.0, gt: pd.DataFrame | None = None,
                 pose_fn: Callable | None = None, engine: StreamEngine | None = None, process_fps: float | None = None, render: bool = False, on_alert: Callable | None = None,
                 progress: Callable[[float, float], None] | None = None, on_frame: Callable | None = None, frames_fn: Callable | None = None, probe_duration: Callable | None = None, out_root: Path | None = None,
                 save: bool = True) -> ReplayResult:
    """Прогон файла через потоковый движок. `speed`: 0 — как можно быстрее, 1 — реальное время, N — в N раз быстрее. `gt` — таблица эталона (весь файл, фильтруется по клипу)."""
    from .worker import make_engine

    video = Path(video)
    cid = video_id(video)
    fps = process_fps or float(profile.cfg()["video"]["process_fps"])
    eng = engine or make_engine(profile, cid, fps, pose_fn=pose_fn, keep_frames=False, mode="replay", source_path=video)
    frames = (frames_fn or _frames)(video, start, end, fps)
    pacer = Pacer(speed)
    cid_gt = gt[gt["clip_id"] == cid] if gt is not None and len(gt) else None
    gt_pos = cid_gt[cid_gt.label == "POSITIVE"] if cid_gt is not None else None
    state: dict[str, dict] = {}
    log: list[dict] = []
    ms: list[float] = []
    n_frames, t_first, t_last = 0, None, None
    trace_frames: list = []
    writer, overlay_path, active_keys = None, None, set()

    def handle(ups: Iterable[AlertUpdate]) -> None:
        for u in ups:
            log.append(dict(t_now=u.t_now, kind=u.kind, key=u.key, tid=u.tid, start=u.start, end=u.end, confidence=u.confidence, explain=u.explain))
            s = state.setdefault(u.key, dict(key=u.key, tid=u.tid, t_open=None, closed=False))
            if u.kind == "open":
                s["t_open"] = u.t_now
                active_keys.add(u.key)
            if u.kind == "close":
                s["closed"] = True
                active_keys.discard(u.key)
            if u.kind in ("open", "update"):
                s.update(start=u.start, end=u.end, peak=u.peak, confidence=u.confidence, rule=u.rule, n_cycles=u.n_cycles, explain=u.explain, box=_box_row(u, eng.frame_hw), last=u)
            if on_alert:
                on_alert(u)

    if render:
        from ..video_io import FFmpegWriter

        out_base = out_root or REPLAY_DIR
        run_dir = out_base / f"{time.strftime('%Y%m%d_%H%M%S')}_{cid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = run_dir / "overlay.mp4"
    for img, t in frames:
        if t_first is None:
            t_first = t
        t_last = t
        pacer.wait(t)
        t0 = time.perf_counter()
        ups = eng.step(img, t)
        handle(ups)
        if render and img is not None:
            if writer is None:
                probe = _annotate(img, eng.last_dets, [], None, t)
                writer = FFmpegWriter(overlay_path, probe.shape[1] - probe.shape[1] % 2, probe.shape[0] - probe.shape[0] % 2, fps)
            act = [state[k]["last"] for k in active_keys if "last" in state[k] and state[k]["last"].start <= t]
            gl = None
            if gt_pos is not None and len(gt_pos[(gt_pos.start_sec <= t) & (gt_pos.end_sec >= t)]):
                gl = "курение"
            fr = _annotate(img, eng.last_dets, act, gl, t)
            writer.write(fr[: fr.shape[0] - fr.shape[0] % 2, : fr.shape[1] - fr.shape[1] % 2])
        ms.append((time.perf_counter() - t0) * 1000)
        row = [round(t, 3), [[int(tid), *(round(float(v), 1) for v in b)] for tid, b in eng.last_dets]]
        trace_frames.append(row)
        if on_frame:
            on_frame(row, eng.frame_hw, eng)     # живой вывод: клиент рисует рамки по мере обработки
        n_frames += 1
        if progress and n_frames % 10 == 0:
            progress(t, end if end is not None else t)
    t_end = (t_last if t_last is not None else start)
    handle(eng.flush(t_end))
    if writer is not None:
        writer.close()
    busy = sum(ms) / 1000.0
    stream_sec = (t_last - t_first) if t_first is not None and t_last is not None else 0.0
    arr = np.array(ms) if ms else np.array([0.0])
    stats = dict(frames=n_frames, stream_sec=round(stream_sec, 2), busy_sec=round(busy, 2), process_fps_target=fps, fps_proc=round(n_frames / busy, 2) if busy > 0 else None,
                 rt_factor=round(stream_sec / busy, 2) if busy > 0 else None, ms_p50=round(float(np.percentile(arr, 50)), 1), ms_p95=round(float(np.percentile(arr, 95)), 1),
                 ms_max=round(float(arr.max()), 1), cycles=int(eng.stats["cycles"]), alerts=int(sum(1 for s in state.values() if s["t_open"] is not None)), speed=speed)
    stats["verdict"] = _verdict(stats)
    rows, evrows = [], []
    for i, s in enumerate(sorted((s for s in state.values() if s["t_open"] is not None), key=lambda s: s["start"]), 1):
        rows.append(dict(key=s["key"], tid=s["tid"], start=s["start"], end=s["end"], peak=s["peak"], confidence=s["confidence"], rule=s["rule"], n_cycles=s["n_cycles"], explain=s["explain"],
                         t_open=s["t_open"], delay=s["t_open"] - s["start"], closed=s["closed"]))
        b = s["box"]
        evrows.append(dict(camera_id=cid, clip_id=cid, event_id=f"{cid}_{i:04d}", start_sec=round(s["start"], 2), end_sec=round(s["end"], 2), confidence=round(s["confidence"], 3), label="smoking_like",
                           person_track_id=s["tid"], peak_sec=round(s["peak"], 2), x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
    alerts = pd.DataFrame(rows, columns=["key", "tid", "start", "end", "peak", "confidence", "rule", "n_cycles", "explain", "t_open", "delay", "closed"])
    events = pd.DataFrame(evrows, columns=EVENT_COLUMNS)
    metrics = report = None
    notes: list[str] = []
    if cid_gt is not None and len(cid_gt):
        win_end = end if end is not None else (t_last or 0.0)
        dur = max(float(win_end) - float(start), 1e-6)
        st = EV.Settings(threshold=profile.threshold)
        gts = [g for g in GT.to_eval_gts(cid_gt, {cid}) if g.end > start and g.start < win_end]
        preds = EV.to_preds(events, cid) if len(events) else []
        report = EV.build_report(preds, gts, {cid: dur}, st, n_boot=0)
        res = evaluate(preds, gts, **st.kw())
        delays = [float(alerts.iloc[pi].t_open - gts[gi].start) for pi, gi in res.matches if pi < len(alerts) and len(preds) == len(alerts)]
        # соответствие индексов предсказаний строкам `alerts`: preds строятся из `events` в том же порядке (по началу), что и `alerts`
        metrics = dict(**{k: report.metrics[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}, alert_delay_median=float(np.median(delays)) if delays else None,
                       alert_delay_max=float(np.max(delays)) if delays else None, n_gt=report.metrics["n_gt"])
    elif gt is not None:
        notes.append("в эталоне нет разметки этого клипа: метрики не считаются")
    cth = float(profile.cfg()["events"]["cycle_th"])
    trace = dict(clip_id=cid, fps=fps, cycles=[dict(c, passed=bool(c["score"] >= cth)) for c in eng.cycle_log], thresholds=dict(cycle=cth, event=float(profile.threshold)), frame_hw=list(eng.frame_hw) if eng.frame_hw else None, frames=trace_frames,
                 alerts=[dict(tid=int(r.tid), start=float(r.start), end=float(r.end), t_open=float(r.t_open), confidence=float(r.confidence), explain=str(r.explain), delay=float(r.delay),
                          rule=str(r.rule), n_cycles=int(r.n_cycles), peak=float(r.peak))
                         for r in alerts.itertuples()],
                 gt=[dict(start=float(g.start_sec), end=float(g.end_sec)) for g in (gt_pos.itertuples() if gt_pos is not None else [])])
    out_dir = None
    if save:
        out_base = out_root or REPLAY_DIR
        out_dir = Path(overlay_path.parent) if overlay_path else out_base / f"{time.strftime('%Y%m%d_%H%M%S')}_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        events.to_csv(out_dir / "events.csv", index=False)
        (out_dir / "trace.json").write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
        alerts.to_csv(out_dir / "alerts.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False, encoding="utf-8-sig")
        (out_dir / "summary.json").write_text(json.dumps(dict(video=str(video), clip_id=cid, profile=profile.name, fingerprint=profile.fingerprint(), window=[start, end], stats=stats, metrics=metrics,
                                                              notes=notes), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return ReplayResult(str(video), cid, profile.name, (start, end), alerts, events, pd.DataFrame(log), stats, metrics, report, str(out_dir) if out_dir else None,
                        str(overlay_path) if overlay_path and overlay_path.exists() else None, notes, trace)


def _verdict(s: dict) -> str:
    rt = s.get("rt_factor")
    if not rt:
        return "нет данных о скорости"
    if rt >= 1.0:
        return f"справляется: видео обрабатывается в {rt:.1f}× быстрее реального времени (≈ {int(rt)} камер(ы) последовательно на этом устройстве)"
    return (f"не справляется: {s['fps_proc']} к/с при нужных {s['process_fps_target']:g} (в {1 / rt:.1f}× медленнее реального времени) — снизьте video.process_fps или pose.imgsz, "
            "отключите уточнение точек")
