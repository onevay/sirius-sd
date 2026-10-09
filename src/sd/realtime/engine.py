"""Потоковый движок событий одной камеры.

    eng = StreamEngine(cfg, camera_id, pose_fn)
    for img, t in frames: updates = eng.step(img, t)     # t — секунды потока; updates — новые/обновлённые/закрытые события
    updates += eng.flush(t_end)

Как работает (и почему так): держим буфер последних `buffer_sec` секунд детекций (рамки + ключевые точки), раз в `eval_every_sec` пересчитываем на нём ряд признаков и автомат
циклов тем же кодом, что offline (`features.build_series`, `cycles.find_cycles`), берём только ЗАКРЫТЫЕ и «осевшие» циклы (конец старше `settle_sec`), оцениваем их и собираем события по регламенту
(`events.build_events`). Событие поднимается тревогой, когда выполнено правило (два цикла за 20 с или цикл + признак) и confidence ≥ порога; начало события — задним числом (начало подъёма руки
в первом цикле), поэтому метрика «задержка» остаётся малой, хотя сама тревога приходит позже — в момент подтверждения.

Чего здесь нет сознательно: склейки треков постфактум (в потоке её нет — смена ID у человека порождает отдельное событие; см. docs/REALTIME.md) и тяжёлых признаков (предмет, VLM) — они подключаются
через `scorer`. Модель позы подставляется параметром `pose_fn(img, t) -> list[Det]` — в тестах это синтетика, в работе `PoseTracker.step`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..cycles import Cycle, find_cycles
from ..events import Event, ScoredCycle, build_events, heuristic_cycle_score
from ..features import build_series, track_grid, visibility_fraction
from ..tracks import Tracks

PoseFn = Callable[[np.ndarray, float], list]


@dataclass
class AlertUpdate:
    """Изменение события: `open` — новая тревога, `update` — событие продлено или изменилась уверенность, `close` — событие завершено (пауза > merge_gap)."""
    kind: str
    key: str
    camera_id: str
    tid: int
    start: float
    end: float
    peak: float
    confidence: float
    rule: str
    n_cycles: int
    explain: str
    box: tuple[float, float, float, float] | None
    t_now: float                       # момент потока, когда это стало известно (задержка тревоги = t_now − start)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["box"] = list(self.box) if self.box else None
        return d


class Scorer(Protocol):
    name: str

    def score(self, cycles: pd.DataFrame, tr: Tracks, ser: pd.DataFrame) -> dict[tuple[int, float], float]:
        """{(tid, round(start, 3)): оценка цикла ∈ [0, 1]}."""


class HeuristicScorer:
    """Эвристика по длительности паузы (как offline без классификатора). Не отличает питьё/телефон/еду — только для отладки конвейера."""
    name = "heuristic"

    def score(self, cycles, tr, ser):
        out = {}
        for r in cycles.itertuples():
            q = float(np.clip(visibility_fraction(ser[ser.tid == r.tid]) / 0.8, 0, 1))
            out[(int(r.tid), round(float(r.start), 3))] = heuristic_cycle_score(Cycle(int(r.tid), r.start, r.mouth_in, r.mouth_out, r.end, r.hold, int(r.hand), r.d_min), q)
        return out


class BundleScorer:
    """Обученный пакет классификатора цикла на «быстрых» признаках (кинематика, ритм, положение точек) — без предмета, фото-модели и VLM, поэтому годится для потока."""
    name = "bundle"

    def __init__(self, bundle_dir: str | Path, cfg: dict, camera_id: str = "cam"):
        from ..bundle import Bundle

        self.b, self.cfg, self.cam = Bundle(Path(bundle_dir)), cfg, camera_id

    def score(self, cycles, tr, ser):
        from ..dataset import build_cycle_table
        from ..pipeline import fast_cycle_features
        from ..pose_feats import pose_rows

        tab = build_cycle_table(Path(self.cam), tr, ser, cycles)
        if tab.empty:
            return {}
        feats = fast_cycle_features(tab, pd.DataFrame(pose_rows(tab.video.iloc[0], tr, self.cfg, ser, cycles)))
        sc = self.b.score(feats)
        return {(int(t), round(float(s), 3)): float(v) for t, s, v in zip(feats.tid, feats.start, sc)}


class FrameRing:
    """Кольцо последних кадров (JPEG, уменьшенные) для миниатюры тревоги и клипа вокруг события; память ограничена `seconds`."""

    def __init__(self, seconds: float = 45.0, fps: float = 4.0, width: int = 640):
        self.seconds, self.step, self.width = seconds, 1.0 / fps, width
        self.items: list[tuple[float, bytes]] = []
        self.scale = 1.0
        self._last = -1e9

    def add(self, img: np.ndarray, t: float) -> None:
        if t - self._last < self.step - 1e-6:
            return
        import cv2

        self._last = t
        h, w = img.shape[:2]
        self.scale = min(1.0, self.width / w)
        small = cv2.resize(img, (int(w * self.scale), int(h * self.scale))) if self.scale < 1.0 else img
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.items.append((t, buf.tobytes()))
        while self.items and t - self.items[0][0] > self.seconds:
            self.items.pop(0)

    def nearest(self, t: float) -> tuple[float, bytes] | None:
        return min(self.items, key=lambda x: abs(x[0] - t)) if self.items else None

    def thumbnail(self, t: float, box: tuple | None, label: str | None = None) -> bytes | None:
        """JPEG кадра у `t` с рамкой человека (рамка — в пикселях исходного кадра)."""
        import cv2

        it = self.nearest(t)
        if it is None:
            return None
        img = cv2.imdecode(np.frombuffer(it[1], np.uint8), cv2.IMREAD_COLOR)
        if box is not None:
            b = [int(v * self.scale) for v in box]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (60, 60, 255), 3)
            if label:
                cv2.putText(img, label, (b[0] + 4, max(b[1] - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def write_clip(self, t0: float, t1: float, path: str | Path, fps: float | None = None) -> bool:
        """MP4 из кадров кольца в [t0, t1]; False, если кадров нет. Кодирует через ffmpeg (`video_io.FFmpegWriter`), как остальные видео проекта."""
        import cv2

        from ..video_io import FFmpegWriter

        sel = [x for x in self.items if t0 <= x[0] <= t1]
        if len(sel) < 2:
            return False
        first = cv2.imdecode(np.frombuffer(sel[0][1], np.uint8), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        w, h = w - w % 2, h - h % 2
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FFmpegWriter(path, w, h, fps or 1.0 / self.step) as wr:
            for _, b in sel:
                wr.write(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:h, :w])
        return path.exists() and path.stat().st_size > 0


@dataclass
class _Emitted:
    end: float
    conf: float
    n: int
    closed: bool = False


class StreamEngine:
    def __init__(self, cfg: dict, camera_id: str, pose_fn: PoseFn, scorer: Scorer | None = None, *, frame_hw: tuple[int, int] | None = None, buffer_sec: float = 90.0,
                 eval_every_sec: float = 1.0, settle_sec: float = 1.0, keep_frames: bool = True, ring_sec: float = 45.0, source_path: str | Path | None = None):
        self.cfg, self.camera_id, self.pose_fn = cfg, camera_id, pose_fn
        self.scorer: Scorer = scorer or HeuristicScorer()
        self.buffer_sec, self.eval_every, self.settle = buffer_sec, eval_every_sec, settle_sec
        self.frame_hw = frame_hw
        self.source_path = str(source_path) if source_path else None      # исходный файл (имитация потока): нужен признакам предмета/фото; в настоящем потоке None
        self.ring = FrameRing(ring_sec) if keep_frames else None
        self._rows: list[tuple] = []          # frame, t, tid, x1, y1, x2, y2, score, h
        self._kp: list[np.ndarray] = []
        self._frames: list[tuple[int, float]] = []
        self._n = 0
        self._last_eval = -1e9
        self._seen: dict[int, set[float]] = {}
        self._scored: dict[int, list[ScoredCycle]] = {}
        self._emitted: dict[str, _Emitted] = {}
        self.threshold = float(cfg["events"]["confidence_threshold"])
        self.stats = dict(frames=0, evals=0, cycles=0)
        self.last_dets: list[tuple[int, list[float]]] = []
        self.cycle_log: list[dict] = []        # каждый оценённый цикл: оценка классификатора и сигналы (предмет, фото, fusion) — для вывода «почему сработало»

    # ------------------------------------------------------------------ вход
    def step(self, img: np.ndarray | None, t: float) -> list[AlertUpdate]:
        """Один обработанный кадр потока. `img=None` допустим для тестов с готовыми детекциями (тогда `pose_fn` получает None)."""
        if img is not None and self.frame_hw is None:
            self.frame_hw = (int(img.shape[0]), int(img.shape[1]))
        dets = self.pose_fn(img, t)
        self.last_dets = [(int(d.tid), [float(x) for x in d.box]) for d in dets]        # для отрисовки поверх кадра (replay)
        for d in dets:
            self._rows.append((self._n, t, int(d.tid), *[float(x) for x in d.box], float(d.score), float(d.box[3] - d.box[1])))
            self._kp.append(np.asarray(d.kp, np.float32))
        self._frames.append((self._n, t))
        self._n += 1
        self.stats["frames"] += 1
        if self.ring is not None and img is not None:
            self.ring.add(img, t)
        if t - self._last_eval >= self.eval_every:
            self._last_eval = t
            return self._evaluate(t, self.settle)
        return []

    def flush(self, t_end: float) -> list[AlertUpdate]:
        """Конец потока: оценка без ожидания «оседания»; все открытые события закрываются."""
        out = self._evaluate(t_end, 0.0, final=True)
        return out

    # ------------------------------------------------------------------ расчёт
    def _tracks(self) -> Tracks | None:
        if not self._rows:
            return None
        df = pd.DataFrame(self._rows, columns=["frame", "t", "tid", "x1", "y1", "x2", "y2", "score", "h"])
        # размер кадра нужен признакам цикла (положение на кадре, расстояние до края); без кадров (тесты, готовые детекции) оцениваем по рамкам
        h, w = self.frame_hw if self.frame_hw else (int(df.y2.max()) + 1, int(df.x2.max()) + 1)
        vi = dict(height=int(h), width=int(w))
        return Tracks(df, np.stack(self._kp), pd.DataFrame(self._frames, columns=["frame", "t"]), dict(video_info=vi, **({"video": self.source_path} if self.source_path else {})))

    def _trim(self, t_now: float) -> None:
        cut = t_now - self.buffer_sec
        if not self._frames or self._frames[0][1] >= cut:
            return
        keep = [i for i, r in enumerate(self._rows) if r[1] >= cut]
        self._rows, self._kp = [self._rows[i] for i in keep], [self._kp[i] for i in keep]
        self._frames = [f for f in self._frames if f[1] >= cut]
        # состояние событий тоже не должно расти сутками: забываем то, что давно закрыто и старше буфера (такие циклы уже не появятся в пересчёте)
        old = t_now - max(self.buffer_sec, 300.0) - float(self.cfg["events"]["merge_gap_sec"])
        for tid in list(self._scored):
            self._scored[tid] = [c for c in self._scored[tid] if c.cycle.end >= old]
            if not self._scored[tid]:
                del self._scored[tid]
        for tid in list(self._seen):
            self._seen[tid] = {k for k in self._seen[tid] if k >= old}
        for key in [k for k, e in self._emitted.items() if e.closed and e.end < old]:
            del self._emitted[key]

    def _evaluate(self, t_now: float, settle: float, final: bool = False) -> list[AlertUpdate]:
        self.stats["evals"] += 1
        tr = self._tracks()
        updates: list[AlertUpdate] = []
        if tr is not None:
            cfg = self.cfg
            min_h = cfg["video"]["min_person_height_px"]
            buf_start = float(tr.frame_t.t.min())
            summ = tr.summary(min_h).set_index("tid")
            sers, new_cycles = [], []
            for tid in tr.tids:
                if bool(summ.loc[tid, "ignore_small"]):
                    continue        # IGNORE: человек ниже min_person_height_px — событий не выдаём
                rows, kp = tr.of(tid)
                s = build_series(track_grid(tr.frame_t, int(rows.frame.min()), int(rows.frame.max())), rows, kp, cfg)
                s.insert(0, "tid", tid)
                sers.append(s)
                cycles, _, _ = find_cycles(s.reset_index(drop=True), cfg, tid=int(tid))
                seen = self._seen.setdefault(int(tid), set())
                for c in cycles:
                    key = round(float(c.start), 2)
                    cut = c.start < buf_start + 1.5 and buf_start > 0      # цикл обрезан началом буфера — доверять нельзя
                    if key in seen or c.end > t_now - settle or cut:
                        continue
                    seen.add(key)
                    new_cycles.append(c.to_dict())
            if new_cycles:
                ser = pd.concat(sers, ignore_index=True)
                cdf = pd.DataFrame(new_cycles, columns=["tid", "start", "mouth_in", "mouth_out", "end", "hold", "hand", "d_min", "peak_t"])
                scores = self.scorer.score(cdf, tr, ser)
                self.stats["cycles"] += len(cdf)
                det = getattr(self.scorer, "last_details", {}) or {}
                for r in cdf.itertuples():
                    sc = scores.get((int(r.tid), round(float(r.start), 3)), 0.0)
                    self.cycle_log.append(dict(tid=int(r.tid), start=round(float(r.start), 2), end=round(float(r.end), 2), peak=round(float(r.peak_t), 2), hold=round(float(r.hold), 2),
                                               score=round(float(sc), 3), **det.get((int(r.tid), round(float(r.start), 3)), {})))
                    self._scored.setdefault(int(r.tid), []).append(ScoredCycle(Cycle(int(r.tid), r.start, r.mouth_in, r.mouth_out, r.end, r.hold, int(r.hand), r.d_min), score=float(sc)))
                for tid in {int(x) for x in cdf.tid}:
                    q = float(np.clip(visibility_fraction(ser[ser.tid == tid]) / 0.8, 0, 1))
                    for ev in build_events(sorted(self._scored[tid], key=lambda c: c.cycle.start), cfg["events"], tid=tid, quality=q):
                        u = self._event_update(ev, tr, t_now)
                        if u:
                            updates.append(u)
            updates += self._close_old(t_now, final)
        self._trim(t_now)
        return updates

    def _event_update(self, ev: Event, tr: Tracks, t_now: float) -> AlertUpdate | None:
        key = f"{self.camera_id}:{ev.tid}:{ev.start:.2f}"
        prev = self._emitted.get(key)
        if ev.confidence < self.threshold:
            return None
        if prev and abs(prev.end - ev.end) < 1e-6 and abs(prev.conf - ev.confidence) < 0.01 and prev.n == ev.n_cycles:
            return None
        kind = "update" if prev else "open"
        self._emitted[key] = _Emitted(ev.end, ev.confidence, ev.n_cycles, prev.closed if prev else False)
        return AlertUpdate(kind, key, self.camera_id, ev.tid, float(ev.start), float(ev.end), float(ev.peak), float(ev.confidence), ev.rule, ev.n_cycles, ev.explain(),
                           tr.box_at(ev.tid, ev.peak), float(t_now))

    def _close_old(self, t_now: float, final: bool) -> list[AlertUpdate]:
        gap = float(self.cfg["events"]["merge_gap_sec"])
        out = []
        for key, e in self._emitted.items():
            if not e.closed and (final or t_now - e.end > gap):
                e.closed = True
                cam, tid, start = key.rsplit(":", 2)
                out.append(AlertUpdate("close", key, cam, int(tid), float(start), e.end, e.end, e.conf, "", e.n, "", None, float(t_now)))
        return out

    # ------------------------------------------------------------------ артефакты тревоги
    def thumbnail(self, u: AlertUpdate) -> bytes | None:
        return self.ring.thumbnail(u.peak, u.box, f"{u.confidence:.0%}") if self.ring else None

    def save_clip(self, u: AlertUpdate, path: str | Path, pre: float = 5.0, post: float = 5.0) -> bool:
        return bool(self.ring and self.ring.write_clip(u.start - pre, min(u.end + post, u.t_now), path))
