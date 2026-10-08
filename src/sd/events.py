"""Сборка событий по регламенту (руководство §4.4, шаг 7, §8.2).

Событие = эпизод ОДНОГО person_track_id. Циклы одного трека склеиваются при паузе <= merge_gap (15 с); разных людей не склеиваем никогда.
Эпизод засчитан, если в окне 20 с есть два полных цикла ИЛИ один цикл + прямой признак (предмет в >= k из n кадров / дым / ответ VLM).
Начало = начало подъёма в первом засчитанном цикле, конец = отведение руки после последнего цикла.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cycles import Cycle


@dataclass
class ScoredCycle:
    cycle: Cycle
    score: float = 0.5
    has_object: bool = False
    has_smoke: bool = False
    object_conf: float = 0.0


@dataclass
class Event:
    tid: int
    start: float
    end: float
    peak: float
    confidence: float
    rule: str                      # "two_cycles" | "cycle+object" | "both"
    n_cycles: int
    cycles: list[ScoredCycle] = field(default_factory=list)

    def explain(self) -> str:
        """Человекочитаемое объяснение тревоги (для оператора)."""
        span = self.cycles[-1].cycle.end - self.cycles[0].cycle.start if self.cycles else 0.0
        parts = [f"{self.n_cycles} цикл(а) за {span:.0f} с"]
        if any(c.has_object for c in self.cycles):
            parts.append("предмет у рта")
        if any(c.has_smoke for c in self.cycles):
            parts.append("дым/пар")
        return ", ".join(parts)


def heuristic_cycle_score(cyc: Cycle, quality: float = 1.0) -> float:
    """ЗАГЛУШКА до обучения классификатора (бейзлайн «только правила регламента»).

    Правдоподобие длительности паузы у рта: плато 1–4 с, спад к границам 0.5 и 5 с; умножается на качество наблюдения [0,1].
    Не различает питьё/телефон/еду — для этого и нужен бустинг на векторе признаков (раздел 4 руководства).
    """
    h = cyc.hold
    if h < 1.0:
        p = 0.5 + 0.5 * (h - 0.5) / 0.5
    elif h <= 4.0:
        p = 1.0
    else:
        p = 1.0 - 0.5 * (h - 4.0) / 1.0
    return float(np.clip(p * quality, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def episode_confidence(group: list[ScoredCycle], quality: float = 1.0, w: dict | None = None) -> float:
    """Монотонная функция признаков эпизода. Веса — заглушка; после калибровки (Platt/isotonic) заменяются обученными."""
    w = w or dict(bias=-2.0, cycles=0.9, score=2.0, obj=1.2, quality=0.8)
    top2 = sorted((g.score for g in group), reverse=True)[:2]
    has_obj = any(g.has_object or g.has_smoke for g in group)
    z = (w["bias"] + w["cycles"] * min(len(group), 4) + w["score"] * (float(np.mean(top2)) - 0.5)
         + w["obj"] * has_obj + w["quality"] * (quality - 0.5))
    return _sigmoid(z)


def build_events(cycles: list[ScoredCycle], cfg_events: dict, tid: int | None = None, quality: float = 1.0) -> list[Event]:
    """Циклы ОДНОГО трека -> события. Если передать циклы нескольких треков — бросим ошибку (никогда не склеиваем людей)."""
    if not cycles:
        return []
    tids = {c.cycle.tid for c in cycles}
    if len(tids) > 1:
        raise ValueError(f"build_events получил циклы нескольких треков {sorted(tids)}: людей склеивать нельзя")
    tid = next(iter(tids)) if tid is None else tid
    e = cfg_events
    good = sorted((c for c in cycles if c.score >= e["cycle_th"]), key=lambda c: c.cycle.start)
    groups: list[list[ScoredCycle]] = []
    cur: list[ScoredCycle] = []
    for c in good:
        if cur and c.cycle.start - cur[-1].cycle.end > e["merge_gap_sec"]:
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)

    events: list[Event] = []
    for g in groups:
        if e.get("window_mode", "contain") == "contain":   # оба цикла целиком внутри окна 20 с
            two = any(b.cycle.end - a.cycle.start <= e["window_sec"] for a, b in zip(g, g[1:]))
        else:                                               # как в примере руководства: старт–старт
            two = any(b.cycle.start - a.cycle.start <= e["window_sec"] for a, b in zip(g, g[1:]))
        obj = any(c.has_object or c.has_smoke for c in g)
        if not (two or obj):
            continue
        peak_c = max(g, key=lambda c: c.score)
        events.append(Event(
            tid=tid, start=g[0].cycle.start, end=g[-1].cycle.end, peak=peak_c.cycle.peak_t,
            confidence=episode_confidence(g, quality), rule="both" if (two and obj) else ("two_cycles" if two else "cycle+object"),
            n_cycles=len(g), cycles=g))
    return events


def events_to_frame(events: list[Event], camera_id: str, clip_id: str, boxes: dict[int, "callable"] | None = None, frame_wh: tuple[int, int] | None = None) -> pd.DataFrame:
    """Таблица в формате организаторов (руководство §10.1).

    boxes[tid](t) -> (x1,y1,x2,y2) рамка ВСЕГО человека на кадре, ближайшем к peak_sec, в пикселях исходного кадра.
    frame_wh=(ширина, высота) — рамка обрезается по кадру и округляется до целых пикселей (§10.1: x1 < x2, y1 < y2, внутри кадра).
    """
    rows = []
    for i, ev in enumerate(sorted(events, key=lambda x: x.start), 1):
        box = boxes[ev.tid](ev.peak) if boxes and ev.tid in boxes else None
        box = box if box is not None else (np.nan,) * 4
        if frame_wh and np.isfinite(box).all():
            w, h = frame_wh
            x1, y1 = max(0.0, min(box[0], w - 2)), max(0.0, min(box[1], h - 2))
            x2, y2 = min(float(w), max(box[2], x1 + 1)), min(float(h), max(box[3], y1 + 1))
            box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        rows.append(dict(camera_id=camera_id, clip_id=clip_id, event_id=f"{camera_id}_{clip_id}_{i:04d}",
                         start_sec=round(ev.start, 2), end_sec=round(ev.end, 2), confidence=round(ev.confidence, 3),
                         label="smoking_like", person_track_id=ev.tid, peak_sec=round(ev.peak, 2),
                         x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
    cols = ["camera_id", "clip_id", "event_id", "start_sec", "end_sec", "confidence", "label", "person_track_id",
            "peak_sec", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(rows, columns=cols)
