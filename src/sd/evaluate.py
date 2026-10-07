"""Наша копия протокола оценки (руководство §9.3). Заменить/сверить, когда организаторы выдадут валидатор.

Реализовано так, как описано в регламенте:
  * эталон расширяется на ±3 с, нужно пересечение предсказания с расширенным эталоном >= 1 с;
  * совпадение по рамке человека: IoU >= 0.30 в момент peak_sec предсказания (рамка эталона берётся из его траектории);
  * сопоставление один к одному (венгерский алгоритм по длине пересечения);
  * IGNORE: предсказание не считается FP, если >= 50% его длительности лежит на IGNORE и оно не пересекает позитив;
  * метрики: TP, FP, FN, precision, recall, F1, FP/час по отрицательному видео, медианная задержка max(0, start_pred − start_gt).

Допущения (уточнить у организаторов): расширенный эталон ±3 с применяется к POSITIVE; IGNORE-интервалы не расширяются;
FP из NEGATIVE-клипов и фона считаются как обычные FP; фоновое время передаётся параметром `negative_hours`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]


@dataclass
class GT:
    clip_id: str
    person_id: str
    start: float
    end: float
    label: str                                    # POSITIVE | NEGATIVE | IGNORE
    box: Box | None = None                        # статичная рамка человека (если траектории нет)
    box_at: Callable[[float], Box | None] | None = None  # рамка человека в момент t (траектория)


@dataclass
class Pred:
    clip_id: str
    start: float
    end: float
    confidence: float
    peak: float
    box: Box
    event_id: str = ""


@dataclass
class Result:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fp_per_hour: float | None = None
    median_latency: float | None = None
    ignored_preds: int = 0
    matches: list[tuple[int, int]] = field(default_factory=list)   # (индекс pred, индекс gt)
    fp_idx: list[int] = field(default_factory=list)
    fn_idx: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("tp", "fp", "fn", "precision", "recall", "f1", "fp_per_hour", "median_latency", "ignored_preds")}


def iou(a: Box, b: Box) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _gt_box(g: GT, t: float) -> Box | None:
    return g.box_at(t) if g.box_at else g.box


def evaluate(preds: Sequence[Pred], gts: Sequence[GT], expand: float = 3.0, min_overlap: float = 1.0, iou_th: float = 0.30,
             ignore_frac: float = 0.5, negative_hours: float | None = None) -> Result:
    pos = [i for i, g in enumerate(gts) if g.label == "POSITIVE"]
    ign = [g for g in gts if g.label == "IGNORE"]

    # 1) предсказания, подавляемые IGNORE (не FP, если не пересекают позитив)
    ign_by_clip: dict[str, list[tuple[float, float]]] = {}
    for g in sorted(ign, key=lambda g: g.start):   # объединяем перекрывающиеся IGNORE: иначе общий участок считался бы дважды и доля превышала бы 1
        iv = ign_by_clip.setdefault(g.clip_id, [])
        if iv and g.start <= iv[-1][1]:
            iv[-1] = (iv[-1][0], max(iv[-1][1], g.end))
        else:
            iv.append((g.start, g.end))
    suppressed = set()
    for pi, p in enumerate(preds):
        dur = max(p.end - p.start, 1e-9)
        on_ign = sum(overlap(p.start, p.end, a, b) for a, b in ign_by_clip.get(p.clip_id, ()))
        touches_pos = any(gts[gi].clip_id == p.clip_id and overlap(p.start, p.end, gts[gi].start - expand, gts[gi].end + expand) > 0
                          for gi in pos)
        if on_ign / dur >= ignore_frac and not touches_pos:
            suppressed.add(pi)

    # 2) матрица допустимых совпадений (вес = длина пересечения)
    cand = np.zeros((len(preds), len(pos)))
    for pi, p in enumerate(preds):
        if pi in suppressed:
            continue
        for k, gi in enumerate(pos):
            g = gts[gi]
            if g.clip_id != p.clip_id:
                continue
            ov = overlap(p.start, p.end, g.start - expand, g.end + expand)
            if ov < min_overlap:
                continue
            gb = _gt_box(g, p.peak)
            if gb is None or iou(p.box, gb) < iou_th:
                continue
            cand[pi, k] = ov
    matches: list[tuple[int, int]] = []
    if cand.size and cand.max() > 0:
        r, c = linear_sum_assignment(-cand)
        matches = [(int(i), int(pos[j])) for i, j in zip(r, c) if cand[i, j] > 0]

    matched_p = {m[0] for m in matches}
    matched_g = {m[1] for m in matches}
    tp = len(matches)
    fp_idx = [i for i in range(len(preds)) if i not in matched_p and i not in suppressed]
    fn_idx = [gi for gi in pos if gi not in matched_g]
    fp, fn = len(fp_idx), len(fn_idx)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lat = [max(0.0, preds[pi].start - gts[gi].start) for pi, gi in matches]
    return Result(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                  fp_per_hour=(fp / negative_hours) if negative_hours else None,
                  median_latency=float(np.median(lat)) if lat else None, ignored_preds=len(suppressed),
                  matches=matches, fp_idx=fp_idx, fn_idx=fn_idx)


def threshold_curve(preds: Sequence[Pred], gts: Sequence[GT], thresholds: Sequence[float], **kw) -> list[dict]:
    """Кривая качества по порогу confidence (для выбора порога в центре плато, §9.4)."""
    out = []
    for th in thresholds:
        r = evaluate([p for p in preds if p.confidence >= th], gts, **kw)
        out.append(dict(threshold=float(th), **r.as_dict()))
    return out
