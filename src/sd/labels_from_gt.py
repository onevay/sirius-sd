"""Метки циклов из эталона событий: один труд разметки кормит и оценку, и обучение классификатора цикла.

Для каждого цикла (из таблицы `analysis/dataset_cycles.parquet` прогона) берётся рамка человека в момент пика и сопоставляется с интервалами эталона того же клипа:
  * внутри POSITIVE (±expand) и IoU рамок ≥ iou_th        → `smoke`
  * внутри POSITIVE по времени, но другой человек           → `other_neg`   (в тот же момент курит не он)
  * внутри NEGATIVE                                         → `other_neg`
  * внутри IGNORE                                           → `ignore`      (в обучение не берётся)
  * вне любых интервалов клипа, который просмотрен           → `other_neg` (если `unlabeled_as_negative`), иначе без метки
Метки пишутся с `source=gt`, ручные метки циклов не затрагиваются. Если в просмотренном клипе размечены не ВСЕ курения, «вне интервалов» даст ложные негативы — отключите опцию.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .evaluate import iou


def cycle_labels_from_gt(cycles: pd.DataFrame, gt: pd.DataFrame, box_at: Callable[[str, int, float], tuple | None], *, iou_th: float = 0.30, expand: float = 3.0,
                         unlabeled_as_negative: bool = True) -> pd.DataFrame:
    """`cycles`: video, tid, peak_t, cx, cy. `gt`: строки эталона. `box_at(video, tid, t)` — рамка человека (из треков прогона)."""
    rows = []
    for c in cycles.itertuples():
        g = gt[gt["clip_id"] == c.video]
        if g.empty:
            continue
        p = float(c.peak_t)
        label, note = None, ""
        pos = g[(g.label == "POSITIVE") & (g.start_sec - expand <= p) & (p <= g.end_sec + expand)]
        b = box_at(c.video, int(c.tid), p)
        if len(pos):
            hit = [r for r in pos.itertuples() if b is not None and np.isfinite([r.x1, r.y1, r.x2, r.y2]).all() and iou(tuple(b), (r.x1, r.y1, r.x2, r.y2)) >= iou_th]
            label, note = ("smoke", "в POSITIVE-интервале эталона") if hit else ("other_neg", "в интервале курения, но другой человек")
        elif len(g[(g.label == "IGNORE") & (g.start_sec <= p) & (p <= g.end_sec)]):
            label, note = "ignore", "IGNORE в эталоне"
        elif len(g[(g.label == "NEGATIVE") & (g.start_sec <= p) & (p <= g.end_sec)]):
            label, note = "other_neg", "NEGATIVE в эталоне"
        elif unlabeled_as_negative:
            label, note = "other_neg", "вне размеченных интервалов просмотренного клипа"
        if label:
            rows.append(dict(video=c.video, tid=int(c.tid), peak_t=p, cx=float(c.cx), cy=float(c.cy), label=label, note=note))
    return pd.DataFrame(rows, columns=["video", "tid", "peak_t", "cx", "cy", "label", "note"])


def summarize(labels: pd.DataFrame) -> dict:
    return {k: int(v) for k, v in labels["label"].value_counts().items()} if len(labels) else {}
