"""Сквозная проверка «циклы → оценка цикла → события по регламенту → протокол оценки» на ТЕКУЩИХ данных.

Это проверка механики и порядка величин, а не оценка качества: «эталон» строится из визуальных меток циклов (`docs/review/assistant_visual_labels.csv`),
то есть из тех же циклов, что находит автомат, и из суждений ассистента. Что она показывает: (1) вся цепочка работает от начала до конца и совпадает с протоколом
(`evaluate.py`: ±3 с, пересечение ≥ 1 с, IoU ≥ 0.30 в peak_sec, один к одному, IGNORE); (2) как меняются ложные события на негативных клипах, если циклы
оценивает классификатор, а не одно правило длительности паузы; (3) как порог на оценку цикла двигает precision/recall. Чего она НЕ показывает: пропуски самого
автомата (puff, которого автомат не нашёл, в «эталон» не попадёт) и качество на скрытом наборе.

Эталон: циклы с меткой `smoke` получают оценку 1, остальные 0 → те же правила сборки (`stages.assemble_events`) дают эталонные события; циклы с меткой `unsure`
становятся интервалами IGNORE. Предсказания: оценки цикла от модели, обученной БЕЗ видео этого цикла (leave-one-video-out), чтобы не оценивать на обучающих клипах.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from . import feature_auc as FA
from . import stages
from .calibrate import cfg_current_for_run, list_runs
from .evaluate import GT, Pred, evaluate
from .tracks import Tracks

KEY = ["video", "tid", "start"]


def lovo_scores(tab: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Оценка цикла моделью (логистическая регрессия на `cols`), обученной на размеченных циклах ДРУГИХ видео; для каждого видео — своя модель.

    Предсказываются все циклы видео (и размеченные, и нет). Если в обучающей части один класс — NaN.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = pd.Series(np.nan, index=tab.index, dtype=float)
    X = tab[cols].astype(float)
    for v in tab.video.unique():
        te = tab.video == v
        tr = (~te) & tab.y.notna()
        if tab.loc[tr, "y"].nunique() < 2:
            continue
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
        m.fit(X[tr], tab.loc[tr, "y"].astype(int))
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def _runs_by_video() -> dict[str, Path]:
    return {v: rd for v, rd in list_runs(pose_sig="rtmpose")}


def run_check(labels: Path = FA.LABELS, vlm: Path | None = None, cols: list[str] | None = None, thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
              an: Path = FA.AN) -> tuple[pd.DataFrame, dict]:
    """Таблица «метод × порог → TP/FP/FN/P/R/F1/ложные события в негативных клипах» и словарь с описанием выборки."""
    allc = FA.all_cycles_features(an, vlm)
    labelled = FA.feature_table(labels, an=an, vlm=None)[KEY + ["label", "y"]]
    allc = allc.merge(labelled, on=KEY, how="left")
    # «unsure» в таблицу признаков меток не попал (load_labelled их отбрасывает) — достанем из исходного CSV для IGNORE
    raw = pd.read_csv(labels)
    unsure = raw[raw.label == "unsure"][["video", "tid", "start"]].copy() if "label" in raw else pd.DataFrame(columns=["video", "tid", "start"])
    pick = FA.pick_columns(allc)
    use = cols or [c for g in pick.values() for c in g]
    allc = allc.assign(p_model=lovo_scores(allc, use))

    runs = _runs_by_video()
    gts: list[GT] = []
    scores_by_method: dict[str, dict[str, dict]] = {"rules": {}, "classifier": {}}
    seen: dict[str, dict] = {}
    neg_hours = 0.0
    for v, rd in runs.items():
        tr = Tracks.load(rd / "pose")
        cfg = cfg_current_for_run(rd)
        ser = stages.stage_features(tr, cfg, rd)
        cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
        sub = allc[allc.video == v]
        key = lambda r: (int(r.tid), round(float(r.start), 3))   # noqa: E731
        gt_scores = {key(r): (1.0 if r.label == "smoke" else 0.0) for r in sub.itertuples()}
        cls_scores = {key(r): float(r.p_model) for r in sub.itertuples() if np.isfinite(r.p_model)}
        _, gt_events = stages.assemble_events(tr, ser, cyc, cfg, "cam", v, gt_scores)
        for e in gt_events:
            gts.append(GT(v, str(e.tid), e.start, e.end, "POSITIVE", box_at=(lambda t, _tr=tr, _tid=e.tid: _tr.box_at(_tid, t))))
        for r in unsure[unsure.video == v].itertuples():
            m = cyc[(cyc.tid == r.tid) & ((cyc.start - r.start).abs() <= 0.06)]
            for c in m.itertuples():
                gts.append(GT(v, str(int(c.tid)), float(c.start), float(c.end), "IGNORE", box_at=(lambda t, _tr=tr, _tid=int(c.tid): _tr.box_at(_tid, t))))
        seen[v] = dict(tr=tr, ser=ser, cyc=cyc, cfg=cfg, cls_scores=cls_scores, fake=FA.folder_of(pd.Series([v])).iloc[0] == "fake")
        if seen[v]["fake"]:
            t = tr.frame_t.t
            neg_hours += float(t.max() - t.min()) / 3600.0

    rows = []
    for method in ("rules", "classifier"):
        for th in thresholds:
            preds: list[Pred] = []
            fake_clip: set[str] = set()
            for v, d in seen.items():
                cfg2 = copy.deepcopy(d["cfg"])
                cfg2["events"]["cycle_th"] = th
                scores = d["cls_scores"] if method == "classifier" else None
                df, _ = stages.assemble_events(d["tr"], d["ser"], d["cyc"], cfg2, "cam", v, scores)
                for r in df.itertuples():
                    preds.append(Pred(v, r.start_sec, r.end_sec, float(r.confidence), r.peak_sec, (r.x1, r.y1, r.x2, r.y2)))
                if d["fake"]:
                    fake_clip.add(v)
            res = evaluate(preds, gts, negative_hours=neg_hours or None)
            fp_neg = sum(1 for i in res.fp_idx if preds[i].clip_id in fake_clip)
            rows.append(dict(method=method, cycle_th=th, events=len(preds), tp=res.tp, fp=res.fp, fn=res.fn, precision=res.precision, recall=res.recall, f1=res.f1,
                             fp_in_negative_clips=fp_neg, fp_per_negative_hour=(fp_neg / neg_hours) if neg_hours else None,
                             median_latency=res.median_latency, ignored=res.ignored_preds))
    info = dict(clips=len(seen), negative_clips=sum(d["fake"] for d in seen.values()), negative_hours=neg_hours, gt_events=sum(g.label == "POSITIVE" for g in gts),
                ignore_intervals=sum(g.label == "IGNORE" for g in gts), cycles=len(allc), labelled=int(allc.y.notna().sum()), features=use)
    return pd.DataFrame(rows), info
