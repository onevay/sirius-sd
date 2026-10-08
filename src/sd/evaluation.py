"""Оценка набора клипов с опорой на целевую Event F1: итоговые метрики, кривая порога, интервал неопределённости, бюджет ошибок, разбор ошибок.

Протокол сопоставления — `evaluate.evaluate` (копия протокола организаторов: ±3 с, пересечение ≥ 1 с, IoU ≥ 0.30 в peak_sec, один к одному, IGNORE).
Здесь — то, что нужно, чтобы по результату принимать решения, а не только читать число:

* порог confidence выбирается по кривой на ВАЛИДАЦИИ (центр плато F1, а не острая вершина — на маленьком скрытом наборе это надёжнее, руководство §9.4);
  на скрытом наборе порог фиксируется заранее, кривая только показывается (подгонка по скрытым меткам запрещена);
* интервал F1 — бутстрэп по клипам: при 12 позитивах одна ошибка стоит 4–8 п.п., и «0.83 против 0.80» без интервала ничего не значит;
* бюджет ошибок: сколько FP+FN допустимо при целевой F1 и сколько стоит исправление одной ошибки;
* причины ошибок (дробление, склейка двух людей, рамка не та, ниже порога, пропуск, сложный негатив, фон) — основа таблицы «ошибки и ограничения» (§9.5).

Модуль без тяжёлых зависимостей: pandas, numpy, scipy (через `evaluate`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .evaluate import GT, Pred, Result, evaluate, iou, overlap

THRESHOLDS = tuple(np.round(np.arange(0.05, 0.9501, 0.025), 3))

CAUSE_RU = {
    "duplicate": "дробление: событие разбито на несколько",
    "wrong_person": "время совпало, рамка нет (другой человек или рамка не всего тела)",
    "hard_negative": "ложная тревога на размеченном негативе",
    "background": "ложная тревога вне эталона",
    "below_threshold": "событие найдено, но confidence ниже порога",
    "merged": "склейка: одно предсказание покрыло два эталона",
    "wrong_box": "время совпало, рамка не прошла по IoU",
    "missed": "события нет: цикл не найден, трек потерян или оценка цикла ниже порога",
    "clip_missed": "в клипе «курение» не найдено ни одного события",
    "clip_false": "в клипе без курения есть событие",
}


@dataclass
class Settings:
    threshold: float = 0.55
    target_f1: float = 0.80
    expand: float = 3.0
    min_overlap: float = 1.0
    iou_th: float = 0.30
    ignore_frac: float = 0.5

    def kw(self) -> dict:
        return dict(expand=self.expand, min_overlap=self.min_overlap, iou_th=self.iou_th, ignore_frac=self.ignore_frac)


@dataclass
class Report:
    settings: dict
    metrics: dict
    ci: dict
    budget: dict
    plateau: dict
    curve: pd.DataFrame
    per_clip: pd.DataFrame
    errors: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    mode: str = "events"                       # events — по ручному эталону событий | clips — слабая клип-уровня (папка = класс)

    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps(dict(mode=self.mode, settings=self.settings, metrics=self.metrics, ci=self.ci, budget=self.budget, plateau=self.plateau,
                                                       notes=self.notes), ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
        self.curve.to_csv(d / "curve.csv", index=False)
        self.per_clip.to_csv(d / "per_clip.csv", index=False, encoding="utf-8-sig")
        self.errors.to_csv(d / "errors.csv", index=False, encoding="utf-8-sig")

    @classmethod
    def load(cls, d: str | Path) -> "Report":
        d = Path(d)
        j = json.loads((d / "report.json").read_text(encoding="utf-8"))
        rd = lambda n: pd.read_csv(d / n, encoding="utf-8-sig") if (d / n).exists() and (d / n).stat().st_size > 2 else pd.DataFrame()   # noqa: E731
        return cls(settings=j["settings"], metrics=j["metrics"], ci=j["ci"], budget=j["budget"], plateau=j["plateau"], curve=rd("curve.csv"), per_clip=rd("per_clip.csv"),
                   errors=rd("errors.csv"), notes=j.get("notes", []), mode=j.get("mode", "events"))


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------------------------------------ предсказания
def to_preds(events: pd.DataFrame, clip_id: str) -> list[Pred]:
    """Таблица событий (формат организаторов: start_sec, end_sec, confidence, peak_sec, x1..y2) → `Pred`; `clip_id` перезаписывается уникальным id клипа."""
    out = []
    for r in events.itertuples():
        out.append(Pred(clip_id=clip_id, start=float(r.start_sec), end=float(r.end_sec), confidence=float(r.confidence), peak=float(r.peak_sec),
                        box=(float(r.x1), float(r.y1), float(r.x2), float(r.y2)), event_id=str(getattr(r, "event_id", ""))))
    return out


def negative_clips(gts: Sequence[GT], clips: Iterable[str]) -> set[str]:
    """Клипы без POSITIVE в эталоне: по ним считаются ложные тревоги в час."""
    pos = {g.clip_id for g in gts if g.label == "POSITIVE"}
    return {c for c in clips if c not in pos}


# ------------------------------------------------------------------------------------------ метрики на одном пороге
def metrics_at(preds: Sequence[Pred], gts: Sequence[GT], st: Settings, durations: dict[str, float], th: float | None = None) -> tuple[dict, Result, list[Pred]]:
    """(метрики, результат протокола, предсказания выше порога). Порог — по confidence события."""
    th = st.threshold if th is None else th
    kept = [p for p in preds if p.confidence >= th]
    res = evaluate(kept, gts, **st.kw())
    neg = negative_clips(gts, durations)
    neg_h = sum(durations[c] for c in neg) / 3600.0
    all_h = sum(durations.values()) / 3600.0
    fp_neg = sum(1 for i in res.fp_idx if kept[i].clip_id in neg)
    m = dict(threshold=float(th), tp=res.tp, fp=res.fp, fn=res.fn, precision=res.precision, recall=res.recall, f1=res.f1, n_pred=len(kept), n_gt=res.tp + res.fn,
             ignored=res.ignored_preds, median_latency=res.median_latency, fp_neg=fp_neg, neg_hours=neg_h, hours=all_h,
             fp_per_hour=(fp_neg / neg_h) if neg_h > 0 else None, fp_per_hour_all=(res.fp / all_h) if all_h > 0 else None, clips=len(durations), neg_clips=len(neg))
    return m, res, kept


def sweep(preds: Sequence[Pred], gts: Sequence[GT], st: Settings, durations: dict[str, float], thresholds: Sequence[float] = THRESHOLDS) -> pd.DataFrame:
    rows = [metrics_at(preds, gts, st, durations, th)[0] for th in thresholds]
    return pd.DataFrame(rows)[["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "n_pred", "fp_per_hour", "median_latency"]]


def plateau(curve: pd.DataFrame, tol: float = 0.02) -> dict:
    """Лучший F1 и «плато»: непрерывный участок порогов, где F1 ≥ лучший − tol; порог в центре плато устойчивее острой вершины."""
    if curve.empty or not np.isfinite(curve.f1).any():
        return dict(best_threshold=None, best_f1=None, lo=None, hi=None, center=None, tol=tol)
    f = curve.f1.to_numpy(float)
    th = curve.threshold.to_numpy(float)
    ib = int(np.nanargmax(f))
    ok = f >= f[ib] - tol
    lo = hi = ib
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(f) - 1 and ok[hi + 1]:
        hi += 1
    center = float(th[(lo + hi) // 2]) if lo != hi else float(th[ib])
    return dict(best_threshold=float(th[ib]), best_f1=float(f[ib]), lo=float(th[lo]), hi=float(th[hi]), center=center, tol=tol)


def f1_of(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def error_budget(tp: int, fp: int, fn: int, target: float) -> dict:
    """Сколько ошибок (FP+FN) допустимо при данном числе TP, чтобы F1 ≥ target, и сколько стоит исправление одной ошибки.

    F1 = 2TP / (2TP + FP + FN)  ⇒  FP + FN ≤ 2TP·(1/target − 1).
    """
    allowed = 2 * tp * (1.0 / target - 1.0) if target > 0 else float("inf")
    errors = fp + fn
    cur = f1_of(tp, fp, fn)
    return dict(target=target, f1=cur, errors=errors, allowed=float(np.floor(allowed + 1e-9)), over=float(max(0.0, errors - np.floor(allowed + 1e-9))), reached=cur >= target,
                gain_fix_fp=(f1_of(tp, fp - 1, fn) - cur) if fp else 0.0, gain_fix_fn=(f1_of(tp + 1, fp, fn - 1) - cur) if fn else 0.0,
                cost_extra_fp=cur - f1_of(tp, fp + 1, fn), cost_extra_fn=cur - f1_of(tp - 1, fp, fn + 1) if tp else 0.0)


def bootstrap(preds: Sequence[Pred], gts: Sequence[GT], st: Settings, durations: dict[str, float], n: int = 300, seed: int = 0, th: float | None = None) -> dict:
    """95% интервал F1/precision/recall бутстрэпом по КЛИПАМ (события одного клипа зависимы). Меньше 3 клипов — интервала нет."""
    clips = sorted(durations)
    out = {k: (None, None) for k in ("f1", "precision", "recall")}
    if len(clips) < 3:
        return dict(out, n=0)
    th = st.threshold if th is None else th
    kept = [p for p in preds if p.confidence >= th]
    pc: dict[str, list[Pred]] = {}
    gc: dict[str, list[GT]] = {}
    for p in kept:
        pc.setdefault(p.clip_id, []).append(p)
    for g in gts:
        gc.setdefault(g.clip_id, []).append(g)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in out}
    for _ in range(n):
        ps, gs = [], []
        for j, c in enumerate(rng.choice(clips, len(clips))):
            cid = f"{c}#{j}"
            ps += [replace(p, clip_id=cid) for p in pc.get(c, ())]
            gs += [replace(g, clip_id=cid) for g in gc.get(c, ())]
        if not any(g.label == "POSITIVE" for g in gs):
            continue
        r = evaluate(ps, gs, **st.kw())
        for k in vals:
            vals[k].append(getattr(r, k))
    if len(vals["f1"]) < 20:
        return dict(out, n=len(vals["f1"]))
    return {**{k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}, "n": len(vals["f1"])}


# ------------------------------------------------------------------------------------------ разбор ошибок
def classify_errors(kept: Sequence[Pred], all_preds: Sequence[Pred], gts: Sequence[GT], res: Result, st: Settings) -> pd.DataFrame:
    """Причина каждой ошибки. Порядок проверок — от самой конкретной причины к самой общей."""
    matched_g = {gi for _, gi in res.matches}
    pos_idx = [i for i, g in enumerate(gts) if g.label == "POSITIVE"]

    def gbox(g: GT, t: float):
        return g.box_at(t) if g.box_at else g.box

    def time_ov(p: Pred, g: GT) -> float:
        return overlap(p.start, p.end, g.start - st.expand, g.end + st.expand)

    def box_ok(p: Pred, g: GT) -> bool:
        b = gbox(g, p.peak)
        return b is not None and iou(p.box, b) >= st.iou_th

    rows = []
    for pi in res.fp_idx:
        p = kept[pi]
        near = [gi for gi in pos_idx if gts[gi].clip_id == p.clip_id and time_ov(p, gts[gi]) >= st.min_overlap]
        if near:
            cause = "duplicate" if any(gi in matched_g and box_ok(p, gts[gi]) for gi in near) else "wrong_person"
            gi = near[0]
        elif any(g.label == "NEGATIVE" and g.clip_id == p.clip_id and overlap(p.start, p.end, g.start, g.end) > 0 for g in gts):
            cause, gi = "hard_negative", None
        else:
            cause, gi = "background", None
        rows.append(dict(kind="FP", clip_id=p.clip_id, start=p.start, end=p.end, confidence=p.confidence, peak=p.peak, person=gts[gi].person_id if gi is not None else "",
                         cause=cause, cause_ru=CAUSE_RU[cause]))
    for gi in res.fn_idx:
        g = gts[gi]
        same = [p for p in all_preds if p.clip_id == g.clip_id and time_ov(p, g) >= st.min_overlap]
        qualify = [p for p in same if box_ok(p, g)]
        kept_ids = {id(p) for p in kept}
        if qualify and all(id(p) not in kept_ids for p in qualify):
            cause, conf = "below_threshold", max(p.confidence for p in qualify)
        elif qualify:
            cause, conf = "merged", max(p.confidence for p in qualify)
        elif same:
            cause, conf = "wrong_box", max(p.confidence for p in same)
        else:
            cause, conf = "missed", float("nan")
        rows.append(dict(kind="FN", clip_id=g.clip_id, start=g.start, end=g.end, confidence=conf, peak=(g.start + g.end) / 2, person=g.person_id, cause=cause, cause_ru=CAUSE_RU[cause]))
    cols = ["kind", "clip_id", "start", "end", "confidence", "peak", "person", "cause", "cause_ru"]
    return pd.DataFrame(rows, columns=cols).sort_values(["clip_id", "start"]).reset_index(drop=True) if rows else pd.DataFrame(columns=cols)


def per_clip_table(kept: Sequence[Pred], gts: Sequence[GT], res: Result, durations: dict[str, float], weak: dict[str, str] | None = None) -> pd.DataFrame:
    rows = {c: dict(clip_id=c, duration=float(d), gt_positive=0, pred=0, tp=0, fp=0, fn=0) for c, d in durations.items()}
    for g in gts:
        if g.clip_id in rows and g.label == "POSITIVE":
            rows[g.clip_id]["gt_positive"] += 1
    for p in kept:
        if p.clip_id in rows:
            rows[p.clip_id]["pred"] += 1
    for pi, _ in res.matches:
        rows[kept[pi].clip_id]["tp"] += 1
    for pi in res.fp_idx:
        rows[kept[pi].clip_id]["fp"] += 1
    for gi in res.fn_idx:
        if gts[gi].clip_id in rows:
            rows[gts[gi].clip_id]["fn"] += 1
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return pd.DataFrame(columns=["clip_id", "duration", "gt_positive", "pred", "tp", "fp", "fn", "f1", "status"])
    df["f1"] = [f1_of(r.tp, r.fp, r.fn) if (r.tp + r.fp + r.fn) else np.nan for r in df.itertuples()]
    df["status"] = np.where((df.fp == 0) & (df.fn == 0), "ok", np.where((df.fp > 0) & (df.fn > 0), "FP+FN", np.where(df.fp > 0, "FP", "FN")))
    if weak:
        df["folder_label"] = df.clip_id.map(weak)
    return df.sort_values(["status", "clip_id"], key=lambda s: s.map({"ok": 9}).fillna(0) if s.name == "status" else s).reset_index(drop=True)


def _notes(m: dict, st: Settings, mode: str) -> list[str]:
    n = []
    if m["clips"] == 0:
        n.append("нет клипов для оценки")
    if mode == "events":
        if m["n_gt"] < 6:
            n.append(f"в эталоне {m['n_gt']} позитивных событий: одна ошибка двигает F1 на 5–15 п.п., интервал широкий — ориентируйтесь на него, а не на точку")
        if m["neg_clips"] == 0:
            n.append("нет клипов без позитивов — «ложных тревог в час» не посчитать; добавьте размеченные негативы или фон")
        elif m["neg_hours"] < 0.25:
            n.append(f"фона всего {m['neg_hours'] * 60:.0f} мин: оценка ложных тревог в час очень грубая")
    else:
        n.append("слабая оценка по папкам: клип «курение» считается верно найденным, если в нём есть событие выше порога; границы и рамки не проверяются")
    return n


# ------------------------------------------------------------------------------------------ сборка отчёта
def build_report(preds: Sequence[Pred], gts: Sequence[GT], durations: dict[str, float], st: Settings, *, tol: float = 0.02, n_boot: int = 300,
                 thresholds: Sequence[float] = THRESHOLDS, weak: dict[str, str] | None = None) -> Report:
    """Полный отчёт по эталону событий для набора клипов `durations` (clip_id → длительность, с)."""
    gts = [g for g in gts if g.clip_id in durations]
    preds = [p for p in preds if p.clip_id in durations]
    m, res, kept = metrics_at(preds, gts, st, durations)
    curve = sweep(preds, gts, st, durations, thresholds)
    pl = plateau(curve, tol)
    ci = bootstrap(preds, gts, st, durations, n=n_boot)
    return Report(settings=dict(threshold=st.threshold, target_f1=st.target_f1, expand=st.expand, min_overlap=st.min_overlap, iou_th=st.iou_th, ignore_frac=st.ignore_frac),
                  metrics=m, ci=ci, budget=error_budget(m["tp"], m["fp"], m["fn"], st.target_f1), plateau=pl, curve=curve,
                  per_clip=per_clip_table(kept, gts, res, durations, weak), errors=classify_errors(kept, preds, gts, res, st), notes=_notes(m, st, "events"))


# ------------------------------------------------------------------------------------------ слабая оценка «папка = класс»
def build_weak_report(clips: pd.DataFrame, st: Settings, *, thresholds: Sequence[float] = THRESHOLDS, tol: float = 0.02, n_boot: int = 300) -> Report:
    """Оценка уровня КЛИПА без ручной разметки. `clips`: clip_id, y (1 — клип «курение»), max_conf (максимум confidence событий клипа или NaN), n_by_th не нужен,
    `events` — список confidence событий клипа (list[float]), duration — длительность, с.

    TP — позитивный клип с событием выше порога; FP — негативный клип с событием; FN — позитивный клип без события. Это грубая проверка «система вообще различает клипы»,
    а не Event F1: времена и рамки здесь не проверяются."""
    d = clips.copy()
    d["conf"] = d["events"].map(lambda e: list(e) if isinstance(e, (list, tuple, np.ndarray)) else [])

    def at(th: float, sub: pd.DataFrame) -> dict:
        hit = sub["conf"].map(lambda e: any(c >= th for c in e))
        y = sub["y"].astype(int)
        tp, fp, fn = int(((y == 1) & hit).sum()), int(((y == 0) & hit).sum()), int(((y == 1) & ~hit).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        neg = sub[y == 0]
        neg_h = float(neg["duration"].sum()) / 3600.0
        ev_neg = int(sum(sum(1 for c in e if c >= th) for e in neg["conf"]))
        return dict(threshold=float(th), tp=tp, fp=fp, fn=fn, precision=pr, recall=rc, f1=f1_of(tp, fp, fn), n_pred=int(sum(sum(1 for c in e if c >= th) for e in sub["conf"])),
                    fp_per_hour=(ev_neg / neg_h) if neg_h > 0 else None, median_latency=None, fp_neg=ev_neg, neg_hours=neg_h, hours=float(sub["duration"].sum()) / 3600.0,
                    n_gt=tp + fn, clips=int(len(sub)), neg_clips=int((y == 0).sum()), ignored=0)

    m = at(st.threshold, d)
    curve = pd.DataFrame([at(t, d) for t in thresholds])[["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "n_pred", "fp_per_hour", "median_latency"]]
    ci = dict(f1=(None, None), precision=(None, None), recall=(None, None), n=0)
    if len(d) >= 3:
        rng = np.random.default_rng(0)
        vs = {"f1": [], "precision": [], "recall": []}
        for _ in range(n_boot):
            b = d.iloc[rng.integers(0, len(d), len(d))]
            if (b.y == 1).any():
                r = at(st.threshold, b)
                for k in vs:
                    vs[k].append(r[k])
        if len(vs["f1"]) >= 20:
            ci = {**{k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vs.items()}, "n": len(vs["f1"])}
    hit = d["conf"].map(lambda e: any(c >= st.threshold for c in e))
    d["tp"] = ((d.y == 1) & hit).astype(int)
    d["fp"] = ((d.y == 0) & hit).astype(int)
    d["fn"] = ((d.y == 1) & ~hit).astype(int)
    d["pred"] = d["conf"].map(lambda e: sum(1 for c in e if c >= st.threshold))
    d["gt_positive"] = d.y.astype(int)
    d["f1"] = np.nan
    d["status"] = np.where(d.fp > 0, "FP", np.where(d.fn > 0, "FN", "ok"))
    per = d[["clip_id", "duration", "gt_positive", "pred", "tp", "fp", "fn", "f1", "status"]].sort_values(["status", "clip_id"], key=lambda s: s.map({"ok": 9}).fillna(0) if s.name == "status" else s)
    err = pd.DataFrame([dict(kind=("FN" if r.fn else "FP"), clip_id=r.clip_id, start=np.nan, end=np.nan, confidence=(max(r.conf) if r.conf else np.nan), peak=np.nan, person="",
                             cause=("clip_missed" if r.fn else "clip_false"), cause_ru=CAUSE_RU["clip_missed" if r.fn else "clip_false"]) for r in d.itertuples() if r.fn or r.fp],
                       columns=["kind", "clip_id", "start", "end", "confidence", "peak", "person", "cause", "cause_ru"])
    return Report(settings=dict(threshold=st.threshold, target_f1=st.target_f1), metrics=m, ci=ci, budget=error_budget(m["tp"], m["fp"], m["fn"], st.target_f1),
                  plateau=plateau(curve, tol), curve=curve, per_clip=per.reset_index(drop=True), errors=err, notes=_notes(m, st, "clips"), mode="clips")


def choose_threshold(rep: Report, policy: str, fixed: float) -> float:
    """Порог для итоговой цифры. `fixed` — заранее зафиксированный (скрытый набор), `plateau` — центр плато F1, `best` — вершина кривой. Только для валидации."""
    p = rep.plateau
    if policy == "plateau" and p.get("center") is not None:
        return float(p["center"])
    if policy == "best" and p.get("best_threshold") is not None:
        return float(p["best_threshold"])
    return float(fixed)


def finalize_report(preds: Sequence[Pred], gts: Sequence[GT], durations: dict[str, float], st: Settings, policy: str = "fixed", **kw) -> Report:
    """Отчёт с учётом политики порога: `fixed` — порог из `st`; `plateau`/`best` — порог берётся с кривой ЭТОГО ЖЕ набора (только для валидации!) и отчёт пересобирается."""
    rep = build_report(preds, gts, durations, st, **kw)
    if policy == "fixed":
        return rep
    th = choose_threshold(rep, policy, st.threshold)
    if abs(th - st.threshold) < 1e-9:
        return rep
    rep2 = build_report(preds, gts, durations, replace(st, threshold=th), **kw)
    rep2.notes.append(f"порог {th:.3f} выбран по кривой этого же набора ({'центр плато' if policy == 'plateau' else 'вершина кривой'}): цифра оптимистична, честная — на другом наборе с этим порогом")
    return rep2


def paired_bootstrap(preds_a: Sequence[Pred], preds_b: Sequence[Pred], gts: Sequence[GT], durations: dict[str, float], st_a: Settings, st_b: Settings | None = None,
                     n: int = 500, seed: int = 0) -> dict:
    """Парный бутстрэп по клипам: насколько надёжно профиль B лучше профиля A на ОДНИХ и тех же клипах. Возвращает среднюю разность F1 (B − A), 95% интервал и долю
    ресэмплов, где B лучше. Интервалы F1 каждого профиля по отдельности перекрываются почти всегда; парная разность чувствительнее, потому что трудные клипы общие."""
    st_b = st_b or st_a
    clips = sorted(durations)
    if len(clips) < 3:
        return dict(n=0, diff=None, lo=None, hi=None, p_better=None, clips=len(clips))
    ka = [p for p in preds_a if p.confidence >= st_a.threshold]
    kb = [p for p in preds_b if p.confidence >= st_b.threshold]
    pa: dict[str, list] = {}
    pb: dict[str, list] = {}
    gg: dict[str, list] = {}
    for src, dst in ((ka, pa), (kb, pb)):
        for p in src:
            dst.setdefault(p.clip_id, []).append(p)
    for g in gts:
        gg.setdefault(g.clip_id, []).append(g)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sa, sb, sg = [], [], []
        for j, c in enumerate(rng.choice(clips, len(clips))):
            cid = f"{c}#{j}"
            sa += [replace(p, clip_id=cid) for p in pa.get(c, ())]
            sb += [replace(p, clip_id=cid) for p in pb.get(c, ())]
            sg += [replace(g, clip_id=cid) for g in gg.get(c, ())]
        if not any(g.label == "POSITIVE" for g in sg):
            continue
        diffs.append(evaluate(sb, sg, **st_b.kw()).f1 - evaluate(sa, sg, **st_a.kw()).f1)
    if len(diffs) < 20:
        return dict(n=len(diffs), diff=None, lo=None, hi=None, p_better=None, clips=len(clips))
    d = np.array(diffs)
    return dict(n=len(d), diff=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p_better=float((d > 0).mean()), clips=len(clips))
