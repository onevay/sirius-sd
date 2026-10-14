"""Активное обучение и дообучение классификатора цикла.

Разметка жестов дорога, а циклов-кандидатов много. Активное обучение отвечает на вопрос «какой цикл разметить следующим, чтобы модель улучшилась быстрее всего»:
  random         случайный цикл — базовая линия (с ней сравниваются остальные стратегии);
  uncertainty    оценка ближе всего к 0.5 (модель сомневается);
  threshold      оценка ближе всего к порогу цикла `events.cycle_th` — именно эти циклы решают, будет ли событие;
  committee      члены ансамбля расходятся (разброс оценок): сомневается не одна модель, а весь ансамбль;
  diverse        из самых неопределённых — набор, далёкий друг от друга в пространстве признаков (чтобы не размечать десять одинаковых циклов);
  suspect        (только размеченные) метка не согласуется с out-of-fold оценкой — кандидат на перепроверку, а не новый цикл.
`simulate` проверяет стратегии на УЖЕ размеченных данных: метки скрываются и «покупаются» по очереди, качество измеряется на отложенных видео — так видно, какая стратегия и какого размера пакет
дают лучшую кривую обучения именно на вашем датасете.

Дообучение (`finetune_cv`, `finetune_save`): пакет переобучается на объединённых данных, но новым и «трудным» циклам даётся больший вес (колонка `_w` — её используют `bundle` и `classifier`
при обучении, оценка идёт без весов). Сравнение — по фолдам, где отложены целые видео/сцены; результат сохраняется новым пакетом с ссылкой на родителя в манифесте.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import bundle as B
from . import classifier as CL
from .feature_auc import _auc

STRATEGIES = {
    "random": "случайный цикл (базовая линия)",
    "uncertainty": "неопределённость: оценка около 0.5",
    "threshold": "около порога цикла",
    "committee": "расхождение членов ансамбля",
    "diverse": "неопределённые и непохожие друг на друга",
}
KEY = ["video", "tid", "start"]


# ---------------------------------------------------------------------------------------------- оценки пула
def spec_from_bundle(path: str | Path) -> dict:
    """Спецификация классификатора из манифеста пакета: полная (`spec_full`, если пакет обучен по спецификации) или собранная из членов (гиперпараметры — по умолчанию)."""
    man = B.Bundle(Path(path)).manifest
    if man.get("spec_full"):
        return man["spec_full"]
    members = {n: dict(kind=m["kind"], features=list(m["features"]), params=dict(CL.PARAMS.get(m["kind"], {}))) for n, m in man["members"].items()}
    return dict(name=Path(path).name, description=f"из пакета {Path(path).name}", set=man.get("feature_set"), features=man["features"], members=members, cv=dict(CL.DEFAULT_CV), calibrate=bool(man.get("calibrated")))


def committee(train: pd.DataFrame, pool: pd.DataFrame, spec: dict, seed: int = 0) -> pd.DataFrame:
    """Обучает членов ансамбля на размеченных циклах и оценивает пул: колонки `p` (среднее), `std` (разброс членов) и оценка каждого члена."""
    t = train[train.y.notna()]
    y = t.y.to_numpy(float).astype(int)
    out = {}
    for n, m in spec["members"].items():
        st = B.fit_member(m["kind"], t, y, m["features"], w=t["_w"].to_numpy(float) if "_w" in t.columns else None, seed=seed, **(m.get("params") or {}))
        out[n] = B.predict_member(st, pool)
    df = pd.DataFrame(out, index=pool.index)
    df.insert(0, "std", df.std(axis=1, ddof=0))
    df.insert(0, "p", df[list(spec["members"])].mean(axis=1))
    return df


def _feature_space(tab: pd.DataFrame, feats: Sequence[str]) -> np.ndarray:
    a = tab[list(feats)].to_numpy(float)
    med = np.nanmedian(a, axis=0)
    a = np.where(np.isnan(a), med[None, :], a)
    sd = np.where(a.std(0) > 1e-9, a.std(0), 1.0)
    return (a - a.mean(0)) / sd


def priority(pool: pd.DataFrame, strategy: str, *, threshold: float = 0.5, space: np.ndarray | None = None, k: int = 10, rng: np.random.Generator | None = None) -> np.ndarray:
    """Порядок разметки: позиции строк `pool` (с колонками `p`, `std`), самые нужные первыми (не более k)."""
    n = len(pool)
    rng = rng or np.random.default_rng(0)
    if strategy == "random":
        return rng.permutation(n)[:k]
    p, sd = pool["p"].to_numpy(float), pool["std"].to_numpy(float)
    if strategy == "uncertainty":
        s = -np.abs(p - 0.5)
    elif strategy == "threshold":
        s = -np.abs(p - threshold)
    elif strategy == "committee":
        s = sd + 1e-3 * (-np.abs(p - 0.5))
    elif strategy == "diverse":
        cand = np.argsort(np.abs(p - 0.5))[: max(k * 4, k)]       # 4k самых неопределённых
        if space is None or len(cand) <= k:
            return cand[:k]
        chosen = [cand[0]]
        d = np.linalg.norm(space[cand] - space[cand[0]], axis=1)
        while len(chosen) < k:
            j = int(np.argmax(d))
            chosen.append(cand[j])
            d = np.minimum(d, np.linalg.norm(space[cand] - space[cand[j]], axis=1))
        return np.array(chosen)
    else:
        raise KeyError(strategy)
    return np.argsort(-s)[:k]


def suspects(tab: pd.DataFrame, spec: dict, top: int = 20) -> pd.DataFrame:
    """Размеченные циклы, у которых out-of-fold оценка сильнее всего расходится с меткой (кандидаты на перепроверку метки)."""
    t = tab[tab.y.notna()].reset_index(drop=True)
    oof = CL.cv_predict(t, spec)["__ensemble__"]
    out = t.assign(oof=oof, расхождение=np.abs(t.y.to_numpy(float) - oof)).sort_values("расхождение", ascending=False)
    return out.head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- симуляция
def _start_set(y: np.ndarray, init: int, rng: np.random.Generator) -> list[int]:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = max(init, 4)
    take = list(rng.choice(pos, size=min(len(pos), max(2, k // 3)), replace=False)) + list(rng.choice(neg, size=min(len(neg), max(2, k - k // 3)), replace=False))
    return [int(i) for i in take]


def simulate(tab: pd.DataFrame, spec: dict, strategies: Sequence[str] = ("random", "uncertainty", "committee", "diverse"), *, init: int = 16, batch: int = 8, rounds: int = 6,
             folds: int = 4, threshold: float = 0.4, by_scene: bool = True, seed: int = 0, progress: Callable[[int, int, str], None] | None = None) -> pd.DataFrame:
    """Кривые обучения стратегий отбора. Для каждой отложенной части видео/сцен: метки пула скрыты, стартовый набор — `init` случайных циклов, затем `rounds` раз модель
    выбирает `batch` циклов, их метки «покупаются», качество (AUC ансамбля) измеряется на отложенных видео. Возвращает: strategy, labels, auc (среднее по частям), std, n."""
    t = tab[tab.y.notna()].reset_index(drop=True)
    y = t.y.to_numpy(float).astype(int)
    groups = CL._groups(t, by_scene)
    space = _feature_space(t, CL.spec_features(spec))
    cv = StratifiedGroupKFold(n_splits=min(folds, len(set(groups))), shuffle=True, random_state=seed)
    total = max(1, len(strategies) * folds * (rounds + 1))
    done = 0
    rows = []
    for f, (tr, te) in enumerate(cv.split(t, y, groups)):
        if len(np.unique(y[te])) < 2:
            continue
        for strat in strategies:
            rng = np.random.default_rng(seed * 1000 + f)
            lab = _start_set(y[tr], init, rng)                 # позиции внутри tr
            for r in range(rounds + 1):
                L = t.iloc[tr[lab]]
                if len(np.unique(L.y)) < 2:
                    break
                cm = committee(L, t.iloc[te], spec, seed=seed + r)
                rows.append(dict(strategy=strat, fold=f, round=r, labels=len(lab), auc=_auc(y[te], cm["p"].to_numpy())))
                done += 1
                if progress:
                    progress(done, total, f"{strat} · часть {f + 1} · раунд {r}")
                if r == rounds:
                    break
                rest = np.array([i for i in range(len(tr)) if i not in set(lab)])
                if not len(rest):
                    break
                pool = t.iloc[tr[rest]]
                cmp_ = committee(L, pool, spec, seed=seed + r)
                pick = priority(cmp_, strat, threshold=threshold, space=space[tr[rest]], k=min(batch, len(rest)), rng=rng)
                lab = lab + [int(rest[i]) for i in pick]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["strategy", "labels", "auc", "std", "n"])
    g = df.groupby(["strategy", "round"]).agg(labels=("labels", "mean"), auc=("auc", "mean"), std=("auc", "std"), n=("auc", "count")).reset_index()
    return g.sort_values(["strategy", "round"], ignore_index=True)


# ---------------------------------------------------------------------------------------------- дообучение
def weights_for(tab: pd.DataFrame, *, new_videos: Sequence[str] = (), w_new: float = 2.0, hard: pd.Series | None = None, w_hard: float = 2.0) -> pd.DataFrame:
    """Таблица с колонкой `_w`: новые видео (`new_videos`) умножают вес на `w_new`, «трудные» циклы (булева серия `hard` по индексу таблицы) — на `w_hard`. Остальные — 1."""
    w = np.ones(len(tab))
    if new_videos:
        w = np.where(tab.video.isin(set(new_videos)).to_numpy(), w * w_new, w)
    if hard is not None:
        w = np.where(hard.reindex(tab.index, fill_value=False).to_numpy(bool), w * w_hard, w)
    return tab.assign(_w=w)


def hard_cycles(tab: pd.DataFrame, spec: dict, margin: float = 0.5) -> pd.Series:
    """Трудные циклы: размеченные, у которых out-of-fold оценка ошибается на `margin` и больше (булева серия по индексу таблицы)."""
    t = tab[tab.y.notna()]
    oof = CL.cv_predict(t.reset_index(drop=True), spec)["__ensemble__"]
    bad = pd.Series(np.abs(t.y.to_numpy(float) - oof) >= margin, index=t.index)
    return bad.reindex(tab.index, fill_value=False)


def finetune_cv(tab: pd.DataFrame, spec: dict, *, new_videos: Sequence[str] = (), w_new: float = 2.0, w_hard: float = 2.0, use_hard: bool = True, by_scene: bool = True,
                repeats: int = 3, seed: int = 0) -> dict:
    """Сравнивает обучение с весами и без них по фолдам, где отложены целые видео/сцены. Метрики: AUC на всех циклах и на циклах `new_videos` (если они есть).
    Возвращает {'table': сводка, 'oof_plain', 'oof_weighted', 'y', 'rows': таблица циклов}."""
    t = tab[tab.y.notna()].reset_index(drop=True)
    sp = {**spec, "cv": {**CL.DEFAULT_CV, **(spec.get("cv") or {}), "by_scene": by_scene, "repeats": repeats, "seed": seed}}
    hard = hard_cycles(t, sp) if use_hard else None
    tw = weights_for(t, new_videos=new_videos, w_new=w_new, hard=hard, w_hard=w_hard)
    plain = CL.cv_predict(t.drop(columns=["_w"], errors="ignore"), sp)["__ensemble__"]
    weighted = CL.cv_predict(tw, sp)["__ensemble__"]
    y = t.y.to_numpy(float).astype(int)
    newm = t.video.isin(set(new_videos)).to_numpy()
    rows = []
    for name, p in (("без весов", plain), ("с весами", weighted)):
        rows.append(dict(вариант=name, AUC_все=round(_auc(y, p), 4), AUC_новые=round(_auc(y[newm], p[newm]), 4) if newm.sum() > 5 and len(np.unique(y[newm])) == 2 else None,
                         циклов=int(len(y)), новых=int(newm.sum()), трудных=int(hard.sum()) if hard is not None else 0))
    return dict(table=pd.DataFrame(rows), oof_plain=plain, oof_weighted=weighted, y=y, rows=tw)


def finetune_save(tab_weighted: pd.DataFrame, spec: dict, out_dir: Path, *, parent: str | None, params: dict) -> dict:
    """Обучает и сохраняет новый пакет на таблице с весами; в манифест — родитель и параметры дообучения (откуда, с какими весами)."""
    meta = dict(parent=parent, finetune=params)
    return CL.train(tab_weighted, spec, out_dir, meta=meta)
