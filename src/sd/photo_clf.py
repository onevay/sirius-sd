"""Фото-классификатор «курит / не курит» на замороженных эмбеддингах: набор моделей, ансамбль, оценка без утечки.

Протокол (почему так, см. docs/ANALYSIS.md §9):
  * CV по группам почти-дубликатов (StratifiedGroupKFold) — дубль в train и test не попадает;
  * межнаборный перенос (обучили на A, проверили на B) — единственная оценка, не зависящая от «почерка» набора;
  * разрезы по «трудным» негативам (drinking, phoning…) — там и проверяется, что модель смотрит на сигарету, а не на «человека у лица»;
  * AUC по страте яркости — проверка на шорткат из `photo_data.audit`.
Модели сохраняются БЕЗ pickle: логистические веса и калибровка — JSON, LightGBM — текстовый файл (см. `save_bundle`).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def auc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def zoo(seed: int = 0) -> dict:
    """Фабрики моделей: линейная, ядерная, дерево-ансамбль. Все умеют predict_proba; веса классов сбалансированы."""
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)),
        "svm": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "et": lambda: ExtraTreesClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", n_jobs=2, random_state=seed),
    }


def fit_predict(kind: str, Xtr, ytr, Xte, seed: int = 0) -> np.ndarray:
    m = zoo(seed)[kind]()
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        oof[te] = fit_predict(kind, X[tr], y[tr], X[te], seed)
    return oof


def rank01(s: np.ndarray) -> np.ndarray:
    """Ранг в [0,1]: усреднять оценки разных моделей по рангам надёжнее, чем по вероятностям с разной калибровкой."""
    return pd.Series(s).rank(pct=True).to_numpy()


def summarize(y, s, thr: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = np.asarray(s) >= thr
    return dict(auc=auc(y, s), ap=float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
                recall=float(pred[y == 1].mean()) if (y == 1).any() else float("nan"),
                fpr=float(pred[y == 0].mean()) if (y == 0).any() else float("nan"), n=int(len(y)), pos=int(y.sum()))


def per_negative_auc(df: pd.DataFrame, s: np.ndarray, min_n: int = 20) -> pd.DataFrame:
    """AUC «курит» против КАЖДОГО негативного класса (колонка cls) — где модель путает."""
    d = df.assign(_s=s)
    pos = d[d.label == 1]
    rows = []
    for c, g in d[d.label == 0].groupby("cls"):
        if len(g) >= min_n and len(pos):
            rows.append(dict(neg_class=c, n=len(g), auc=auc(np.r_[np.ones(len(pos)), np.zeros(len(g))], np.r_[pos._s.to_numpy(), g._s.to_numpy()]),
                             fpr=float((g._s >= 0.5).mean())))
    return pd.DataFrame(rows).sort_values("auc") if rows else pd.DataFrame(columns=["neg_class", "n", "auc", "fpr"])


def stratified_auc(y: np.ndarray, s: np.ndarray, strat: np.ndarray, bins: int = 4, min_each: int = 8) -> dict:
    """AUC внутри страт признака (квантили яркости): если шорткат даёт всё качество, AUC внутри страты падает к 0.5."""
    q = pd.qcut(pd.Series(strat).rank(method="first"), bins, labels=False).to_numpy()
    per, w = [], []
    for b in range(bins):
        m = q == b
        if (y[m] == 1).sum() >= min_each and (y[m] == 0).sum() >= min_each:
            per.append(auc(y[m], s[m]))
            w.append(m.sum())
    return dict(per_bin=[round(a, 3) for a in per], mean=float(np.average(per, weights=w)) if per else float("nan"))


# ------------------------------------------------------------------------------------------ безопасный формат модели
def _lr_to_json(pipe) -> dict:
    sc, lr = pipe.steps[0][1], pipe.steps[-1][1]
    return dict(kind="logreg", mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))


def predict_lr_json(d: dict, X: np.ndarray) -> np.ndarray:
    z = ((np.asarray(X) - np.array(d["mean"])) / np.array(d["scale"])) @ np.array(d["coef"]) + d["intercept"]
    return 1 / (1 + np.exp(-z))


def isotonic_to_json(iso: IsotonicRegression) -> dict:
    return dict(kind="isotonic", x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist())


def apply_isotonic_json(d: dict, s: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(s, float), d["x"], d["y"])


def save_bundle(path: Path, backbones: list[str], X_by_bb: dict[str, np.ndarray], y: np.ndarray, oof: np.ndarray | None, meta: dict, seed: int = 0) -> Path:
    """Фото-модель-пакет без pickle: по логрегрессии на каждый backbone (JSON) + калибровка (JSON) + manifest.

    Оценка пакета = среднее вероятностей по backbone'ам → изотоническая калибровка по OOF-оценкам (если переданы).
    """
    path.mkdir(parents=True, exist_ok=True)
    parts = {}
    for bb in backbones:
        m = zoo(seed)["logreg"]()
        m.fit(X_by_bb[bb], y)
        parts[bb] = _lr_to_json(m)
    (path / "logreg.json").write_text(json.dumps(parts), encoding="utf-8")
    cal = None
    if oof is not None:
        cal = isotonic_to_json(IsotonicRegression(out_of_bounds="clip").fit(oof, y))
        (path / "isotonic.json").write_text(json.dumps(cal), encoding="utf-8")
    manifest = dict(kind="photo_classifier", format=1, backbones=backbones, calibrated=cal is not None, size=224, preprocess="to_square_rgb (reflect pad) + per-backbone mean/std",
                    **meta)
    (path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_bundle(path: Path) -> dict:
    return dict(manifest=json.loads((path / "manifest.json").read_text(encoding="utf-8")),
                logreg=json.loads((path / "logreg.json").read_text(encoding="utf-8")),
                isotonic=json.loads((path / "isotonic.json").read_text(encoding="utf-8")) if (path / "isotonic.json").exists() else None)


def score_bundle(bundle: dict, X_by_bb: dict[str, np.ndarray]) -> np.ndarray:
    p = np.mean([predict_lr_json(bundle["logreg"][bb], X_by_bb[bb]) for bb in bundle["manifest"]["backbones"]], axis=0)
    return apply_isotonic_json(bundle["isotonic"], p) if bundle["isotonic"] else p
