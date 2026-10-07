"""Обучение классификатора циклов (руководство §4.3): LightGBM + групповая кросс-валидация + изотоническая калибровка.

Группировка по видео (актёр+камера+дубль) обязательна. Сравнение с бейзлайнами на ОДНОЙ валидации: логистическая регрессия на
кинематике (A+B), «только правила регламента» (эвристическая оценка паузы), и (при наличии) только X-CLIP. Бустинг должен их обыграть —
иначе берём более простую модель (правило руководства).
Модель, обученная на слабых метках (по папке), помечается `weak_only` и не годится для сдачи: только проверка, что конвейер работает.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .dataset import FEATURE_GROUPS, feature_columns
from .events import heuristic_cycle_score
from .cycles import Cycle

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=8, max_depth=3, min_child_samples=15, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, n_estimators=400, verbose=-1)
# монотонные ограничения: больше циклов / есть предмет -> не меньше уверенность (стабилизирует модель на малых данных)
MONOTONE = {"n_cycles_pm20": 1, "obj_hit": 1, "obj_max_conf": 1}


def _best_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ths = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, p >= t, zero_division=0) for t in ths]
    j = int(np.argmax(f))
    return float(f[j]), float(ths[j])


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return dict(n=len(y), auc=None, pr_auc=None, f1=None, th=None)
    f1, th = _best_f1(y, p)
    return dict(n=int(len(y)), pos=int(y.sum()), auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)), f1=f1, th=th)


def rules_baseline(df: pd.DataFrame) -> np.ndarray:
    """Бейзлайн «только правила»: эвристическая оценка паузы у рта (см. events.heuristic_cycle_score)."""
    return np.array([heuristic_cycle_score(Cycle(0, r.start, r.mouth_in, r.mouth_out, r.end, r.hold, int(r.hand), r.d_min)) for r in df.itertuples()])


def train_cycle_model(df: pd.DataFrame, out_dir: Path, groups: list[str] | None = None, n_splits: int = 5, seed: int = 42,
                      weak_weight: float = 0.3, manual_only_eval: bool = True) -> dict:
    """df — таблица циклов с колонками признаков, `y` (0/1), `group`, `label_source` (manual|weak)."""
    import lightgbm as lgb

    d = df[df.y.notna()].reset_index(drop=True)
    feats = feature_columns(groups, d)
    # колонки без единого значения бесполезны и ломают импутер
    feats = [c for c in feats if d[c].notna().any()]
    y = d.y.astype(int).values
    grp = d.group.values
    w = np.where(d.label_source == "manual", 1.0, weak_weight)
    n_groups = len(set(grp))
    res: dict = dict(n_samples=len(d), n_pos=int(y.sum()), n_groups=n_groups, features=feats, manual=int((d.label_source == "manual").sum()),
                     weak_only=bool((d.label_source != "manual").all()))
    if len(np.unique(y)) < 2 or n_groups < 3:
        res["error"] = "нужно >= 2 класса и >= 3 группы (видео) для групповой валидации"
        return res
    cv = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=seed)
    mono = [MONOTONE.get(c, 0) for c in feats]
    oof = {k: np.full(len(d), np.nan) for k in ("lgbm", "logreg", "rules")}
    kin = [c for c in feature_columns(["A_kinematics", "B_rhythm"], d) if d[c].notna().any()]
    for tr_i, va_i in cv.split(d[feats], y, grp):
        m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, monotone_constraints=mono)
        m.fit(d.loc[tr_i, feats], y[tr_i], sample_weight=w[tr_i])
        oof["lgbm"][va_i] = m.predict_proba(d.loc[va_i, feats])[:, 1]
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, C=0.5))
        lr.fit(d.loc[tr_i, kin], y[tr_i], logisticregression__sample_weight=w[tr_i])
        oof["logreg"][va_i] = lr.predict_proba(d.loc[va_i, kin])[:, 1]
    oof["rules"] = rules_baseline(d)
    ev_mask = (d.label_source == "manual").values if (manual_only_eval and (d.label_source == "manual").any()) else np.ones(len(d), bool)
    res["eval_on"] = "manual" if ev_mask.sum() != len(d) or (d.label_source == "manual").all() else "all (weak!)"
    res["metrics"] = {k: _metrics(y[ev_mask], v[ev_mask]) for k, v in oof.items()}
    # калибровка на out-of-fold предсказаниях бустинга
    calib = IsotonicRegression(out_of_bounds="clip").fit(oof["lgbm"][ev_mask], y[ev_mask])
    res["metrics"]["lgbm_calibrated"] = _metrics(y[ev_mask], calib.predict(oof["lgbm"][ev_mask]))
    # финальная модель на всех данных
    final = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, monotone_constraints=mono).fit(d[feats], y, sample_weight=w)
    imp = pd.Series(final.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    res["importance_gain"] = {k: round(float(v), 2) for k, v in imp.head(15).items()}
    # абляция по группам признаков (по одной группе убираем, смотрим AUC на тех же фолдах)
    abl = {}
    for g, cols in FEATURE_GROUPS.items():
        keep = [c for c in feats if c not in cols]
        if not keep or len(keep) == len(feats):
            continue
        p = np.full(len(d), np.nan)
        for tr_i, va_i in cv.split(d[feats], y, grp):
            m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, monotone_constraints=[MONOTONE.get(c, 0) for c in keep])
            m.fit(d.loc[tr_i, keep], y[tr_i], sample_weight=w[tr_i])
            p[va_i] = m.predict_proba(d.loc[va_i, keep])[:, 1]
        abl[f"без {g}"] = _metrics(y[ev_mask], p[ev_mask])
    res["ablation"] = abl
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=final, calib=calib, features=feats, version="cycle-lgbm-0.1", weak_only=res["weak_only"], params=LGB_PARAMS), out_dir / "cycle_model.joblib")
    (out_dir / "train_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(dict(video=d.video, peak_t=d.peak_t, y=y, label_source=d.label_source, **{f"oof_{k}": v for k, v in oof.items()})).to_csv(out_dir / "oof_predictions.csv", index=False)
    return res


def load_cycle_model(path: Path) -> dict:
    return joblib.load(path)


def score_cycles(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Калиброванная оценка цикла: score = calib(model(x))."""
    p = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    return bundle["calib"].predict(p)
