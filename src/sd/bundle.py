"""Пакет классификатора цикла БЕЗ pickle: переносится на другую машину и проверяется без выполнения чужого кода.

Состав каталога `models/cycle/<имя>/`:
  manifest.json      — формат, список признаков, члены ансамбля, калибровка, метрики кросс-валидации, чем обучен (число циклов/видео, дата);
  lgbm_<имя>.txt     — LightGBM (текстовый дамп дерева);
  members.json       — логрегрессии и MLP (веса и импутация числами);
  isotonic.json      — калибровка ансамбля.
Оценка = среднее вероятностей членов → изотоническая калибровка по out-of-fold-оценкам ансамбля (если они были получены).
Нужные признаки перечислены в manifest.json — `recognize` умеет проверить, что все экстракторы подключены, прежде чем оценивать.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

FORMAT = 2
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=4, max_depth=2, min_child_samples=8, feature_fraction=0.7, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=5.0, n_estimators=150, verbose=-1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _prep(X: pd.DataFrame, feats: list[str], median: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    a = X[feats].to_numpy(float)
    a = np.where(np.isnan(a), median[None, :], a)
    return (a - mean) / scale


# ------------------------------------------------------------------------------------------ члены ансамбля
def fit_member(kind: str, X: pd.DataFrame, y: np.ndarray, feats: list[str], w: np.ndarray | None = None, seed: int = 0, **kw) -> dict:
    """Обучает одного члена и возвращает его состояние (словарь из чисел/строк; для LightGBM — ещё `_booster_text`)."""
    feats = [c for c in feats if X[c].notna().any()]
    if kind == "lgbm":
        import lightgbm as lgb

        m = lgb.LGBMClassifier(**{**LGB, **kw}, random_state=seed).fit(X[feats], y, sample_weight=w)
        return dict(kind="lgbm", features=feats, _booster_text=m.booster_.model_to_string())
    a = X[feats].to_numpy(float)
    median = np.nanmedian(a, axis=0)
    a = np.where(np.isnan(a), median[None, :], a)
    mean, scale = a.mean(0), np.where(a.std(0) > 1e-9, a.std(0), 1.0)
    z = (a - mean) / scale
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=kw.get("C", 0.3), max_iter=2000, class_weight="balanced").fit(z, y, sample_weight=w)
        return dict(kind="logreg", features=feats, median=median.tolist(), mean=mean.tolist(), scale=scale.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        h = int(kw.get("hidden", 12))
        mlp = MLPClassifier(hidden_layer_sizes=(h,), alpha=kw.get("alpha", 0.5), max_iter=2000, random_state=seed, early_stopping=False).fit(z, y)
        return dict(kind="mlp", features=feats, median=median.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                    W1=mlp.coefs_[0].tolist(), b1=mlp.intercepts_[0].tolist(), W2=mlp.coefs_[1].tolist(), b2=mlp.intercepts_[1].tolist())
    raise KeyError(kind)


def predict_member(st: dict, X: pd.DataFrame, booster_text: str | None = None) -> np.ndarray:
    k = st["kind"]
    if k == "lgbm":
        import lightgbm as lgb

        bst = lgb.Booster(model_str=booster_text or st["_booster_text"])
        return bst.predict(X[st["features"]].to_numpy(float))
    z = _prep(X, st["features"], np.array(st["median"]), np.array(st["mean"]), np.array(st["scale"]))
    if k == "logreg":
        return _sigmoid(z @ np.array(st["coef"]) + st["intercept"])
    if k == "mlp":
        h = np.maximum(z @ np.array(st["W1"]) + np.array(st["b1"]), 0)
        return _sigmoid(h @ np.array(st["W2"]) + np.array(st["b2"]))[:, 0]
    raise KeyError(k)


# ------------------------------------------------------------------------------------------ ансамбль, CV и пакет
def oof_members(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, spec: dict[str, dict], n_splits: int = 5, repeats: int = 1, seed: int = 0) -> dict[str, np.ndarray]:
    """Out-of-fold оценки каждого члена (фолды по группам = видео). При repeats > 1 оценки усредняются по перемешиваниям."""
    out = {n: np.zeros(len(y)) for n in spec}
    cnt = np.zeros(len(y))
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=min(n_splits, len(set(groups))), shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            for n, s in spec.items():
                st = fit_member(s["kind"], X.iloc[tr], y[tr], s["features"], seed=seed + r, **s.get("params", {}))
                out[n][te] += predict_member(st, X.iloc[te])
            cnt[te] += 1
    return {n: v / np.maximum(cnt, 1) for n, v in out.items()}


def train_bundle(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, spec: dict[str, dict], out_dir: Path, meta: dict | None = None, n_splits: int = 5,
                 repeats: int = 3, seed: int = 0, calibrate: bool = True) -> dict:
    """OOF по членам → калибровка ансамбля → финальное обучение на всех данных → пакет на диск. Возвращает manifest."""
    from .feature_auc import _auc

    oof = oof_members(X, y, groups, spec, n_splits, repeats, seed)
    ens = np.mean([oof[n] for n in spec], axis=0)
    iso = IsotonicRegression(out_of_bounds="clip").fit(ens, y) if calibrate else None   # без калибровки оценка = среднее членов, как в out-of-fold: пороги, подобранные по oof, переносятся как есть
    members, boosters = {}, {}
    for n, s in spec.items():
        st = fit_member(s["kind"], X, y, s["features"], seed=seed, **s.get("params", {}))
        if st["kind"] == "lgbm":
            boosters[n] = st.pop("_booster_text")
            st["file"] = f"lgbm_{n}.txt"
        members[n] = st
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, txt in boosters.items():
        (out_dir / f"lgbm_{n}.txt").write_text(txt, encoding="utf-8")
    (out_dir / "members.json").write_text(json.dumps({n: m for n, m in members.items() if m["kind"] != "lgbm"}), encoding="utf-8")
    if iso is not None:
        (out_dir / "isotonic.json").write_text(json.dumps(dict(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist())), encoding="utf-8")
    else:
        (out_dir / "isotonic.json").unlink(missing_ok=True)
    feats = sorted({c for m in members.values() for c in m["features"]})
    manifest = dict(kind="cycle_classifier", format=FORMAT, created=time.strftime("%Y-%m-%d %H:%M:%S"), features=feats,
                    members={n: dict(kind=m["kind"], features=m["features"], **({"file": m["file"]} if "file" in m else {})) for n, m in members.items()},
                    calibrated=bool(calibrate), n=int(len(y)), n_pos=int(y.sum()), n_groups=int(len(set(groups))),
                    cv=dict(n_splits=n_splits, repeats=repeats, auc_members={n: round(_auc(y, oof[n]), 4) for n in spec}, auc_ensemble=round(_auc(y, ens), 4),
                            auc_ensemble_calibrated=round(_auc(y, iso.predict(ens)), 4) if iso is not None else None),
                    **(meta or {}))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame({"y": y, "group": groups, **{f"oof_{n}": v for n, v in oof.items()}, "oof_ensemble": ens}).to_csv(out_dir / "oof.csv", index=False)
    return manifest


class Bundle:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("kind") != "cycle_classifier":
            raise ValueError(f"{path}: не пакет классификатора цикла")
        jm = json.loads((self.path / "members.json").read_text(encoding="utf-8")) if (self.path / "members.json").exists() else {}
        self.members = {}
        for n, m in self.manifest["members"].items():
            if m["kind"] == "lgbm":
                self.members[n] = (dict(m), (self.path / m["file"]).read_text(encoding="utf-8"))
            else:
                self.members[n] = (jm[n], None)
        self.iso = json.loads((self.path / "isotonic.json").read_text(encoding="utf-8")) if self.manifest.get("calibrated") else None

    @property
    def features(self) -> list[str]:
        return self.manifest["features"]

    def missing(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.features if c not in df.columns]

    def score_members(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.features)
        return pd.DataFrame({n: predict_member(st, X, txt) for n, (st, txt) in self.members.items()}, index=df.index)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        p = self.score_members(df).mean(axis=1).to_numpy()
        return np.interp(p, self.iso["x"], self.iso["y"]) if self.iso else p
