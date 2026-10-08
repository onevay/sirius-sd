import json

import numpy as np
import pandas as pd

from sd import bundle as B


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    df = pd.DataFrame(dict(a=rng.normal(size=2 * n) + 1.5 * y, b=rng.normal(size=2 * n) + 0.8 * y, c=rng.normal(size=2 * n), d=np.nan))
    df.loc[rng.random(2 * n) < 0.1, "b"] = np.nan                   # пропуски: импутация и бустинг должны их переживать
    groups = np.repeat(np.arange(2 * n // 4), 4)
    return df, y, groups


SPEC = {"lr": dict(kind="logreg", features=["a", "b", "c", "d"]), "gb": dict(kind="lgbm", features=["a", "b", "c"]),
        "nn": dict(kind="mlp", features=["a", "b", "c"], params=dict(hidden=6))}


def test_members_fit_predict_and_survive_nan_and_empty_columns():
    X, y, _ = _data()
    for n, s in SPEC.items():
        st = B.fit_member(s["kind"], X, y, s["features"], **s.get("params", {}))
        p = B.predict_member(st, X)
        assert p.shape == (len(y),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        from sklearn.metrics import roc_auc_score

        assert roc_auc_score(y, p) > 0.8, n
    assert "d" not in B.fit_member("logreg", X, y, ["a", "d"])["features"]      # колонка без значений отброшена


def test_bundle_roundtrip_is_pickle_free_and_reproduces_scores(tmp_path):
    X, y, g = _data()
    man = B.train_bundle(X, y, g, SPEC, tmp_path / "b", meta=dict(note="тест"), n_splits=4, repeats=2)
    files = sorted(f.name for f in (tmp_path / "b").iterdir())
    assert files == ["isotonic.json", "lgbm_gb.txt", "manifest.json", "members.json", "oof.csv"]
    assert man["cv"]["auc_ensemble"] > 0.85 and man["note"] == "тест" and man["n_groups"] == len(set(g))
    assert json.loads((tmp_path / "b" / "manifest.json").read_text(encoding="utf-8"))["format"] == B.FORMAT
    b = B.Bundle(tmp_path / "b")
    assert b.missing(X) == [] and b.missing(X.drop(columns=["a"])) == ["a"]
    s = b.score(X)
    assert s.shape == (len(y),) and (np.diff(np.sort(s)) >= -1e-12).all() and 0 <= s.min() and s.max() <= 1
    # загрузка с нуля даёт те же числа, что и повторная загрузка (детерминизм формата)
    assert np.allclose(B.Bundle(tmp_path / "b").score(X), s)
    mem = b.score_members(X)
    assert list(mem.columns) == ["lr", "gb", "nn"]


def test_score_works_when_extra_or_missing_nonrequired_columns_present(tmp_path):
    X, y, g = _data()
    B.train_bundle(X, y, g, {"lr": SPEC["lr"]}, tmp_path / "b", n_splits=3, repeats=1)
    b = B.Bundle(tmp_path / "b")
    Z = X.assign(extra=1.0)
    assert np.allclose(b.score(Z), b.score(X))


def test_uncalibrated_bundle_scores_equal_member_mean(tmp_path):
    """Без калибровки оценка пакета = среднее членов (как в out-of-fold): пороги, подобранные по oof-оценкам, переносятся без пересчёта шкалы."""
    X, y, g = _data()
    man = B.train_bundle(X, y, g, SPEC, tmp_path / "b", n_splits=4, repeats=1, calibrate=False)
    assert man["calibrated"] is False and man["cv"]["auc_ensemble_calibrated"] is None
    assert not (tmp_path / "b" / "isotonic.json").exists()
    b = B.Bundle(tmp_path / "b")
    assert b.iso is None and np.allclose(b.score(X), b.score_members(X).mean(axis=1))
