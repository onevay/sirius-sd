import json

import numpy as np
import pandas as pd

from sd import photo_clf as PC


def _blobs(n=200, d=16, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    X = rng.normal(size=(2 * n, d))
    X[y == 1, :4] += sep
    groups = np.arange(2 * n)
    return X, y, groups


def test_oof_auc_high_on_separable_and_chance_on_noise():
    X, y, g = _blobs()
    for kind in ("logreg", "svm", "et"):
        assert PC.auc(y, PC.oof_scores(X, y, g, kind)) > 0.9, kind
    rng = np.random.default_rng(1)
    noise = PC.auc(y, PC.oof_scores(rng.normal(size=X.shape), y, g, "logreg"))
    assert 0.4 < noise < 0.6


def test_groups_are_not_split_between_train_and_test():
    from sklearn.model_selection import StratifiedGroupKFold

    X, y, _ = _blobs(60)
    groups = np.repeat(np.arange(len(y) // 4), 4)           # по 4 снимка в группе (дубли)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])


def test_bundle_roundtrip_matches_sklearn_and_has_no_pickle(tmp_path):
    X, y, g = _blobs(120, d=8)
    X2 = X + np.random.default_rng(2).normal(scale=0.1, size=X.shape)
    oof = PC.oof_scores(X, y, g, "logreg")
    p = PC.save_bundle(tmp_path / "b", ["a", "b"], {"a": X, "b": X2}, y, oof, dict(note="t"))
    assert sorted(f.name for f in p.iterdir()) == ["isotonic.json", "logreg.json", "manifest.json"]
    assert json.loads((p / "manifest.json").read_text())["backbones"] == ["a", "b"]
    bundle = PC.load_bundle(p)
    s = PC.score_bundle(bundle, {"a": X, "b": X2})
    ref = np.mean([PC.fit_predict("logreg", X, y, X), PC.fit_predict("logreg", X2, y, X2)], axis=0)
    ref = PC.apply_isotonic_json(bundle["isotonic"], ref)
    assert np.allclose(s, ref, atol=1e-9)
    assert PC.auc(y, s) > 0.9


def test_stratified_auc_and_per_negative():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    strat = rng.normal(size=n)
    s_good = y + rng.normal(scale=0.5, size=n)
    out = PC.stratified_auc(y, s_good, strat)
    assert len(out["per_bin"]) == 4 and out["mean"] > 0.8
    s_shortcut = strat + 3 * y * 0                  # оценка = страта, не зависит от класса
    assert PC.stratified_auc(y, s_shortcut, strat)["mean"] < 0.65
    df = pd.DataFrame(dict(label=np.r_[np.ones(30), np.zeros(60)].astype(int), cls=["smoking"] * 30 + ["drinking"] * 30 + ["phoning"] * 30))
    sc = np.r_[np.full(30, 0.9), np.full(30, 0.1), np.linspace(0.2, 0.95, 30)]
    t = PC.per_negative_auc(df, sc, min_n=20)
    assert list(t.neg_class)[0] == "phoning" and t.auc.iloc[0] < t.auc.iloc[1] == 1.0


def test_rank01_and_summarize():
    assert np.allclose(PC.rank01(np.array([3.0, 1.0, 2.0])), [1.0, 1 / 3, 2 / 3])
    r = PC.summarize(np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.3, 0.6]))
    assert r["recall"] == 0.5 and r["fpr"] == 0.5 and r["n"] == 4


def test_ov_parity_reports_cosine_between_runtimes(monkeypatch, tmp_path):
    """Сверка конвертации: косинус OpenVINO-эмбеддингов с torch (подставные эмбеддеры); без картинок — синтетика."""
    import numpy as np

    from sd import photo_feats as PF

    class FakeTorch:
        def __init__(self, name): pass
        def embed(self, rgb): return np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (len(rgb), 1))

    class FakeOV:
        def __init__(self, name): pass
        def embed(self, rgb): return np.tile(np.array([[1.0, 0.5, 0.0, 0.0]], np.float32), (len(rgb), 1))   # косинус 1/sqrt(1.25) ≈ 0.894

    monkeypatch.setattr(PF, "TorchEmbedder", FakeTorch)
    monkeypatch.setattr(PF, "OVEmbedder", FakeOV)
    r = PF.ov_parity("clip", n=3, images=[])                                   # нет картинок → синтетические
    assert r["n"] == 3 and r["real"] is False and abs(r["cos_mean"] - 1 / np.sqrt(1.25)) < 1e-6 and abs(r["cos_min"] - r["cos_mean"]) < 1e-6
    import cv2

    cv2.imwrite(str(tmp_path / "a.jpg"), np.full((40, 30, 3), 100, np.uint8))
    r = PF.ov_parity("clip", n=3, images=[tmp_path / "a.jpg", tmp_path / "missing.jpg"])   # нечитаемый файл пропускается
    assert r["n"] == 1 and r["real"] is True
