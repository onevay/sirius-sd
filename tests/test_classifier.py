import numpy as np
import pandas as pd
import pytest

from sd import classifier as C


def _tab(n_videos=12, per=14, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_videos):
        for i in range(per):
            y = float(rng.random() < 0.4)
            rows.append(dict(video=f"v{v}", tid=1, peak_t=float(i), y=y, label="smoke" if y else "drink", hold=1 + y * signal + rng.normal(0, 1), dur=2 + rng.normal(0, 1), noise=rng.normal(0, 1)))
    return pd.DataFrame(rows)


def _spec(tab, feats=("hold", "dur", "noise")):
    return dict(name="t", set=None, features=list(feats), calibrate=False, cv=dict(n_splits=3, repeats=1, by_scene=False, seed=0),
                members={"lr": dict(kind="logreg", features=list(feats), params=dict(C=0.3)), "gb": dict(kind="lgbm", features=list(feats), params={"n_estimators": 30, "num_leaves": 3})})


def test_validate_catches_problems():
    t = _tab()
    s = _spec(t)
    assert C.validate(s, t) == []
    s["members"]["lr"]["params"] = {"bad": 1}
    s["members"]["gb"]["features"] = ["missing"]
    s["members"]["x"] = dict(kind="forest", features=["hold"])
    msg = " ".join(C.validate(s, t))
    assert "bad" in msg and "missing" in msg and "forest" in msg
    assert C.validate(dict(members={}), t)


def test_cv_predict_is_out_of_fold_and_detects_signal():
    t = _tab()
    out = C.cv_predict(t, _spec(t))
    y = t.y.to_numpy()
    from sd.feature_auc import _auc
    assert _auc(y, out["__ensemble__"]) > 0.75
    assert set(out) == {"lr", "gb", "__ensemble__"}


def test_label_shuffle_control_is_near_chance():
    t = _tab(signal=0.0)
    out = C.cv_predict(t, _spec(t), shuffle_labels=True)
    from sd.feature_auc import _auc
    assert _auc(t.y.to_numpy(), out["__ensemble__"]) < 0.7


def test_diagnose_reports_gap_and_importance():
    t = _tab()
    d = C.diagnose(t, _spec(t), n_perm=1, curve=(0.5, 1.0), importance=True)
    m = d["members"]
    assert (m.AUC_train >= m.AUC_oof - 1e-9).all()
    imp = d["importance"].set_index("признак").падение_AUC
    assert imp["hold"] > imp["noise"]
    assert len(d["curve"]) == 2 and d["verdict"]


def test_train_writes_loadable_bundle(tmp_path):
    from sd.bundle import Bundle

    t = _tab()
    man = C.train(t, _spec(t), tmp_path / "b")
    assert man["n"] == len(t) and man["calibrated"] is False
    b = Bundle(tmp_path / "b")
    sc = b.score(t)
    assert len(sc) == len(t)


def test_train_refuses_invalid_spec(tmp_path):
    t = _tab()
    s = _spec(t)
    s["members"]["lr"]["features"] = ["absent"]
    with pytest.raises(ValueError):
        C.train(t, s, tmp_path / "b")


def test_spec_roundtrip(tmp_path):
    t = _tab()
    s = _spec(t)
    C.save_spec(s, tmp_path)
    assert C.list_specs(tmp_path) == ["t"]
    assert C.load_spec("t", tmp_path)["members"]["lr"]["params"] == {"C": 0.3}
    with pytest.raises(ValueError):
        C.save_spec({**s, "name": "../x"}, tmp_path)
