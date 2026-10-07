import numpy as np
import pandas as pd

from sd import cycle_models as CM


def _tab(n=90, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = pd.DataFrame(dict(video=[f"v{i // 3}" for i in range(n)], y=y, hold=rng.normal(size=n) + 1.2 * y, dur=rng.normal(size=n) + 0.8 * y, mouth_dur=rng.normal(size=n),
                          d_min=rng.normal(size=n), obj_any_max_conf=rng.random(n) * (0.5 + 0.5 * y), p_wm_dist=rng.normal(size=n), p_elbow_peak=rng.normal(size=n),
                          track_len=rng.normal(size=n) + 3 * y,                       # признак клипа: должен быть отброшен, даже будучи «идеальным»
                          vlm_yesno=np.where(rng.random(n) < 0.3, np.nan, rng.random(n)), rare=np.where(rng.random(n) < 0.8, np.nan, 1.0)))
    return t


def test_available_drops_confounds_and_sparse_columns():
    t = _tab()
    a = CM.available(["hold", "track_len", "rare", "vlm_yesno", "missing_col", "hold"], t)
    assert a == ["hold", "vlm_yesno"]                       # дубль убран, track_len/rare/нет такой колонки — отброшены (vlm_yesno заполнен на 70%)


def test_members_for_has_small_and_all_and_other_kinds():
    spec = CM.members_for(["hold", "dur", "d_min", "obj_any_max_conf"])
    assert set(spec) == {"lr_small", "lr_all", "gb", "nn"}
    assert spec["lr_small"]["features"] == ["hold", "dur", "obj_any_max_conf"]
    assert spec["gb"]["kind"] == "lgbm" and spec["nn"]["kind"] == "mlp"
    only = CM.members_for(["hold", "dur"], kinds=("lr",))
    assert list(only) == ["lr_small"]


def test_compare_and_train_roundtrip(tmp_path):
    t = _tab()
    sets = {"a": ["hold", "dur", "mouth_dur", "d_min"], "b": ["hold", "dur", "obj_any_max_conf", "p_wm_dist", "p_elbow_peak", "track_len"]}
    res = CM.compare(t, sets, repeats=2, n_splits=4, n_boot=40)
    assert list(res["набор"]) == ["a", "b"] and (res["AUC ансамбль"] > 0.6).all() and (res["признаков"] == [4, 5]).all()
    CM.SETS["tmp"] = sets["b"]
    man = CM.train(t, "tmp", tmp_path / "m", meta=dict(note="тест"), repeats=2)
    del CM.SETS["tmp"]
    assert man["feature_set"] == "tmp" and "track_len" not in man["features"] and man["cv"]["auc_ensemble"] > 0.6
    from sd.bundle import Bundle

    assert Bundle(tmp_path / "m").score(t).shape == (len(t),)
