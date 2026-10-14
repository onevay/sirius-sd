"""Лаборатория: модели по временным рядам, активное обучение, дообучение с весами, ручки этапов и запись метки жеста (на синтетических данных, без видео и весов)."""
import numpy as np
import pandas as pd
import pytest

from sd import active as AC
from sd import classifier as CL
from sd import lab
from sd import tsmodels as TS


def _synthetic(n_videos: int = 10, per: int = 12, seed: int = 0):
    """Циклы и ряды: у «затяжек» d_t уходит к 0.3 и держится, у остальных — плавно колеблется около 1.2."""
    rng = np.random.default_rng(seed)
    cycles, series = [], {}
    for v in range(n_videos):
        t = np.arange(0, 400, 0.2)
        d = np.full(len(t), 1.5)
        rows = []
        for i in range(per):
            t0 = 5 + i * 30
            smoke = (i + v) % 2 == 0
            m = (t >= t0) & (t <= t0 + 4)
            d[m] = (0.3 + 0.05 * rng.normal(size=m.sum())) if smoke else (1.2 + 0.1 * rng.normal(size=m.sum()))
            rows.append(dict(video=f"v{v}", tid=1, start=t0, end=t0 + 4, peak_t=t0 + 2, hold=3.0 if smoke else 1.0, hand=0, y=float(smoke), label="smoke" if smoke else "drink"))
        cycles += rows
        series[f"v{v}"] = pd.DataFrame(dict(tid=1, t=t, d=d, d_raw=d, v=np.gradient(d, t), elbowL=np.full(len(t), 90.0), elbowR=np.full(len(t), 90.0), head_tilt=0.0,
                                            wrist_h=0.0, wrist_conf=0.9))
    return pd.DataFrame(cycles), series


def test_cycle_tensor_shapes_and_missing_series():
    cyc, ser = _synthetic(2, 3)
    X, S = TS.cycle_tensor(cyc, ser, ("d", "v", "elbow"), 32, 1.0)
    assert X.shape == (6, 3, 32) and S.shape == (6, 2)
    assert np.isfinite(X).all() and S[:, 0].min() > 5
    X2, _ = TS.cycle_tensor(cyc, {}, ("d",), 16, 1.0)          # нет рядов — NaN, не исключение
    assert np.isnan(X2).all()


@pytest.mark.parametrize("model", ["stats", "rocket"] + (["cnn1d"] if TS.torch_available() else []))
def test_ts_models_learn_save_load(model, tmp_path):
    cyc, ser = _synthetic(10, 10)
    spec = TS.TSSpec(model, ["d", "v"], 32, 1.0, {"epochs": 25} if model == "cnn1d" else {"n_kernels": 80} if model == "rocket" else {})
    X, S = TS.cycle_tensor(cyc, ser, spec.channels, spec.length, spec.pad)
    y = cyc.y.to_numpy(int)
    groups = cyc.video.to_numpy()
    oof = TS.cv_oof(X, S, y, groups, spec, n_splits=3, repeats=1)
    from sd.feature_auc import _auc

    assert _auc(y, oof) > 0.9                                   # форма «ушла к 0.3 и держится» отличима
    m = TS.TSModel(spec).fit(X, S, y)
    p = m.predict(X, S)
    m.save(tmp_path / "m")
    p2 = TS.TSModel.load(tmp_path / "m").predict(X, S)
    assert np.allclose(p, p2, atol=1e-5)
    assert not any(f.suffix == ".pkl" for f in (tmp_path / "m").iterdir())      # без pickle


def test_ts_blend_and_prototypes():
    cyc, ser = _synthetic(6, 8)
    X, S = TS.cycle_tensor(cyc, ser, ("d",), 24, 0.5)
    y = cyc.y.to_numpy(int)
    pr = TS.prototypes(X, y, ("d",))
    assert set(pr.cls) == {"затяжка", "не затяжка"}
    cv = TS.blend_curve(y, np.random.default_rng(0).random(len(y)), y * 0.8 + 0.1, cyc.video.to_numpy(), steps=3)
    assert list(cv.w) == [0.0, 0.5, 1.0] and cv.AUC.iloc[-1] > cv.AUC.iloc[0]


def _tab_and_spec(n: int = 160, seed: int = 1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    tab = pd.DataFrame(dict(video=[f"v{i % 16}" for i in range(n)], tid=1, start=np.arange(n) * 5.0, peak_t=np.arange(n) * 5.0 + 2, y=y.astype(float),
                            label=np.where(y == 1, "smoke", "drink"), f1=y + rng.normal(0, 0.8, n), f2=rng.normal(size=n)))
    spec = dict(name="t", members={"lr": dict(kind="logreg", features=["f1", "f2"], params={"C": 0.5}), "gb": dict(kind="lgbm", features=["f1", "f2"], params={"n_estimators": 30})},
                cv=dict(CL.DEFAULT_CV, repeats=1, n_splits=4))
    return tab, spec


def test_committee_priority_and_strategies():
    tab, spec = _tab_and_spec()
    train, pool = tab.iloc[:120], tab.iloc[120:].drop(columns=["y", "label"])
    cm = AC.committee(train, pool, spec)
    assert {"p", "std", "lr", "gb"} <= set(cm.columns) and len(cm) == 40 and cm.p.between(0, 1).all()
    space = AC._feature_space(pool.assign(f1=pool.f1), ["f1", "f2"])
    for s in AC.STRATEGIES:
        pos = AC.priority(cm, s, threshold=0.4, space=space, k=7, rng=np.random.default_rng(0))
        assert len(pos) == 7 and len(set(pos)) == 7 and pos.max() < len(cm)
    assert (np.abs(cm.p.to_numpy()[AC.priority(cm, "uncertainty", k=3)] - 0.5) <= np.abs(cm.p.to_numpy() - 0.5).max()).all()


def test_simulation_and_suspects():
    tab, spec = _tab_and_spec()
    g = AC.simulate(tab, spec, ("random", "uncertainty"), init=12, batch=6, rounds=2, folds=3, by_scene=False)
    assert set(g.strategy) == {"random", "uncertainty"} and (g.groupby("strategy").labels.max() > g.groupby("strategy").labels.min()).all()
    flipped = tab.copy()
    flipped.loc[3, "y"] = 1 - flipped.loc[3, "y"]
    sus = AC.suspects(flipped, spec, 5)
    assert 3 in sus.index or len(sus) == 5 and sus.расхождение.iloc[0] > 0.5


def test_weights_reach_training_and_finetune_cv(tmp_path):
    tab, spec = _tab_and_spec()
    w = AC.weights_for(tab, new_videos=["v1", "v2"], w_new=4.0)
    assert set(w._w.unique()) == {1.0, 4.0}
    assert (w[w.video.isin(["v1", "v2"])]._w == 4.0).all()
    r = AC.finetune_cv(tab, spec, new_videos=["v1", "v2", "v3"], w_new=3.0, use_hard=True, by_scene=False, repeats=1)
    assert list(r["table"].вариант) == ["без весов", "с весами"] and r["rows"]._w.max() >= 3.0
    man = AC.finetune_save(r["rows"], {**spec, "name": "ft"}, tmp_path / "ft", parent="models/cycle/base", params={"w_new": 3.0})
    assert man["parent"] == "models/cycle/base" and man["finetune"]["w_new"] == 3.0 and (tmp_path / "ft" / "manifest.json").exists()


def test_set_path_changed_and_diff_events():
    cfg = {"cycles": {"th_in": 0.65, "th_out": 0.9}, "features": {"mouth_offset": 0.35}}
    new = {"cycles": {"th_in": 0.5, "th_out": 0.9}, "features": {"mouth_offset": 0.35}}
    assert lab.changed(cfg, new) == {"cycles.th_in": (0.65, 0.5)}
    lab.set_path(new, "events.cycle_th", 0.3)
    assert new["events"]["cycle_th"] == 0.3
    base = pd.DataFrame(dict(start_sec=[10.0, 50.0], end_sec=[20.0, 60.0], confidence=[0.9, 0.8]))
    cur = pd.DataFrame(dict(start_sec=[11.0, 100.0], end_sec=[21.0, 110.0], confidence=[0.7, 0.9]))
    d = lab.diff_events(base, cur)
    assert sorted(d.статус) == ["осталось", "появилось", "пропало"]
    fig = lab.events_timeline(120.0, [("a", pd.DataFrame(dict(start=[1.0], end=[3.0])), "#16a34a")])
    assert len(fig.layout.shapes) == 1


def test_add_gesture_writes_version(tmp_path, monkeypatch):
    from sd import gestures as GS
    from sd import labels_history as LH
    from sd import start_eval as SE

    f = tmp_path / "gesture_gt.csv"
    monkeypatch.setattr(SE, "GT_CSV", f)
    monkeypatch.setattr(LH, "snapshot", lambda *a, **k: None)
    from sd.ui import lab_common as LC

    row = pd.Series(dict(video="курение__1", tid=3, peak_t=12.34, start=11.0))
    msg = LC.add_gesture(row, "smoke")
    df = GS.load()
    assert "затяжка" in msg and len(df) == 1 and df.iloc[0].label == "smoke" and abs(df.iloc[0].peak_t - 12.34) < 1e-6 and df.iloc[0].source == "active"
    with pytest.raises(ValueError):
        LC.add_gesture(row, "нет_такой_метки")
