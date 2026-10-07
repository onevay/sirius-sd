import numpy as np
import pandas as pd
import pytest

from sd import dataset as D
from sd.cycles import Cycle
from sd.dataset import attach_labels, feature_columns, save_label
from sd.train import rules_baseline, train_cycle_model


@pytest.fixture()
def tmp_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "LABELS", tmp_path)
    monkeypatch.setattr(D, "LABELS_CSV", tmp_path / "cycle_labels.csv")
    return tmp_path


def fake_table(n_videos=8, per=30, seed=0):
    """Синтетическая таблица циклов: «курение» — пауза дольше и ритмичнее; шум в остальном."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_videos):
        weak = "smoking" if v % 2 == 0 else "fake"
        for i in range(per):
            pos = (weak == "smoking") and rng.random() < 0.7
            hold = rng.normal(2.2 if pos else 1.6, 0.5)
            rows.append(dict(video=f"v{v}", group=f"v{v}", weak_label=weak, tid=1, start=10.0 * i, mouth_in=10.0 * i + 1, mouth_out=10.0 * i + 1 + hold,
                             end=10.0 * i + 2 + hold, peak_t=10.0 * i + 1 + hold / 2, cx=500.0, cy=300.0, hold=hold, d_min=rng.uniform(0.1, 0.3),
                             d_med_hold=rng.uniform(0.1, 0.4), d_std_hold=rng.uniform(0, 0.1), t_approach=1.0, t_retract=1.0, v_approach_max=1.0,
                             v_retract_max=1.0, amplitude=1.5, elbow_hold=90.0, elbow_range=20.0, hand=int(rng.integers(0, 2)),
                             head_tilt_hold=0.0, head_tilt_delta=0.0, wrist_h_hold=-0.2, nose_disp=0.1, wrist_disp=1.5,
                             n_cycles_pm20=int(rng.integers(0, 4)) + (2 if pos else 0), interval_prev=8.0, interval_next=8.0, interval_std=1.0,
                             same_hand_frac=0.8, time_at_mouth_frac60=0.1, H=300.0, s_px=60.0, wrist_conf=0.8, nose_conf=0.9, missing_frac=0.1,
                             overlap_iou=0.0, edge_dist=1.0, track_len=60.0, brightness=100.0, truth=int(pos)))
    return pd.DataFrame(rows)


def test_labels_roundtrip_and_matching(tmp_labels):
    t = fake_table(2, 3)
    t.loc[0, "peak_t"] = 2.35                                           # метка поставлена на 2.3 с: допуск по времени 1 с
    save_label("v0", 1, 2.3, 500, 300, "smoke", labeler="t")
    save_label("v0", 1, 2.3, 500, 300, "drink", labeler="t")            # перезапись той же метки
    assert len(D.load_labels()) == 1
    out = attach_labels(t, use_weak=False)
    assert out.loc[0, "label"] == "drink" and out.loc[0, "y"] == 0 and out.loc[0, "label_source"] == "manual"
    assert out.label.notna().sum() == 1                                 # соседние циклы (≥ 10 с) не задеты


def test_weak_labels_and_priority(tmp_labels):
    t = fake_table(2, 3)
    out = attach_labels(t, use_weak=True)
    assert (out[out.weak_label == "fake"].y == 0).all() and (out[out.weak_label == "smoking"].y == 1).all()
    assert (out.label_source == "weak").all()
    save_label("v0", 1, float(t.loc[0, "peak_t"]), 500, 300, "drink")
    out = attach_labels(t, use_weak=True)
    assert out.loc[0, "label_source"] == "manual" and out.loc[0, "y"] == 0   # ручная метка важнее слабой


def test_feature_columns_groups():
    kin = feature_columns(["A_kinematics"])
    assert "hold" in kin and "n_cycles_pm20" not in kin
    assert "obj_hit" in feature_columns(["D_object"])


def test_train_on_synthetic_beats_noise_and_is_group_split(tmp_path, tmp_labels):
    t = fake_table(10, 40)
    t["y"] = t.truth.astype(float)
    t["label_source"] = "manual"
    res = train_cycle_model(t, tmp_path / "m", n_splits=5)
    assert res["n_groups"] == 10 and not res["weak_only"]
    assert res["metrics"]["lgbm"]["auc"] > 0.8
    assert res["metrics"]["lgbm_calibrated"]["auc"] > 0.8
    assert (tmp_path / "m" / "cycle_model.joblib").exists()
    assert "без A_kinematics" in res["ablation"]


def test_train_needs_enough_groups(tmp_path):
    t = fake_table(2, 20)
    t["y"] = t.truth.astype(float)
    t["label_source"] = "manual"
    assert "error" in train_cycle_model(t, tmp_path / "m")


def test_rules_baseline_range():
    p = rules_baseline(fake_table(2, 5))
    assert ((p >= 0) & (p <= 1)).all()
