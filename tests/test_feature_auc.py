"""feature_auc: стыковка меток с таблицей циклов по округлённому start, производные признаки, AUC с бутстрэпом по видео."""
import numpy as np
import pandas as pd

from sd import feature_auc as FA


def test_asof_merge_tolerates_csv_rounding():
    lab = pd.DataFrame(dict(video=["a", "a", "a"], tid=[1, 1, 2], start=[37.6, 48.1, 10.0], label=["smoke", "drink", "smoke"], y=[1, 0, 1]))
    cyc = pd.DataFrame(dict(video=["a", "a", "a"], tid=[1, 1, 3], start=[37.6049, 48.0961, 10.0], hold=[1.0, 2.0, 3.0]))
    m = FA._asof_merge(lab, cyc)
    got = m.set_index(["tid", "start"]).hold
    assert got.loc[(1, 37.6)] == 1.0 and got.loc[(1, 48.1)] == 2.0
    assert np.isnan(got.loc[(2, 10.0)]) if (2, 10.0) in got.index else True   # трека 2 в таблице циклов нет -> пары нет, строка не выдумывается


def test_asof_merge_rejects_far_starts():
    lab = pd.DataFrame(dict(video=["a"], tid=[1], start=[10.0], label=["smoke"], y=[1]))
    cyc = pd.DataFrame(dict(video=["a"], tid=[1], start=[10.5], hold=[1.0]))
    m = FA._asof_merge(lab, cyc, tol=0.06)
    assert m.start_cyc.isna().all()


def test_derived_xclip_and_videomae():
    t = pd.DataFrame({FA.XCLIP_SMOKE: [0.7, 0.1], "p_a person vaping": [0.1, 0.1], "p_a person eating": [0.2, 0.8],
                      "videomae_k_smoking": [0.6, 0.1], "videomae_k_drinking": [0.1, 0.5], "videomae_top1_p": [0.6, 0.5]})
    d = FA._derived(t)
    assert np.allclose(d.xclip_smoke_minus_max_other, [0.5, -0.7])
    assert np.allclose(d.xclip_smoke_or_vape.values, [0.8, 0.2])
    assert d.videomae_smoke_minus_drink.iloc[0] > 0 > d.videomae_smoke_minus_drink.iloc[1]


def test_bootstrap_auc_separates_informative_from_noise():
    rng = np.random.default_rng(0)
    n_vid, per = 14, 5
    y = np.tile([1, 0, 1, 0, 0], n_vid)
    t = pd.DataFrame(dict(video=np.repeat([f"v{i}" for i in range(n_vid)], per), y=y,
                          label=np.where(y == 1, "smoke", "touch_face")))
    t["hold"] = y * 1.5 + rng.normal(0, 0.5, len(t))
    t["d_min"] = rng.normal(0, 1, len(t))
    good, lo, hi = FA.auc_by_video_bootstrap(t, "hold", n_boot=200)
    assert good > 0.85 and lo > 0.6
    noise, nlo, nhi = FA.auc_by_video_bootstrap(t, "d_min", n_boot=200)
    assert nlo < 0.5 < nhi
    res = FA.per_feature_auc(t, n_boot=100)
    assert set(res.feature) == {"hold", "d_min"} and res.set_index("feature").loc["hold", "signal"] == "+"


def test_folder_of_does_not_mix_prefixes():
    f = FA.folder_of(pd.Series(["курение__1", "лжекурение__1_drink", "other__x"]))
    assert f.tolist() == ["smoking", "fake", "unknown"]   # «лжекурение__» не начинается с «курение__»: подстрока внутри слова не считается


def test_object_recall_by_label():
    t = pd.DataFrame(dict(label=["smoke", "smoke", "smoke", "drink", "touch_face"], y=[1, 1, 1, 0, 0],
                          obj_a_hit=[True, True, False, True, False]))
    r = FA.object_recall(t).set_index("detector").loc["obj_a_hit"]
    assert abs(r["smoke (recall)"] - 2 / 3) < 1e-9 and abs(r["не курение (FP rate)"] - 0.5) < 1e-9


def test_confound_check_flags_scene_feature():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(10):    # затяжки и часть негативов — из папки «курение»; остальные негативы — из «лжекурения»
        rows += [dict(video=f"курение__s{i}", y=1, label="smoke", hold=rng.normal(1, .3), scene=rng.normal(0, .3)) for _ in range(3)]
        rows += [dict(video=f"курение__n{i}", y=0, label="touch_face", hold=rng.normal(0, .3), scene=rng.normal(0, .3)) for _ in range(1)]
        rows += [dict(video=f"лжекурение__f{i}", y=0, label="drink", hold=rng.normal(0, .3), scene=rng.normal(3, .3)) for _ in range(3)]
    t = pd.DataFrame(rows)   # `scene` — признак, выдающий сцену: у клипов «лжекурения» он большой независимо от жеста
    c = FA.confound_check(t, ["hold", "scene"]).set_index("feature")
    assert c.loc["hold", "auc_in_smoking_dir"] > 0.9 and abs(c.loc["hold", "auc_dir_among_neg"] - 0.5) < 0.2
    assert c.loc["scene", "auc_dir_among_neg"] < 0.1        # «сцена» отлично различает папку среди одних негативов -> предупреждение
    assert c.loc["scene", "n_pos_dir"] == 30 and c.loc["scene", "n_neg_dir"] == 10


def test_feature_table_accepts_project_label_format(tmp_path):
    """Метки вкладки «Разметка» (video, peak_t, cx, cy) стыкуются с циклами без tid и start; unsure и далёкие по положению метки отбрасываются."""
    cyc = pd.DataFrame(dict(video=["a", "a", "b"], run=["r"] * 3, tid=[1, 2, 1], start=[10.0, 30.0, 5.0], end=[12.0, 31.5, 6.0], mouth_in=[10.4, 30.3, 5.2],
                            mouth_out=[11.5, 31.2, 5.9], peak_t=[11.0, 30.7, 5.5], hold=[1.1, 0.9, 0.7], d_min=[0.3, 0.4, 0.2], hand=[0, 1, 0], idx=[0, 1, 0]))
    cyc.to_parquet(tmp_path / "cycles_all.parquet")
    ds = pd.DataFrame(dict(video=["a", "a", "b"], tid=[1, 2, 1], start=[10.0, 30.0, 5.0], cx=[100.0, 500.0, 300.0], cy=[100.0, 200.0, 250.0]))
    ds.to_parquet(tmp_path / "dataset.parquet")
    lab = pd.DataFrame(dict(video=["a", "a", "b", "b"], tid=[9, 9, 9, 9], peak_t=[11.2, 30.6, 5.4, 5.5], cx=[110.0, 505.0, 305.0, 900.0], cy=[95.0, 205.0, 245.0, 900.0],
                            label=["smoke", "unsure", "drink", "smoke"]))
    lab.to_csv(tmp_path / "labels.csv", index=False)
    tab = FA.feature_table(tmp_path / "labels.csv", tmp_path, tmp_path / "dataset.parquet")
    assert len(tab) == 2                                                    # unsure выброшена, четвёртая метка далеко по положению и не нашла цикл
    t = tab.set_index("video")
    assert t.loc["a", "label"] == "smoke" and t.loc["a", "y"] == 1 and t.loc["b", "y"] == 0
    assert abs(t.loc["a", "dur"] - 2.0) < 1e-9 and abs(t.loc["b", "mouth_dur"] - 0.7) < 1e-9


def test_feature_table_empty_when_nothing_matches(tmp_path):
    cyc = pd.DataFrame(dict(video=["a"], run=["r"], tid=[1], start=[10.0], end=[12.0], mouth_in=[10.4], mouth_out=[11.5], peak_t=[11.0], hold=[1.1], d_min=[0.3], hand=[0], idx=[0]))
    cyc.to_parquet(tmp_path / "cycles_all.parquet")
    pd.DataFrame(dict(video=["a"], tid=[1], start=[10.0], cx=[1.0], cy=[1.0])).to_parquet(tmp_path / "dataset.parquet")
    pd.DataFrame(dict(video=["zzz"], peak_t=[11.0], cx=[1.0], cy=[1.0], label=["smoke"])).to_csv(tmp_path / "labels.csv", index=False)
    assert FA.feature_table(tmp_path / "labels.csv", tmp_path, tmp_path / "dataset.parquet").empty


def test_auc_nan_for_constant_or_single_class():
    assert np.isnan(FA._auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))
    assert np.isnan(FA._auc(np.array([0, 1, 0]), np.array([1.0, 1.0, 1.0])))
