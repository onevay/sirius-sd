import numpy as np
import pandas as pd

from sd import oof_eval as O


def test_match_gestures_ignores_track_id_but_prefers_same_track():
    g = pd.DataFrame([dict(video="v", tid=2, peak_t=10.0, label="smoke", start_quality="ok"), dict(video="v", tid=5, peak_t=10.4, label="drink", start_quality=None),
                      dict(video="w", tid=1, peak_t=3.0, label="unsure", start_quality=None)])
    c = pd.DataFrame([dict(video="v", tid=4, peak_t=10.1), dict(video="v", tid=5, peak_t=10.1), dict(video="v", tid=1, peak_t=30.0), dict(video="w", tid=9, peak_t=3.9)])
    r = O.match_gestures(c, g)
    assert [None if pd.isna(x) else x for x in r.label] == ["smoke", "drink", None, "unsure"]          # другой трек: ближайший по времени; тот же трек предпочитается; далеко — нет метки; допуск 1 с


def test_oof_scores_never_train_on_own_video():
    rng = np.random.default_rng(0)
    n = 160
    vid = np.repeat([f"v{i}" for i in range(8)], n // 8)
    y = (rng.random(n) < 0.4).astype(float)
    tab = pd.DataFrame(dict(video=vid, y=y, hold=y * 2 + rng.normal(0, 1, n), dur=rng.normal(0, 1, n), mouth_dur=y + rng.normal(0, 1, n), d_min=rng.normal(0, 1, n)))
    tab.loc[::7, "y"] = np.nan                                              # неразмеченные циклы получают оценку, но не обучают
    s = O.oof_scores(tab, "kin", n_splits=4, repeats=1, kinds=("lr",))
    assert s.shape == (n,) and np.isfinite(s).all() and ((0 <= s) & (s <= 1)).all()
    m = np.isfinite(tab.y.to_numpy())
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(tab.y[m], s[m]) > 0.65                              # сигнал есть, оценка получена вне обучающих видео
    # «утечка»: если дать модели свою же метку как признак в одном видео, oof всё равно честен — проверяем, что оценка видео не зависит от его собственных меток
    t2 = tab.copy()
    t2.loc[t2.video == "v0", "y"] = 1 - t2.loc[t2.video == "v0", "y"]        # портим метки одного видео
    s2 = O.oof_scores(t2, "kin", n_splits=4, repeats=1, kinds=("lr",))
    assert np.allclose(s[tab.video == "v0"], s2[t2.video == "v0"])           # его оценки получены без его меток → не изменились


def test_scene_group_merges_same_camera_clips():
    g = O.scene_group
    assert g("лжекурение__smoking_noyabrsk_2") == g("лжекурение__smoking_noyabrsk_3") == "noyabrsk"
    assert g("курение__smoking_Jar_1") == g("лжекурение__drinking_Jar") == "jar"
    assert g("лжекурение__2025_11_06_08_13_58") == g("лжекурение__2025_11_06_08_35_37") == "woman_1106"
    assert g("курение__sm_6") == "курение__sm_6"                              # остальные — сами по себе


def test_oof_by_scene_keeps_scene_out_of_its_own_training():
    import numpy as np

    rng = np.random.default_rng(1)
    vid = np.array(["лжекурение__smoking_noyabrsk_2"] * 30 + ["лжекурение__smoking_noyabrsk_3"] * 30 + [f"v{i}" for i in range(6) for _ in range(10)])
    y = (rng.random(len(vid)) < 0.5).astype(float)
    tab = pd.DataFrame(dict(video=vid, y=y, hold=y + rng.normal(0, 1, len(vid)), dur=rng.normal(0, 1, len(vid)), mouth_dur=rng.normal(0, 1, len(vid)), d_min=rng.normal(0, 1, len(vid))))
    a = O.oof_scores(tab, "kin", n_splits=4, repeats=1, kinds=("lr",), by_scene=True)
    t2 = tab.copy()
    m = t2.video.str.contains("noyabrsk")
    t2.loc[m, "y"] = 1 - t2.loc[m, "y"]                                        # портим метки обеих частей сцены
    b = O.oof_scores(t2, "kin", n_splits=4, repeats=1, kinds=("lr",), by_scene=True)
    assert np.allclose(a[m.to_numpy()], b[m.to_numpy()])                       # оценки сцены не зависят от её меток: соседний клип той же сцены в обучение не попал
