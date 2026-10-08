"""Внешние датасеты «папка = класс»: id клипа, метка по имени папки, выборка клипов и сводка по классам."""
import numpy as np
import pandas as pd

from sd import clips as C
from sd import paths as P


def test_video_id_of_user_data_is_unchanged():
    assert P.video_id(P.DATA / "курение" / "sm_3.mp4") == "курение__sm_3"          # id существующих запусков не должен меняться


def test_external_video_id_includes_dataset_and_class():
    a = P.video_id(P.EXTERNAL / "hmdb51" / "hmdb51_org" / "smoke" / "v_a.avi")
    b = P.video_id(P.EXTERNAL / "kinetics" / "k400_subset" / "smoke" / "v_a.avi")
    assert a == "hmdb51_org__smoke__v_a" and b == "k400_subset__smoke__v_a" and a != b   # одинаковые класс и имя в разных датасетах не склеиваются


def test_weak_label_by_class_folder():
    ext = P.EXTERNAL / "hmdb51" / "hmdb51_org"
    assert P.weak_label(ext / "smoke" / "x.avi") == "smoking" and P.weak_label(ext / "Smoking Hookah" / "x.mp4") == "smoking"
    assert P.weak_label(ext / "drink" / "x.avi") == "fake" and P.weak_label(ext / "eat" / "x.avi") == "fake"
    assert P.weak_label(P.DATA / "курение" / "1.mp4") == "smoking" and P.weak_label(P.DATA / "лжекурение" / "1.mp4") == "fake"
    assert P.weak_label(P.DATA / "неизвестная_папка" / "1.mp4") is None                # вне data_external неизвестная папка метки не получает


def test_class_label_custom_positive_names():
    assert P.class_label("cig", {"cig"}) == "smoking" and P.class_label("smoking", {"cig"}) == "fake"


def _tree(tmp_path):
    for cls, n in (("smoke", 7), ("drink", 5), ("eat", 1)):
        d = tmp_path / cls
        d.mkdir()
        for i in range(n):
            (d / f"c{i}.mp4").write_bytes(b"x")
    return tmp_path


def test_pick_clips_limits_each_class_deterministically(tmp_path):
    root = _tree(tmp_path)
    a = C.pick_clips(root, per_class=3, seed=0)
    b = C.pick_clips(root, per_class=3, seed=0)
    assert a == b
    by = pd.Series([p.parent.name for p in a]).value_counts().to_dict()
    assert by == {"smoke": 3, "drink": 3, "eat": 1}                                   # меньше лимита — берём всё
    assert {p.parent.name for p in C.pick_clips(root, only_class={"drink"})} == {"drink"}
    assert len(C.pick_clips(root)) == 13


def test_summarize_by_class_reports_shares_and_rates():
    df = pd.DataFrame([
        dict(video="a", cls="smoke", label="smoking", cycles=3, events=1, person_min=0.5, h_med=300.0),
        dict(video="b", cls="smoke", label="smoking", cycles=0, events=0, person_min=0.5, h_med=280.0),
        dict(video="c", cls="drink", label="fake", cycles=2, events=1, person_min=1.0, h_med=200.0),
        dict(video="d", cls="drink", label="fake", error="RuntimeError: нет кадров"),
    ])
    s = C.summarize(df).set_index("cls")
    assert s.loc["smoke", "clips"] == 2 and s.loc["smoke", "with_cycle"] == 0.5 and abs(s.loc["smoke", "cycles_per_person_min"] - 3.0) < 1e-9
    assert s.loc["drink", "clips"] == 1 and s.loc["drink", "errors"] == 1              # битый клип считается ошибкой, а не клипом без циклов
    assert np.isclose(s.loc["smoke", "h_med_px"], 290.0)


def test_class_auc_clip_level_uses_max_cycle_score_and_zero_for_empty_clips():
    clips = pd.DataFrame([
        dict(video="s1", cls="smoke", label="smoking"), dict(video="s2", cls="smoke", label="smoking"), dict(video="s3", cls="smoke", label="smoking"),
        dict(video="d1", cls="drink", label="fake"), dict(video="d2", cls="drink", label="fake"), dict(video="e1", cls="eat", label="fake"), dict(video="bad", cls="eat", label="fake", error="RuntimeError: x"),
    ])
    cyc = pd.DataFrame([
        dict(video="s1", cls="smoke", score=0.9), dict(video="s1", cls="smoke", score=0.2), dict(video="s2", cls="smoke", score=0.7),     # s3 — циклов нет → 0
        dict(video="d1", cls="drink", score=0.8), dict(video="d1", cls="drink", score=0.1), dict(video="d2", cls="drink", score=0.3), dict(video="e1", cls="eat", score=0.4),
    ])
    r = C.class_auc(clips, cyc, n_boot=50)
    assert r["clips"] == 6 and r["positives"] == 3                                                   # битый клип не считается
    # оценки клипов: s1 .9, s2 .7, s3 0 | d1 .8, d2 .3, e1 .4 → пар (поз., нег.): 9, побед: s1 над всеми 3 + s2 над d2, e1 (2) + ничьих нет → 5/9
    assert abs(r["clip_auc"] - 5 / 9) < 1e-9
    by = {x["cls"]: x for x in r["per_class"]}
    assert abs(by["drink"]["auc"] - 3 / 6) < 1e-9 and abs(by["eat"]["auc"] - 2 / 3) < 1e-9 and by["drink"]["n"] == 2
    assert 0.0 <= r["lo"] <= r["clip_auc"] <= r["hi"] <= 1.0
    assert r["cycles"] == 7 and 0.0 <= r["cycle_auc"] <= 1.0


def test_class_auc_without_two_classes_is_nan_not_an_error():
    clips = pd.DataFrame([dict(video="a", cls="smoke", label="smoking")])
    r = C.class_auc(clips, pd.DataFrame([dict(video="a", cls="smoke", score=0.5)]))
    assert np.isnan(r["clip_auc"]) and r["per_class"] == []


def test_summarize_reports_reach_and_wrist_confidence_when_available():
    df = pd.DataFrame([
        dict(video="a", cls="smoke", label="smoking", cycles=0, events=0, person_min=0.1, h_med=200.0, d_min_track=0.5, wrist_conf=0.6),
        dict(video="b", cls="smoke", label="smoking", cycles=0, events=0, person_min=0.1, h_med=200.0, d_min_track=1.2, wrist_conf=0.4),
        dict(video="c", cls="smoke", label="smoking", cycles=1, events=0, person_min=0.1, h_med=200.0, d_min_track=0.9, wrist_conf=0.8),
    ])
    s = C.summarize(df, th_in=0.65).set_index("cls")
    assert abs(s.loc["smoke", "reach_th_in"] - 1 / 3) < 1e-9 and abs(s.loc["smoke", "wrist_conf_med"] - 0.6) < 1e-9
    assert "reach_th_in" not in C.summarize(df.drop(columns=["d_min_track", "wrist_conf"])).columns        # без диагностики колонок нет
