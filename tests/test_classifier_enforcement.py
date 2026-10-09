from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rt_helpers import make_bundle, make_pose_fn, run_stream
from sd import gt as GT
from sd import profiles as PR
from sd import runner as R
from sd.paths import video_id
from sd.realtime import scoring as SC
from sd.realtime import worker as W
from sd.realtime.engine import StreamEngine

FAST = ["hold", "d_min", "mouth_dur", "p_wm_dist"]
G2 = [(5.0, 2.0, 1.5, 1.5), (16.0, 2.0, 1.5, 1.5)]


def world(tmp_path):
    d = tmp_path / "курение"
    d.mkdir()
    (d / "s1.mp4").write_bytes(b"x")
    f = tmp_path / "gt.csv"
    GT.add(video_id(d / "s1.mp4"), 8, 22, "POSITIVE", person=1, box=(0, 0, 100, 200), path=f)
    return d, GT.load(f)


def fake(video, start, end, profile):
    ev = pd.DataFrame([dict(camera_id="c", clip_id="s1", event_id="e", start_sec=10, end_sec=20, confidence=0.9, label="x", person_track_id=1, peak_sec=15, x1=0, y1=0, x2=100, y2=200)],
                      columns=R.EVENT_COLUMNS)
    return SimpleNamespace(events=ev, meta=dict(run_dir=None, out_dir=None, counts=dict(cycles=1, people=1), warnings=[]))


kw = dict(policy="fixed", recognize_fn=fake, probe_fn=lambda v: SimpleNamespace(duration=100.0), save=False, n_boot=10)


def test_evaluation_refuses_to_run_without_classifier(tmp_path):
    d, gt = world(tmp_path)
    with pytest.raises(ValueError, match="обязателен"):
        R.evaluate_dirs([d], PR.default_profile(), gt_df=gt, cache_root=tmp_path / "c", **kw)
    with pytest.raises(ValueError, match="обязателен"):
        R.evaluate_dirs([d], PR.default_profile(), mode="clips", cache_root=tmp_path / "c", **kw)


def test_classifier_is_chosen_as_part_of_the_check(tmp_path):
    d, gt = world(tmp_path)
    b = make_bundle(tmp_path / "cycle" / "fast1", FAST)
    out = R.evaluate_dirs([d], PR.default_profile(), classifier=str(b), gt_df=gt, cache_root=tmp_path / "c", **kw)
    assert out.report.metrics["tp"] == 1 and out.meta["describe"]["cycle_model"] == "fast1"
    assert PR.default_profile().options == {}                                        # выбор не меняет исходный профиль
    out2 = R.evaluate_dirs([d], PR.heuristic_profile(), classifier=str(make_bundle(tmp_path / "cycle" / "fast2", FAST, seed=5)), gt_df=gt, cache_root=tmp_path / "c", **kw)
    assert out2.meta["describe"]["cycle_model"] == "fast2"
    # разные классификаторы — разные отпечатки: кэш прогонов не смешивается
    assert out.meta["fingerprint"] != out2.meta["fingerprint"]


def test_evaluation_refuses_when_extractors_do_not_cover_classifier(tmp_path):
    d, gt = world(tmp_path)
    b = make_bundle(tmp_path / "cycle" / "heavy", FAST + ["obj_det_a_max_conf"], seed=2)
    with pytest.raises(ValueError, match="предмета детекторов"):
        R.evaluate_dirs([d], PR.default_profile(), classifier=str(b), gt_df=gt, cache_root=tmp_path / "c", **kw)
    R.evaluate_dirs([d], PR.Profile("ok", options={"objects": ["det_a"]}), classifier=str(b), gt_df=gt, cache_root=tmp_path / "c", **kw)


def test_pipeline_recognize_requires_classifier_unless_allowed(tmp_path):
    from sd.pipeline import ClassifierRequired, Options, recognize

    with pytest.raises(ClassifierRequired):
        recognize(tmp_path / "x.mp4", 0.0, None, {}, Options())


def test_stream_engine_factory_requires_classifier_and_builds_scorer(tmp_path):
    with pytest.raises(ValueError, match="обязателен"):
        W.make_engine(PR.default_profile(), "c", pose_fn=make_pose_fn(G2))
    b = make_bundle(tmp_path / "cycle" / "fast1", FAST)
    eng = W.make_engine(PR.Profile("p", options={"cycle_bundle": str(b)}, config={"cycles.th_in": 0.35, "cycles.th_out": 0.55}), "c", pose_fn=make_pose_fn(G2), keep_frames=False)
    assert isinstance(eng.scorer, SC.ClassifierScorer)
    run_stream(eng, 45.0)
    assert eng.stats["cycles"] == 2 and eng.scorer.missing_last == []                  # классификатор получил все свои признаки
    heavy = make_bundle(tmp_path / "cycle" / "heavy", FAST + ["obj_det_a_max_conf"], seed=2)
    hp = PR.Profile("h", options={"cycle_bundle": str(heavy), "objects": ["det_a"]})
    with pytest.raises(ValueError, match="live_features|настоящем потоке"):
        W.make_engine(hp, "c", pose_fn=make_pose_fn(G2))                               # в настоящем потоке предмета нет — не запускаем, а не «молча без признаков»
    assert isinstance(W.make_engine(hp, "c", pose_fn=make_pose_fn(G2), mode="replay", source_path=tmp_path / "v.mp4").scorer, SC.ClassifierScorer)
    with pytest.raises(SC.FeatureUnavailable):
        SC.ClassifierScorer(heavy, {}, "c", objects=["det_a"], source_path=None)
    with pytest.raises(SC.FeatureUnavailable, match="детекторы"):
        SC.ClassifierScorer(heavy, {}, "c", objects=[], source_path=tmp_path / "v.mp4")


def test_scorer_adds_object_features_from_source_without_changing_base_vector(tmp_path, monkeypatch):
    """Предмет считается экстрактором из файла и подаётся в тот же классификатор; быстрые признаки и их значения не меняются."""
    from sd import analysis as A

    heavy = make_bundle(tmp_path / "cycle" / "heavy", FAST + ["obj_det_a_max_conf"], seed=2)
    calls = []

    def fake_evidence(cyc, tr, ser, cfg, ids):
        calls.append((list(ids), tr.meta.get("video")))
        return [dict(video=r.video, tid=r.tid, start=r.start, obj_det_a_max_conf=0.8, obj_det_a_hit_frames=3, obj_det_a_hit=True, n_det_frames=3) for r in cyc.itertuples()]

    monkeypatch.setattr(A, "evidence_rows", fake_evidence)
    from sd.config import load_config

    cfg = load_config(overrides=["cycles.th_in=0.35", "cycles.th_out=0.55"])
    src_video = tmp_path / "clip.mp4"
    sc = SC.ClassifierScorer(heavy, cfg, "c", objects=["det_a"], source_path=src_video)
    eng = StreamEngine(cfg, "c", make_pose_fn(G2), sc, keep_frames=False, source_path=src_video)
    run_stream(eng, 45.0)
    assert calls and calls[0] == (["det_a"], str(src_video)) and sc.missing_last == []
