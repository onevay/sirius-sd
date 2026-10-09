import json

import numpy as np
import pandas as pd
import pytest

from rt_helpers import make_bundle, make_pose_fn
from sd import gt as GT
from sd import preds_check as PC
from sd import profiles as PR
from sd.config import load_config
from sd.paths import video_id
from sd.realtime import replay as RP
from sd.realtime.engine import StreamEngine

G2 = [(5.0, 2.0, 1.5, 1.5), (16.0, 2.0, 1.5, 1.5)]
CFG = ["cycles.th_in=0.35", "cycles.th_out=0.55", "events.confidence_threshold=0.5"]
PROF = PR.Profile("t", config={"cycles.th_in": 0.35, "cycles.th_out": 0.55, "events.confidence_threshold": 0.5}, options={"allow_heuristic": True})


def frames(video, start, end, fps, dur=45.0):
    for i in range(int(dur * fps)):
        yield np.full((120, 160, 3), (i * 5) % 255, np.uint8), i / fps


def engine(**kw):
    return StreamEngine(load_config(overrides=CFG), "clip", make_pose_fn(G2), keep_frames=False, **kw)


def test_replay_without_gt_gives_alerts_stats_and_valid_events(tmp_path):
    r = RP.replay_video(tmp_path / "clip.mp4", PROF, engine=engine(), frames_fn=frames, out_root=tmp_path / "rp")
    assert len(r.alerts) == 1 and r.metrics is None and r.report is None
    a = r.alerts.iloc[0]
    assert a.rule == "two_cycles" and a.t_open > a.end and 0 < a.delay < 20 and a.closed
    s = r.stats
    assert s["frames"] == 450 and s["stream_sec"] == pytest.approx(44.9, abs=0.2) and s["alerts"] == 1 and s["rt_factor"] > 1 and "справляется" in s["verdict"] and s["cycles"] == 2
    assert PC.validate(r.events.assign(x1=r.events.x1.fillna(0), y1=r.events.y1.fillna(0), x2=r.events.x2.fillna(5), y2=r.events.y2.fillna(5))) == []
    assert (tmp_path / "rp").exists() and json.loads((tmp_path / "rp" / sorted(p.name for p in (tmp_path / "rp").iterdir())[0] / "summary.json").read_text())["stats"]["alerts"] == 1
    assert set(r.log.kind) >= {"open", "close"}


def test_replay_with_gt_reports_metrics_and_alert_delay(tmp_path):
    video = tmp_path / "курение" / "clip.mp4"
    cid = video_id(video)
    gt = tmp_path / "gt.csv"
    GT.add(cid, 3.0, 20.0, "POSITIVE", person=1, box=(60, 30, 140, 200), path=gt)
    GT.add("other_clip", 1.0, 9.0, "POSITIVE", person=1, box=(0, 0, 10, 10), path=gt)           # чужой клип не мешает
    r = RP.replay_video(video, PROF, engine=StreamEngine(load_config(overrides=CFG), cid, make_pose_fn(G2), keep_frames=False), frames_fn=frames, gt=GT.load(gt), save=False)
    m = r.metrics
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0) and m["f1"] == 1.0
    assert m["alert_delay_median"] == pytest.approx(r.alerts.iloc[0].t_open - 3.0) and 8 < m["alert_delay_median"] < 25        # от начала курения до тревоги
    assert r.out_dir is None
    empty = RP.replay_video(tmp_path / "unknown.mp4", PROF, engine=StreamEngine(load_config(overrides=CFG), "unknown__x", make_pose_fn(G2), keep_frames=False), frames_fn=frames,
                            gt=GT.load(gt), save=False)
    assert empty.metrics is None and any("нет разметки" in n for n in empty.notes)


def test_replay_gt_miss_counts_fn_and_window_limits_gt(tmp_path):
    video = tmp_path / "k" / "c.mp4"
    cid = video_id(video)
    gt = tmp_path / "gt.csv"
    GT.add(cid, 25.0, 40.0, "POSITIVE", person=1, box=(60, 30, 140, 200), path=gt)         # курения в эти секунды система не видела (нет жестов)
    GT.add(cid, 100.0, 110.0, "POSITIVE", person=1, box=(60, 30, 140, 200), path=gt)       # за пределами окна
    r = RP.replay_video(video, PROF, engine=StreamEngine(load_config(overrides=CFG), cid, make_pose_fn([]), keep_frames=False), frames_fn=frames, gt=GT.load(gt), end=45.0, save=False)
    assert (r.metrics["tp"], r.metrics["fn"]) == (0, 1) and r.metrics["alert_delay_median"] is None


def test_replay_requires_classifier_when_engine_not_injected(tmp_path):
    with pytest.raises(ValueError, match="обязателен"):
        RP.replay_video(tmp_path / "c.mp4", PR.default_profile(), pose_fn=make_pose_fn(G2), frames_fn=frames, save=False)
    b = make_bundle(tmp_path / "cycle" / "f", ["hold", "d_min", "mouth_dur", "p_wm_dist"])
    p = PR.Profile("c", options={"cycle_bundle": str(b)}, config={"cycles.th_in": 0.35, "cycles.th_out": 0.55})
    r = RP.replay_video(tmp_path / "c.mp4", p, pose_fn=make_pose_fn(G2), frames_fn=frames, save=False)
    assert r.stats["cycles"] == 2                                                          # реальный пакет классификатора оценил оба цикла


def test_replay_renders_overlay_with_boxes_and_alert_banner(tmp_path):
    pytest.importorskip("cv2")
    r = RP.replay_video(tmp_path / "clip.mp4", PROF, engine=engine(), frames_fn=lambda v, s, e, f: frames(v, s, e, f, 30.0), render=True, out_root=tmp_path / "rp")
    assert r.overlay and (tmp_path / "rp").exists() and __import__("pathlib").Path(r.overlay).stat().st_size > 1000


def test_verdict_texts():
    assert "справляется" in RP._verdict(dict(rt_factor=2.5, fps_proc=25, process_fps_target=10))
    v = RP._verdict(dict(rt_factor=0.5, fps_proc=5.0, process_fps_target=10))
    assert "не справляется" in v and "5.0 к/с при нужных 10" in v and RP._verdict(dict(rt_factor=None)) == "нет данных о скорости"
