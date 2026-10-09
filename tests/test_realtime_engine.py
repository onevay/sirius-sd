import numpy as np
import pytest

from rt_helpers import make_pose_fn, run_stream
from sd.realtime.engine import AlertUpdate, FrameRing, StreamEngine
from sd.config import load_config

G2 = [(5.0, 2.0, 1.5, 1.5), (16.0, 2.0, 1.5, 1.5)]          # два жеста: паузы у рта ≈ 1.5 с, между ними ≈ 8 с


@pytest.fixture()
def rt_cfg():
    return load_config(overrides=["cycles.th_in=0.35", "cycles.th_out=0.55", "events.confidence_threshold=0.5"])


def test_two_cycles_raise_alert_with_backdated_start(rt_cfg):
    eng = StreamEngine(rt_cfg, "gatchina-01", make_pose_fn(G2), keep_frames=False)
    ups = run_stream(eng, 45.0)
    opens = [u for u in ups if u.kind == "open"]
    assert len(opens) == 1 and opens[0].rule == "two_cycles" and opens[0].n_cycles == 2 and opens[0].camera_id == "gatchina-01"
    a = opens[0]
    assert 3.5 <= a.start <= 5.6                       # начало — начало подъёма руки в ПЕРВОМ цикле (задним числом)
    assert a.t_now > a.end                              # тревога приходит после закрытия второго цикла
    assert a.t_now - a.end <= 3.0                       # …но с небольшим запаздыванием «осаждения»
    assert a.box == pytest.approx((60.0, 30.0, 140.0, 200.0)) and a.confidence >= 0.5
    assert a.key == f"gatchina-01:1:{a.start:.2f}"
    assert [u.kind for u in ups if u.key == a.key][-1] == "close"      # событие закрылось, когда пауза превысила склейку 15 с


def test_single_cycle_without_object_gives_no_alert(rt_cfg):
    eng = StreamEngine(rt_cfg, "c", make_pose_fn(G2[:1]), keep_frames=False)
    assert [u for u in run_stream(eng, 40.0) if u.kind == "open"] == []


def test_cycles_far_apart_do_not_merge_into_alert(rt_cfg):
    eng = StreamEngine(rt_cfg, "c", make_pose_fn([(5.0, 2.0, 1.5, 1.5), (40.0, 2.0, 1.5, 1.5)]), keep_frames=False)
    assert [u for u in run_stream(eng, 60.0) if u.kind == "open"] == []     # 35 с между циклами > окна 20 с и склейки 15 с


def test_small_person_is_ignored(rt_cfg):
    eng = StreamEngine(rt_cfg, "c", make_pose_fn(G2, box=(60.0, 30.0, 140.0, 90.0)), keep_frames=False)    # высота 60 px < 80
    assert run_stream(eng, 45.0) == []


def test_threshold_blocks_low_confidence_event():
    cfg = load_config(overrides=["cycles.th_in=0.35", "cycles.th_out=0.55", "events.confidence_threshold=0.999"])
    eng = StreamEngine(cfg, "c", make_pose_fn(G2), keep_frames=False)
    assert [u for u in run_stream(eng, 45.0) if u.kind == "open"] == []


def test_stream_matches_offline_event_on_same_input(rt_cfg):
    """Поток и offline на одном и том же входе дают одно событие с теми же границами (допуск — один шаг кадра)."""
    import pandas as pd

    from sd.cycles import find_cycles
    from sd.events import ScoredCycle, build_events, heuristic_cycle_score
    from sd.features import build_series, track_grid
    from sd.tracks import Tracks

    pf = make_pose_fn(G2)
    rows, kps, ft = [], [], []
    for i in range(450):
        t = i / 10
        ft.append((i, t))
        for d in pf(None, t):
            rows.append((i, t, 1, *d.box.tolist(), 0.9, 170.0))
            kps.append(d.kp)
    tr = Tracks(pd.DataFrame(rows, columns=["frame", "t", "tid", "x1", "y1", "x2", "y2", "score", "h"]), np.stack(kps), pd.DataFrame(ft, columns=["frame", "t"]), {})
    r, kp = tr.of(1)
    s = build_series(track_grid(tr.frame_t, 0, 449), r, kp, rt_cfg)
    cyc, _, _ = find_cycles(s, rt_cfg, tid=1)
    off = build_events([ScoredCycle(c, heuristic_cycle_score(c, 1.0)) for c in cyc], rt_cfg["events"], tid=1)
    ups = [u for u in run_stream(StreamEngine(rt_cfg, "c", make_pose_fn(G2), keep_frames=False), 45.0) if u.kind == "open"]
    assert len(off) == 1 and len(ups) == 1
    assert ups[0].start == pytest.approx(off[0].start, abs=0.11) and ups[0].end == pytest.approx(off[0].end, abs=0.11)


def test_buffer_is_bounded_and_state_survives_trimming(rt_cfg):
    eng = StreamEngine(rt_cfg, "c", make_pose_fn(G2), keep_frames=False, buffer_sec=30.0)
    run_stream(eng, 120.0, flush=False)
    assert len(eng._frames) <= 30 * 10 + 20 and len(eng._rows) == len(eng._kp) and eng.stats["frames"] == 1200
    assert len(eng._emitted) == 1                               # событие не потеряно и не продублировано после обрезки буфера


def test_frame_ring_thumbnail_and_clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    ring = FrameRing(seconds=10.0, fps=4.0, width=320)
    for i in range(80):
        img = np.full((360, 640, 3), (i * 3) % 255, np.uint8)
        ring.add(img, i / 8.0)
    assert 0 < len(ring.items) <= 41 and ring.scale == pytest.approx(0.5)
    jpg = ring.thumbnail(5.0, (100, 100, 300, 300), "90%")
    assert jpg and cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (180, 320)
    assert ring.thumbnail(5.0, None) is not None and FrameRing().thumbnail(0, None) is None
    assert ring.write_clip(4.0, 9.0, tmp_path / "c.mp4") and (tmp_path / "c.mp4").stat().st_size > 0
    assert not ring.write_clip(100.0, 101.0, tmp_path / "x.mp4")


def test_alert_update_serializes():
    import json

    u = AlertUpdate("open", "k", "c", 1, 1.0, 2.0, 1.5, 0.7, "two_cycles", 2, "e", (1, 2, 3, 4), 3.0)
    assert json.loads(json.dumps(u.to_dict()))["box"] == [1, 2, 3, 4]
