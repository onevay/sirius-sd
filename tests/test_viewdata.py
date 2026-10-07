import numpy as np
import pandas as pd

from sd import gt as G
from sd import viewdata as VD
from sd.tracks import Tracks


def _tracks(tmp_path, h=200):
    rows = [(i, i / 10, 1, 10 + i, 20, 110 + i, 20 + h, 0.9, h) for i in range(50)] + [(i, i / 10, 2, 300, 20, 400, 80, 0.9, 60) for i in range(50)]
    df = pd.DataFrame(rows, columns=["frame", "t", "tid", "x1", "y1", "x2", "y2", "score", "h"])
    tr = Tracks(df, np.zeros((len(df), 17, 3), np.float32), pd.DataFrame({"frame": range(50), "t": [i / 10 for i in range(50)]}), dict(video_info=dict(width=640, height=360)))
    d = tmp_path / "runs" / "clip" / "run1" / "pose"
    tr.save(d)
    return tmp_path / "runs"


def test_find_runs_and_payload(tmp_path):
    root = _tracks(tmp_path)
    runs = VD.find_pose_runs("clip", root)
    assert len(runs) == 1 and VD.find_pose_runs("нет", root) == []
    p = VD.tracks_payload(runs[0])
    assert set(p["tracks"]) == {"1", "2"} and p["src_w"] == 640 and p["tracks"]["1"][0] == [0.0, 10, 20, 110, 220]
    assert set(VD.tracks_payload(runs[0], min_height_px=100)["tracks"]) == {"1"}          # мелкий человек отфильтрован
    assert VD.box_at(p["tracks"], 1, 2.0) == (30.0, 20.0, 130.0, 220.0) and VD.box_at(p["tracks"], 9, 1.0) is None


def test_payloads_are_json_safe():
    import json

    gt = pd.DataFrame(dict(id=["a", "b"], clip_id=["c", "c"], person_gt_id=["1", ""], start_sec=[1.0, 5.0], end_sec=[3.0, 9.0], label=["POSITIVE", "NEGATIVE"], peak_sec=[2.0, np.nan],
                           x1=[1, np.nan], y1=[1, np.nan], x2=[5, np.nan], y2=[5, np.nan], note=["", None]))
    out = VD.gt_payload(gt, "c")
    assert out[0]["box"] == [1.0, 1.0, 5.0, 5.0] and out[1]["box"] is None and out[1]["peak"] is None
    err = pd.DataFrame(dict(kind=["FP", "FN"], clip_id=["c", "c"], start=[2.0, np.nan], end=[4.0, np.nan], confidence=[0.9, np.nan], cause=["x", "y"], cause_ru=["a", "b"]))
    ev = pd.DataFrame(dict(clip_id=["c", "c"], person_track_id=[1, 2], start_sec=[1.0, 2.0], end_sec=[3.0, 4.0], confidence=[0.9, 0.2]))
    assert len(VD.errors_payload(err, "c")) == 1 and len(VD.events_payload(ev, "c", 0.5)) == 1
    json.dumps([out, VD.errors_payload(err, "c"), VD.events_payload(ev, "c")], allow_nan=False)
    assert VD.errors_payload(pd.DataFrame(), "c") == [] and VD.events_payload(pd.DataFrame(), "c") == []
