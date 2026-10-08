import pandas as pd

from sd import preds_check as PC
from sd.events import Event, events_to_frame


def row(**kw):
    r = dict(camera_id="c", clip_id="k", event_id="e1", start_sec=1.0, end_sec=9.0, confidence=0.8, label="smoking_like", person_track_id=3, peak_sec=5.0, x1=10, y1=10, x2=50, y2=90)
    r.update(kw)
    return r


def test_valid_file_has_no_issues_and_problems_are_named():
    ok = pd.DataFrame([row(), row(event_id="e2")])
    assert PC.validate(ok, {"k": (640, 360)}) == []
    bad = pd.DataFrame([row(), row(start_sec=9.0, end_sec=2.0, confidence=1.5, x2=5, peak_sec=99), row(x2=700, event_id="e3")])
    msg = " | ".join(PC.validate(bad, {"k": (640, 360)}))
    for part in ("event_id не уникален", "end_sec ≤ start_sec", "peak_sec вне интервала", "confidence вне", "x1 < x2", "вне кадра"):
        assert part in msg, part
    assert PC.validate(pd.DataFrame({"a": [1]})) and PC.validate(pd.DataFrame(columns=PC.COLUMNS)) == []
    assert any("пустыми" in w for w in PC.validate(pd.DataFrame([row(x1=float("nan"))])))


def test_events_to_frame_clips_box_into_frame_and_rounds():
    ev = Event(tid=1, start=1.0, end=9.0, peak=5.0, confidence=0.9, rule="two_cycles", n_cycles=2)
    df = events_to_frame([ev], "c", "k", {1: lambda t: (-5.4, 10.2, 700.7, 400.0)}, frame_wh=(640, 360))
    assert df[["x1", "y1", "x2", "y2"]].iloc[0].tolist() == [0, 10, 640, 360]
    assert PC.validate(df, {"k": (640, 360)}) == []
    nb = events_to_frame([ev], "c", "k", {1: lambda t: None}, frame_wh=(640, 360))
    assert nb[["x1", "y1"]].isna().all(axis=None) and any("пустыми" in w for w in PC.validate(nb))
