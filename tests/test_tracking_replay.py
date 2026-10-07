import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ultralytics", reason="трекеры берутся из Ultralytics")

from sd.bench import replay_tracking, track_metrics  # noqa: E402


def synth_raw(n=80, fps=10.0, gap=None):
    """Двое: A идёт слева направо, B стоит. gap — кадры, где A не найден детектором (кратковременный пропуск)."""
    rows, ft = [], []
    for i in range(n):
        t = i / fps
        ft.append((i, t))
        if not (gap and gap[0] <= i < gap[1]):
            x = 100 + 8 * i
            rows.append((i, t, x, 200, x + 80, 420, 0.9))
        rows.append((i, t, 900, 220, 980, 440, 0.85))
    return pd.DataFrame(rows, columns=["frame", "t", "x1", "y1", "x2", "y2", "conf"]), pd.DataFrame(ft, columns=["frame", "t"])


@pytest.mark.parametrize("tracker", ["botsort", "bytetrack"])
def test_two_people_two_tracks(cfg, tracker):
    raw, ft = synth_raw()
    df = replay_tracking(raw, ft, cfg, tracker=tracker, buffer_sec=3.0)
    m = track_metrics(df, raw, ft, cfg)
    assert m["tracks"] == 2 and m["n_people"] == 2 and m["tracks_per_person"] == 1.0
    assert m["dup_frames"] == 0


def test_short_gap_does_not_split_track_with_buffer(cfg):
    raw, ft = synth_raw(gap=(30, 36))               # 0.6 с без детекций
    df = replay_tracking(raw, ft, cfg, tracker="bytetrack", buffer_sec=3.0)
    m = track_metrics(df, raw, ft, cfg)
    assert m["tracks"] == 2                         # буфер 3 с удержал трек (или склейка после) — человека не «потеряли»


def test_empty_detections_ok(cfg):
    raw = pd.DataFrame(columns=["frame", "t", "x1", "y1", "x2", "y2", "conf"])
    ft = pd.DataFrame(dict(frame=range(10), t=np.arange(10) / 10))
    df = replay_tracking(raw, ft, cfg, tracker="botsort")
    assert df.empty and track_metrics(df, raw, ft, cfg)["tracks"] == 0
