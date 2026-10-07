"""Сквозной тест: ключевые точки -> ряд признаков -> циклы -> таблица циклов с вектором признаков (то, что падало на реальных данных)."""
import numpy as np
import pandas as pd
from test_features import synth_track

from sd.cycles import find_cycles
from sd.dataset import FEATURE_GROUPS, KINEMATIC, QUALITY, RHYTHM, build_cycle_table
from sd.features import build_series
from sd.tracks import Tracks


def make_tracks(n_repeat=2):
    """Один человек делает жест дважды (две затяжки с паузой), 10 к/с."""
    g1, r1, k1 = synth_track(n=70, gesture=(10, 20, 35, 45))
    g2, r2, k2 = synth_track(n=70, gesture=(10, 20, 35, 45))
    grid = pd.concat([g1, g2.assign(frame=g2.frame + 70, t=g2.t + 7.0)], ignore_index=True)
    rows = pd.concat([r1, r2.assign(frame=r2.frame + 70)], ignore_index=True)
    kp = np.concatenate([k1, k2])
    rows["tid"] = 1
    rows["t"] = grid.set_index("frame").loc[rows.frame, "t"].values
    rows["h"] = rows.y2 - rows.y1
    df = rows[["frame", "t", "tid", "x1", "y1", "x2", "y2", "score", "h"]]
    meta = dict(video_info=dict(width=1920, height=1080, fps=10.0), stride=1, start=0.0, end=None)
    return Tracks(df, kp, grid[["frame", "t"]], meta), grid, rows, kp


def test_build_cycle_table_has_all_feature_groups(cfg):
    tr, grid, rows, kp = make_tracks()
    s = build_series(grid, rows, kp, cfg)
    s.insert(0, "tid", 1)
    cycles, _, _ = find_cycles(s, cfg, tid=1)
    assert len(cycles) == 2
    cyc = pd.DataFrame([c.to_dict() for c in cycles])
    tab = build_cycle_table("data/курение/synthetic.mp4", tr, s, cyc, brightness=100.0)
    assert len(tab) == 2 and (tab.video == "курение__synthetic").all() and (tab.weak_label == "smoking").all()
    for col in KINEMATIC + RHYTHM + QUALITY:
        assert col in tab.columns, col
    r0 = tab.iloc[0]
    assert r0.hand == 1 and 0.5 <= r0.hold <= 5.0 and r0.d_min < 0.3
    assert r0.n_cycles_pm20 == 1 and r0.interval_next > 0          # соседний цикл виден в окне ±20 с
    assert r0.H > 100 and 0 < r0.edge_dist and r0.missing_frac < 0.3
    assert tab.brightness.iloc[0] == 100.0
    assert set(FEATURE_GROUPS) >= {"A_kinematics", "B_rhythm", "C_quality"}
