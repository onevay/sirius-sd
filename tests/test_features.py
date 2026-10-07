import numpy as np
import pandas as pd
import pytest

from sd.cycles import find_cycles
from sd.features import build_series, interp_small_gaps, smooth_segments


def synth_track(n=100, fps=10.0, gesture=(20, 30, 50, 60), conf=0.9, drop=None):
    """Человек: плечи 40 px, нос над серединой плеч, правое запястье поднимается ко рту."""
    kp = np.zeros((n, 17, 3), np.float32)
    kp[:, 0] = (100, 50, conf)             # нос
    kp[:, 1] = (96, 46, conf); kp[:, 2] = (104, 46, conf)
    kp[:, 3] = (90, 50, conf); kp[:, 4] = (110, 50, conf)
    kp[:, 5] = (80, 70, conf); kp[:, 6] = (120, 70, conf)
    kp[:, 7] = (75, 105, conf); kp[:, 8] = (125, 105, conf)
    kp[:, 9] = (80, 140, conf); kp[:, 10] = (120, 140, conf)
    kp[:, 11] = (85, 130, conf); kp[:, 12] = (115, 130, conf)
    a, b, c, dd = gesture
    for i in range(n):
        if a <= i < b:
            k = (i - a) / (b - a)
            kp[i, 10, :2] = (120 + (104 - 120) * k, 140 + (55 - 140) * k)
        elif b <= i < c:
            kp[i, 10, :2] = (104, 55)
        elif c <= i < dd:
            k = (i - c) / (dd - c)
            kp[i, 10, :2] = (104 + (120 - 104) * k, 55 + (140 - 55) * k)
    frames = np.arange(n)
    rows = pd.DataFrame(dict(frame=frames, x1=60.0, y1=30.0, x2=140.0, y2=200.0, score=0.9))
    if drop is not None:
        keep = ~np.isin(frames, drop)
        rows, kp = rows[keep].reset_index(drop=True), kp[keep]
    grid = pd.DataFrame(dict(frame=frames, t=frames / fps))
    return grid, rows, kp


def test_d_series_and_hand_detection(cfg):
    grid, rows, kp = synth_track()
    s = build_series(grid, rows, kp, cfg)
    rest = s.d[(s.t > 0.2) & (s.t < 1.5)].median()
    hold = s.d[(s.t > 3.2) & (s.t < 4.8)].median()
    assert rest == pytest.approx(2.1, abs=0.15)           # рука внизу: ~2 ширины плеч
    assert hold < 0.2                                      # рука у рта
    assert (s.hand[(s.t > 3.2) & (s.t < 4.8)] == 1).all()  # правая
    assert s.s.median() == pytest.approx(40, abs=1)


def test_cycle_found_end_to_end_from_keypoints(cfg):
    grid, rows, kp = synth_track()
    s = build_series(grid, rows, kp, cfg)
    cycles, rejected, _ = find_cycles(s, cfg, tid=1)
    assert len(cycles) == 1
    assert 1.5 <= cycles[0].start <= 3.1 and 4.8 <= cycles[0].end <= 6.3
    assert cycles[0].hand == 1


def test_hand_extend_moves_hand_point_along_forearm(cfg):
    from sd.config import load_config

    grid, rows, kp = synth_track()
    s0 = build_series(grid, rows, kp, cfg)                                   # hand_extend = 0 -> кисть = запястье
    assert np.allclose(s0.handR_x.dropna(), s0.wristR_x.dropna()) and np.allclose(s0.handR_y.dropna(), s0.wristR_y.dropna())
    cfg2 = load_config(overrides=["features.hand_extend=0.5"])
    s1 = build_series(grid, rows, kp, cfg2)
    i = int(np.argmin(np.abs(s1.t.values - 4.0)))                           # пауза у рта: запястье (104,55), локоть (125,105)
    assert s1.handR_x[i] == pytest.approx(104 + 0.5 * (104 - 125), abs=1e-3)
    assert s1.handR_y[i] == pytest.approx(55 + 0.5 * (55 - 105), abs=1e-3)
    assert s1.d[i] != s0.d[i]                                                # расстояние считается по продлённой точке


def test_missing_detections_become_nan_and_small_gaps_are_interpolated(cfg):
    grid, rows, kp = synth_track(drop=[10, 11])           # 0.2 с — меньше interp_gap_sec = 0.3
    s = build_series(grid, rows, kp, cfg)
    assert (~s.have_det).sum() == 2
    assert np.isfinite(s.d[(s.frame >= 10) & (s.frame <= 11)]).all()   # заполнено интерполяцией
    grid, rows, kp = synth_track(drop=list(range(10, 20)))  # 1.0 с — остаётся «нет данных»
    s = build_series(grid, rows, kp, cfg)
    assert np.isnan(s.d[(s.frame >= 12) & (s.frame <= 17)]).all()


def test_low_confidence_wrist_is_not_used(cfg):
    grid, rows, kp = synth_track()
    kp[:, 10, 2] = 0.1                                      # запястье не видно (< kp_conf_min)
    s = build_series(grid, rows, kp, cfg)
    assert np.isnan(s.dR).all()
    assert np.isfinite(s.dL).any()                          # левая рука видна -> d считается по ней


def test_interp_small_gaps_respects_limit_and_edges():
    t = np.arange(10) * 0.1
    y = np.array([1, np.nan, 3, np.nan, np.nan, np.nan, np.nan, 8, np.nan, np.nan])
    out = interp_small_gaps(t, y, max_gap=0.3)
    assert out[1] == pytest.approx(2.0)                      # короткий пропуск заполнен
    assert np.isnan(out[3:7]).all()                          # длинный (0.5 с) — нет
    assert np.isnan(out[8:]).all()                           # хвост не экстраполируем


def test_smooth_segments_does_not_bridge_nan():
    y = np.concatenate([np.ones(8), [np.nan] * 3, np.full(8, 5.0)])
    out = smooth_segments(y, 3, 5, 2)
    assert np.isnan(out[8:11]).all()
    assert out[:8].mean() == pytest.approx(1.0, abs=1e-6) and out[11:].mean() == pytest.approx(5.0, abs=1e-6)
