import numpy as np
import pandas as pd
import pytest
from conftest import make_d

from sd.cycles import CycleDetector, find_cycles


def run(cfg, t, d, hand=1):
    s = pd.DataFrame(dict(t=t, d=d, hand=np.full(len(t), hand)))
    return find_cycles(s, cfg, tid=7)


def test_single_clean_cycle_boundaries(cfg):
    t, d = make_d(gestures=[(5.0, 1.0, 2.0, 1.0, 0.2)])
    cycles, rejected, states = run(cfg, t, d)
    assert len(cycles) == 1 and not rejected
    c = cycles[0]
    assert c.tid == 7 and c.hand == 1
    assert 4.8 <= c.start <= 5.2           # начало подъёма найдено назад по скорости
    assert 5.8 <= c.mouth_in <= 6.2        # d < th_in
    assert 7.9 <= c.mouth_out <= 8.4       # d > th_out
    assert c.mouth_out < c.end <= 8.9      # отведение завершено
    assert 1.8 <= c.hold <= 2.4
    assert c.d_min == pytest.approx(0.2, abs=0.05)
    assert c.peak_t == pytest.approx(0.5 * (c.mouth_in + c.mouth_out))


def test_short_touch_is_not_a_cycle(cfg):
    t, d = make_d(gestures=[(5.0, 0.5, 0.3, 0.5, 0.2)])
    cycles, rejected, _ = run(cfg, t, d)
    assert cycles == []
    assert [r.reason for r in rejected] == ["short_touch"]


def test_long_hold_phone_at_ear_is_not_a_cycle_in_strict_mode(cfg):
    from sd.config import load_config

    strict = load_config(overrides=["cycles.emit_long_hold=false"])        # режим регламента буквально: пауза 0.5–5 с
    t, d = make_d(duration=30, gestures=[(5.0, 1.0, 9.0, 1.0, 0.2)])
    cycles, rejected, _ = run(strict, t, d)
    assert cycles == []
    assert [r.reason for r in rejected] == ["long_hold"]


def test_long_hold_is_a_candidate_up_to_cap_and_rejected_beyond(cfg):
    t, d = make_d(duration=40, gestures=[(5.0, 1.0, 7.0, 1.0, 0.2)])        # курильщик держит сигарету у лица 7 с (как sm_4)
    cycles, rejected, _ = run(cfg, t, d)
    assert len(cycles) == 1 and 6.0 <= cycles[0].hold <= 8.0 and not rejected
    t, d = make_d(duration=40, gestures=[(5.0, 1.0, 20.0, 1.0, 0.2)])       # 20 с > hold_cap_sec = 12: телефон у уха
    cycles, rejected, _ = run(cfg, t, d)
    assert cycles == [] and [r.reason for r in rejected] == ["long_hold"]


def test_two_puffs_two_cycles(cfg):
    t, d = make_d(gestures=[(4.0, 1.0, 1.5, 1.0, 0.2), (12.0, 1.0, 1.5, 1.0, 0.2)])
    cycles, _, _ = run(cfg, t, d)
    assert len(cycles) == 2
    assert cycles[0].end < cycles[1].start


def test_hysteresis_noise_on_threshold_does_not_split_cycle(cfg):
    # внутри паузы d дрожит между th_in и th_out (0.40–0.52) — один цикл, а не несколько
    t, d = make_d(gestures=[(5.0, 1.0, 3.0, 1.0, 0.2)])
    hold = (t >= 6.5) & (t < 8.0)
    d[hold] = 0.46 + 0.06 * np.sin(np.arange(hold.sum()) * 1.7)
    cycles, rejected, _ = run(cfg, t, d)
    assert len(cycles) == 1


def test_quick_second_puff_without_full_retract_gives_two_cycles(cfg):
    t = np.arange(0, 20, 0.1)
    d = np.full_like(t, 2.0)
    # подъём -> пауза 1.2 с -> отведение лишь до 0.62 (между th_out и th_out+0.15) -> снова ко рту
    for i, ti in enumerate(t):
        if 3.0 <= ti < 4.0:
            d[i] = 2.0 - 1.8 * (ti - 3.0)
        elif 4.0 <= ti < 5.2:
            d[i] = 0.2
        elif 5.2 <= ti < 5.7:
            d[i] = 0.62
        elif 5.7 <= ti < 6.2:
            d[i] = 0.2
        elif 6.2 <= ti < 6.7:
            d[i] = 0.2 + (2.0 - 0.2) * (ti - 6.2) / 0.5
    cycles, _, _ = run(cfg, t, d)
    assert len(cycles) == 2


def test_track_starting_with_hand_at_mouth_does_not_crash(cfg):
    t = np.arange(0, 10, 0.1)
    d = np.full_like(t, 0.2)
    d[t > 3] = 2.0
    cycles, rejected, _ = run(cfg, t, d)
    assert isinstance(cycles, list)  # без исключений, d[i-1] на i=0 — ошибка примера из руководства


def test_data_gap_inside_gesture_marks_lost(cfg):
    t, d = make_d(gestures=[(5.0, 1.0, 3.0, 1.0, 0.2)])
    d[(t > 6.5) & (t < 8.0)] = np.nan   # пропуск 1.5 с > max_gap_sec
    cycles, rejected, _ = run(cfg, t, d)
    assert any(r.reason == "lost" for r in rejected)   # прерванный жест помечен как потерянный
    assert len(cycles) <= 1                            # оставшаяся часть паузы после пропуска — не более одного цикла


def test_streaming_equals_batch(cfg):
    t, d = make_d(gestures=[(4.0, 1.0, 1.5, 1.0, 0.2), (12.0, 1.0, 2.0, 1.0, 0.25)], noise=0.01)
    batch, _, _ = run(cfg, t, d)
    det = CycleDetector(cfg["cycles"], tid=7)
    out = []
    for ti, di in zip(t, d):
        c = det.update(float(ti), float(di), 1)
        if c:
            out.append(c)
    f = det.flush(float(t[-1]))
    if f:
        out.append(f)
    assert [(round(c.start, 2), round(c.end, 2)) for c in out] == [(round(c.start, 2), round(c.end, 2)) for c in batch]


def test_track_end_during_retract_still_closes_cycle(cfg):
    # медленное отведение 5 с: порог th_out=0.55 пересекается ~на 8.97 с, а th_out+0.15=0.70 — ~на 9.4 с; трек обрывается между ними
    t, d = make_d(duration=9.3, gestures=[(5.0, 1.0, 2.0, 5.0, 0.2)])
    cycles, rejected, _ = run(cfg, t, d)
    assert len(cycles) == 1
