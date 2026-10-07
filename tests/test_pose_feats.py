import numpy as np
import pandas as pd

from sd import pose_feats as PF


def _scene(hand=0, n=40, dt=0.1):
    """Человек анфас: плечи на 300/400 px (s=100), нос (350,200), рот (350,230); активная кисть подходит ко рту на 1.0–3.0 с."""
    t = np.arange(n) * dt
    kp = np.zeros((n, 17, 3), np.float32)
    kp[..., 2] = 0.9
    kp[:, 5, :2] = (400, 300)    # LSHO (на изображении правее)
    kp[:, 6, :2] = (300, 300)    # RSHO
    kp[:, 0, :2] = (350, 200)    # NOSE
    kp[:, 7, :2] = (420, 380)    # LELB
    kp[:, 8, :2] = (280, 380)    # RELB
    wrist = np.tile([[430.0, 420.0]], (n, 1))
    at = (t >= 1.0) & (t <= 3.0)
    wrist[at] = (360, 230)       # к рту (350, 230) со смещением 10 px
    kp[:, 9, :2] = wrist         # LWRI
    kp[:, 10, :2] = (270, 420)   # RWRI — вторая рука опущена
    rows = pd.DataFrame(dict(frame=np.arange(n), t=t))
    ser = pd.DataFrame(dict(frame=np.arange(n), t=t, s=100.0, mouth_x=350.0, mouth_y=230.0, elbowL=np.where(at, 60.0, 170.0), elbowR=170.0,
                            d=np.where(at, 0.1, 1.5)))
    c = pd.Series(dict(start=0.6, mouth_in=1.0, mouth_out=3.0, end=3.4, peak_t=2.0, hand=hand, hold=2.0, d_min=0.1, tid=1))
    return rows, kp, ser, c


def test_geometry_of_a_cycle_is_in_shoulder_widths_and_mirrored():
    rows, kp, ser, c = _scene()
    tc = pd.DataFrame(dict(peak_t=[2.0, 20.0]))
    v = PF.pose_vector(rows, kp, ser, c, tc, thr=0.3)
    assert abs(v["p_wm_dist"] - 0.1) < 1e-6
    assert abs(v["p_es_dy"] - 0.8) < 1e-6 and abs(v["p_es_dx"] - (-0.2)) < 1e-6          # локоть ниже плеча, от середины тела: dx зеркалится
    assert abs(v["p_upperarm_deg"] - 104.04) < 0.1 and abs(v["p_forearm_deg"] - 68.2) < 0.1
    assert v["p_elbow_peak"] == 60.0 and v["p_elbow_min"] == 60.0
    assert v["p_other_wrist_dy"] > 1.0                      # вторая рука ниже линии плеч
    assert v["p_both_hands_frac"] == 0.0
    assert v["p_wrist_above_mouth"] == 0.0 and v["p_head_yaw"] == 0.0
    assert v["cycle_index"] == 0 and v["cycle_count_track"] == 2 and abs(v["hold_frac"] - 2.0 / 2.8) < 1e-9
    assert abs(v["t_peak_in_track"] - 2.0) < 1e-9


def test_mirror_gives_same_values_for_the_other_hand():
    rows, kp, ser, c = _scene()
    tc = pd.DataFrame(dict(peak_t=[2.0]))
    base = PF.pose_vector(rows, kp, ser, c, tc, thr=0.3)
    # зеркалим картинку по x (x -> 700 - x) и меняем лево/право: ответ для правой руки должен совпасть
    kp2 = kp.copy()
    kp2[..., 0] = 700 - kp2[..., 0]
    swap = [(5, 6), (7, 8), (9, 10)]
    for a, b in swap:
        kp2[:, [a, b]] = kp2[:, [b, a]]
    ser2 = ser.assign(mouth_x=700 - ser.mouth_x, elbowL=ser.elbowR, elbowR=ser.elbowL)
    c2 = c.copy()
    c2["hand"] = 1
    mir = PF.pose_vector(rows, kp2, ser2, c2, tc, thr=0.3)
    for k in ("p_wm_dx", "p_wm_dy", "p_es_dx", "p_es_dy", "p_forearm_deg", "p_upperarm_deg", "p_elbow_peak"):
        assert abs(base[k] - mir[k]) < 1e-6, k


def test_unknown_hand_or_empty_window_gives_nans():
    rows, kp, ser, c = _scene()
    c["hand"] = -1
    v = PF.pose_vector(rows, kp, ser, c, pd.DataFrame(dict(peak_t=[2.0])), thr=0.3)
    assert all(np.isnan(x) for x in v.values())
    assert set(v) == set(PF.POSE_COLS)
