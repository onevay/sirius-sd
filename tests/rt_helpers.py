"""Синтетический человек для тестов потока: жесты «рука ко рту» по расписанию, без моделей и видео."""
from types import SimpleNamespace

import numpy as np

REST_WRIST = (120.0, 140.0)
MOUTH_WRIST = (104.0, 55.0)


def person_kp(wrist_xy, conf=0.9):
    kp = np.zeros((17, 3), np.float32)
    kp[0] = (100, 50, conf)
    kp[1] = (96, 46, conf); kp[2] = (104, 46, conf)
    kp[3] = (90, 50, conf); kp[4] = (110, 50, conf)
    kp[5] = (80, 70, conf); kp[6] = (120, 70, conf)
    kp[7] = (75, 105, conf); kp[8] = (125, 105, conf)
    kp[9] = (80, 140, conf)
    kp[10] = (*wrist_xy, conf)
    kp[11] = (85, 130, conf); kp[12] = (115, 130, conf)
    return kp


def wrist_at(t, gestures):
    """gestures = [(t0, подъём, пауза, отведение), …] в секундах."""
    for t0, ap, hold, ret in gestures:
        if t0 <= t < t0 + ap:
            k = (t - t0) / ap
        elif t0 + ap <= t < t0 + ap + hold:
            k = 1.0
        elif t0 + ap + hold <= t < t0 + ap + hold + ret:
            k = 1.0 - (t - (t0 + ap + hold)) / ret
        else:
            continue
        return (REST_WRIST[0] + (MOUTH_WRIST[0] - REST_WRIST[0]) * k, REST_WRIST[1] + (MOUTH_WRIST[1] - REST_WRIST[1]) * k)
    return REST_WRIST


def make_pose_fn(gestures, tid=1, box=(60.0, 30.0, 140.0, 200.0), calls=None):
    def pose_fn(img, t):
        if calls is not None:
            calls.append(t)
        return [SimpleNamespace(tid=tid, box=np.array(box, np.float32), score=0.9, kp=person_kp(wrist_at(t, gestures)))]

    return pose_fn


def run_stream(engine, duration, fps=10.0, flush=True):
    out = []
    for i in range(int(duration * fps)):
        out += engine.step(None, i / fps)
    if flush:
        out += engine.flush(duration)
    return out
