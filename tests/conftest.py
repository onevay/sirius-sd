import numpy as np
import pytest

from sd.config import load_config


@pytest.fixture(scope="session")
def cfg():
    # Алгоритмические тесты используют синтетические сигналы под опорные пороги руководства (0.35 / 0.55);
    # калибровка default.yaml (ANALYSIS.md, 7.3) на их результат влиять не должна.
    return load_config(overrides=["cycles.th_in=0.35", "cycles.th_out=0.55"])


def make_d(duration=30.0, fps=10.0, rest=2.0, gestures=(), noise=0.0, seed=0):
    """Синтетический d(t): отдых на уровне `rest`, жесты = (t_start, approach_s, hold_s, retract_s, d_mouth)."""
    t = np.arange(0, duration, 1 / fps)
    d = np.full_like(t, rest)
    for t0, ap, hold, ret, dm in gestures:
        for i, ti in enumerate(t):
            if t0 <= ti < t0 + ap:
                d[i] = rest + (dm - rest) * (ti - t0) / ap
            elif t0 + ap <= ti < t0 + ap + hold:
                d[i] = dm
            elif t0 + ap + hold <= ti < t0 + ap + hold + ret:
                d[i] = dm + (rest - dm) * (ti - (t0 + ap + hold)) / ret
    if noise:
        d = d + np.random.default_rng(seed).normal(0, noise, len(d))
    return t, d
