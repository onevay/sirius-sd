"""Признаки «дыма» на кропах рта: направление изменений на синтетической «дымке» и устойчивость к отсутствию кропов/эмбеддингов."""
import numpy as np
import pandas as pd

from sd import smoke_cues as SC


def _textured(seed=0, n=224):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (n, n, 3)).astype(np.uint8)
    img[:, :, 2] = 255 - img[:, :, 0]                 # цветная текстура с высоким контрастом
    return img


def _fog(img, k=0.6):
    return (img * (1 - k) + 235 * k).astype(np.uint8)  # молочная дымка: меньше контраст и насыщенность, больше светлых малонасыщенных пикселей


def test_haze_deltas_positive_when_late_frames_are_foggier():
    base = _textured()
    frames = [base, base, base, _fog(base), _fog(base, 0.7), _fog(base, 0.8)]
    d = SC.haze_deltas(frames)
    assert set(d) == set(SC.HAZE_COLS)
    assert d["smoke_d_sharp"] > 0.3 and d["smoke_d_low"] > 0.1 and d["smoke_d_sat"] > 0.3 and d["smoke_d_white"] > 0.1
    same = SC.haze_deltas([base] * 6)
    assert all(abs(v) < 1e-9 for v in same.values())               # без изменений — нулевые разности


def test_haze_deltas_need_six_frames():
    d = SC.haze_deltas([_textured()] * 5)
    assert all(np.isnan(v) for v in d.values())


def test_clip_cues_use_late_minus_early_and_handle_short_input():
    text_smoke, text_none = np.array([[1.0, 0.0]], np.float32), np.array([[0.0, 1.0]], np.float32)
    # сходство с «дымом» растёт к концу: score кадра = x − y
    emb = np.array([[0.1, 0.9], [0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], np.float32)
    c = SC.clip_cues(emb, text_smoke, text_none)
    assert abs(c["smoke_clip_max"] - 0.8) < 1e-6 and abs(c["smoke_clip_delta"] - (0.8 - (-0.8))) < 1e-6
    assert all(np.isnan(v) for v in SC.clip_cues(emb[:4], text_smoke, text_none).values())


def test_evaluate_reports_scene_checks_and_skips_missing_columns():
    rng = np.random.default_rng(0)
    n = 80
    video = np.array([f"{'курение' if i % 2 else 'лжекурение'}__v{i // 4}" for i in range(n)])
    y = (rng.random(n) < 0.4).astype(int)
    tab = pd.DataFrame(dict(video=video, y=y, smoke_d_sat=y + rng.normal(0, 0.8, n)))       # признак с реальным сигналом
    r = SC.evaluate(tab, n_boot=60)
    assert list(r["признак"]) == ["smoke_d_sat"] and r.AUC.iloc[0] > 0.65 and r.lo.iloc[0] < r.AUC.iloc[0] < r.hi.iloc[0]
    assert 0.0 <= r["папка_среди_негативов"].iloc[0] <= 1.0


def test_cycle_cues_skips_cycles_without_crops(tmp_path, monkeypatch):
    from sd import photo_video as PV

    monkeypatch.setattr(PV, "CROPS", tmp_path)
    cyc = pd.DataFrame([dict(video="v", tid=1, start=1.0)])
    assert SC.cycle_cues(cyc, with_clip=False).empty                # кропов нет — строки нет, ошибки нет
    import cv2

    d = PV.cycle_dir("v", 1, 1.0)
    d.mkdir(parents=True)
    base = _textured()
    for j, f in enumerate([base, base, base, _fog(base), _fog(base), _fog(base)]):
        cv2.imwrite(str(d / f"{j}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])
    out = SC.cycle_cues(cyc, with_clip=False)
    assert len(out) == 1 and out.smoke_d_sharp.iloc[0] > 0 and out.start.iloc[0] == 1.0
