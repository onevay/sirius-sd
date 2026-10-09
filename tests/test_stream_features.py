"""Признаки предмета в потоке: те же, что offline (регрессия: в потоке модель mvp не давала тревог, потому что `obj_any_*` не считались, кадры брались не те и всё пересчитывалось заново)."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from sd import solver as SV
from sd.pose_track import _hand_near_face
from sd.realtime.scoring import ClassifierScorer
from sd.tracks import Tracks
from sd.video_io import window_frame_index


def _kp(nose, lw, rw, ls=(100, 100), rs=(200, 100)):
    k = np.zeros((17, 3), np.float32)
    k[0] = (*nose, 1)
    k[9] = (*lw, 1)
    k[10] = (*rw, 1)
    k[5] = (*ls, 1)
    k[6] = (*rs, 1)
    return k


def test_hand_near_face_gate():
    box = np.array([0, 0, 300, 800])
    near = _kp((150, 60), (160, 90), (400, 700))
    far = _kp((150, 60), (20, 700), (400, 700))
    assert _hand_near_face(near, box, 2.0) and not _hand_near_face(far, box, 2.0)
    assert _hand_near_face(far, box, 20.0)                 # широкий порог пропускает всех
    unseen = _kp((150, 60), (0, 0), (0, 0))
    unseen[9, 2] = unseen[10, 2] = 0
    assert _hand_near_face(unseen, box, 2.0)               # кисти не видны — не рискуем, уточняем


def _video(path, n=40, fps=10.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), (i * 5) % 255, np.uint8))
    w.release()


def test_window_frame_index_matches_time(tmp_path):
    p = tmp_path / "v.mp4"
    _video(p)
    idx, ts = window_frame_index(p, 1.0, 2.0)
    assert idx[0] >= 9 and ts[0] <= 1.1 and ts[-1] <= 2.0 + 1e-6 and (np.diff(idx) == 1).all() and (np.diff(ts) > 0).all()


def test_stream_frames_are_mapped_to_file_frames(tmp_path):
    p = tmp_path / "v.mp4"
    _video(p, n=60, fps=30.0)
    ft = pd.DataFrame({"frame": np.arange(10), "t": np.arange(10) * 0.1})            # поток 10 к/с: кадры 0,1,2… = кадры файла 0,3,6…
    rows = pd.DataFrame({"frame": [2, 3], "t": [0.2, 0.3], "tid": [1, 1], "x1": 0, "y1": 0, "x2": 1, "y2": 1, "score": 1.0, "h": 1.0})
    tr = Tracks(rows, np.zeros((2, 17, 3), np.float32), ft, {})
    ser = pd.DataFrame({"frame": [2, 3], "t": [0.2, 0.3], "tid": [1, 1]})
    sc = object.__new__(ClassifierScorer)
    sc.source = p
    tr2, ser2 = sc._to_source_frames(tr, ser, pd.DataFrame({"start": [0.1], "end": [0.5]}))
    assert tr2.df.frame.tolist() == [6, 9] and ser2.frame.tolist() == [6, 9] and tr2.meta["video"] == str(p)
    assert tr.df.frame.tolist() == [2, 3]                   # исходные треки не изменены
    far = sc._to_source_frames(tr, ser, pd.DataFrame({"start": [40.0], "end": [41.0]}))[0]
    assert far.frame_t.frame.min() >= 10 ** 9              # вне окна файла — номеров, совпадающих с кадрами, нет


def test_bundle_with_obj_any_requires_detectors(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(dict(kind="cycle_classifier", format=2, features=["hold", "obj_any_max_conf", "obj_any_hit_frames"], members={}, cv={})), encoding="utf-8")
    info = SV.describe_bundle(d)
    assert info["requires"]["objects_any"] and info["requires"]["objects"] == []
    (d / "manifest.json").write_text(json.dumps(dict(kind="cycle_classifier", format=2, features=["hold"], members={}, cv={})), encoding="utf-8")
    assert not SV.describe_bundle(d)["requires"]["objects_any"]


def test_scorer_refuses_obj_any_bundle_without_detectors(tmp_path):
    from sd.realtime.scoring import FeatureUnavailable

    d = tmp_path / "b"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(dict(kind="cycle_classifier", format=2, features=["hold", "obj_any_max_conf"], members={}, cv={})), encoding="utf-8")
    (d / "members.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FeatureUnavailable):
        ClassifierScorer(d, {"events": {}}, objects=[], source_path=tmp_path / "x.mp4")


def test_prefetch_keeps_order_propagates_errors_and_stops_early():
    from sd.video_io import prefetch

    assert list(prefetch(iter(range(50)), size=3)) == list(range(50))

    def bad():
        yield 1
        raise RuntimeError("источник упал")

    got = []
    with pytest.raises(RuntimeError):
        for x in prefetch(bad()):
            got.append(x)
    assert got == [1]
    consumed = []

    def src():
        for i in range(10_000):
            consumed.append(i)
            yield i

    for x in prefetch(src(), size=2):
        if x == 3:
            break
    import time

    time.sleep(0.5)
    n = len(consumed)
    time.sleep(0.3)
    assert len(consumed) == n and n < 100                 # поток остановился, а не дочитал источник до конца


def test_profile_overrides_are_validated_and_applied():
    from sd import profiles as PR

    p = PR.Profile(name="x", options={"objects": ["a"]})
    q = PR.with_overrides(p, ["video.process_fps=5", "pose.refine.gate_s=3.0", "options.objects=[b, c]", "events.cycle_th=0.35"])
    c = q.cfg()
    assert c["video"]["process_fps"] == 5 and c["pose"]["refine"]["gate_s"] == 3.0 and q.opts()["objects"] == ["b", "c"] and c["events"]["cycle_th"] == 0.35
    assert p.options["objects"] == ["a"] and not p.config                      # исходный профиль не изменён
    with pytest.raises(KeyError):
        PR.with_overrides(p, ["video.process_fpss=5"])
    with pytest.raises((ValueError, KeyError)):
        PR.with_overrides(p, ["options.nonsense=1"])
    with pytest.raises(ValueError):
        PR.with_overrides(p, ["без_равно"])
    assert PR.with_overrides(p, None) is p
