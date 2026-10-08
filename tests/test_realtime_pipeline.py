import threading
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rt_helpers import make_pose_fn
from sd import gt as GT
from sd import profiles as PR
from sd.realtime import feedback as FB
from sd.realtime import sources as S
from sd.realtime import worker as W
from sd.realtime.engine import AlertUpdate, StreamEngine
from sd.realtime.store import AlertStore

G2 = [(5.0, 2.0, 1.5, 1.5), (16.0, 2.0, 1.5, 1.5)]


@pytest.mark.parametrize("name,district,index,start", [
    ("gatchina-03-20261009T143000", "gatchina", "03", datetime(2026, 10, 9, 14, 30, 0)),
    ("gatchina_03_2026-10-09_14-30-00", "gatchina", "03", datetime(2026, 10, 9, 14, 30, 0)),
    ("tsarskoe-selo-12-1430", "tsarskoe-selo", "12", datetime(2026, 1, 2, 14, 30, 0)),
    ("Гатчина-7-14-30-15", "Гатчина", "7", datetime(2026, 1, 2, 14, 30, 15)),
    ("pavlovsk-01-202610091430", "pavlovsk", "01", datetime(2026, 10, 9, 14, 30, 0)),
])
def test_stream_dir_names(tmp_path, name, district, index, start):
    d = tmp_path / name
    d.mkdir()
    (d / "a.mp4").write_bytes(b"x")
    s = S.parse_stream_dir(d, base_date=date(2026, 1, 2))
    assert (s.district, s.index, s.start, s.parsed) == (district, index, start, True) and s.camera_id == f"{district}-{index}"


def test_unparsable_folder_falls_back_and_chunks_sorted_naturally(tmp_path):
    d = tmp_path / "просто папка"
    d.mkdir()
    for n in ("chunk10.mp4", "chunk2.mp4", "chunk1.mp4", "readme.txt"):
        (d / n).write_bytes(b"x")
    s = S.parse_stream_dir(d)
    assert not s.parsed and s.district == "просто папка" and [c.name for c in S.list_chunks(d)] == ["chunk1.mp4", "chunk2.mp4", "chunk10.mp4"]
    assert S.parse_start("2026-13-45_99-99") is None


def test_scan_streams_and_abs_time(tmp_path):
    for n in ("b-02-20261009T100000", "a-10-20261009T100000", "a-02-20261009T100000", "empty-01-20261009T100000"):
        (tmp_path / n).mkdir()
    for n in ("b-02-20261009T100000", "a-10-20261009T100000", "a-02-20261009T100000"):
        (tmp_path / n / "c1.mp4").write_bytes(b"x")
    ss = S.scan_streams(tmp_path)
    assert [s.camera_id for s in ss] == ["a-02", "a-10", "b-02"]                   # пустая папка не камера, порядок по району и номеру
    assert S.abs_time(ss[0], 90).isoformat() == "2026-10-09T10:01:30"


def test_pacer_realtime_and_unthrottled():
    now = [0.0]
    sleeps = []
    p = S.Pacer(2.0, clock=lambda: now[0], sleep=lambda d: (sleeps.append(d), now.__setitem__(0, now[0] + d)))
    p.wait(10.0)
    p.wait(12.0)               # 2 с потока при speed=2 → 1 с реального времени
    assert sleeps == [pytest.approx(1.0)]
    q = S.Pacer(0.0, sleep=lambda d: sleeps.append(99))
    q.wait(1.0); q.wait(100.0)
    assert 99 not in sleeps


def alert(key="c:1:5.00", kind="open", start=5.0, end=20.0, conf=0.8, box=(1, 2, 3, 4)):
    return AlertUpdate(kind, key, "c", 1, start, end, (start + end) / 2, conf, "two_cycles", 2, "2 цикла", box, end + 1)


def test_store_lifecycle_review_and_stats(tmp_path):
    st = AlertStore(tmp_path / "m.db")
    st.upsert_camera("c", "gatchina", "01", "$ROOT/streams/c", "2026-10-09T10:00:00")
    meta = dict(district="gatchina", cam_index="01", chunk="$ROOT/x.mp4", chunk_offset=0.0, t_abs="2026-10-09T10:00:05", fingerprint="fp")
    aid, k = st.apply_update(alert(), meta)
    assert k == "open" and aid > 0 and st.max_id() == aid
    st.review(aid, "confirmed", "оператор", "ок")
    aid2, k2 = st.apply_update(alert(kind="update", end=30.0, conf=0.9), meta)      # продление события не сбрасывает решение оператора
    a = st.get(aid)
    assert aid2 == aid and k2 == "update" and a["end_sec"] == 30.0 and a["status"] == "confirmed" and a["confidence"] == 0.9 and a["box"] == [1, 2, 3, 4]
    assert st.apply_update(alert(key="zzz", kind="close"), meta) == (-1, "close")
    st.apply_update(alert(kind="close"), meta)
    assert st.get(aid)["state"] == "closed"
    st.apply_update(alert(key="c:1:50.00", start=50.0, end=60.0, conf=0.6), {**meta, "t_abs": "2026-10-09T10:00:50"})
    s = st.stats("2026-10-09")
    assert (s["new"], s["confirmed"], s["total"], s["precision"]) == (1, 1, 2, 1.0) and s["today"]["new"] == 1 and s["median_review_sec"] is not None
    df = st.list_alerts(status=["new"], districts=["gatchina"], min_conf=0.5)
    assert len(df) == 1 and df.iloc[0]["key"] == "c:1:50.00"
    assert st.list_alerts(since="2026-10-09T10:00:30").shape[0] == 1 and st.list_alerts(newest_first=False).iloc[0]["id"] == aid
    assert st.cameras().iloc[0]["alerts"] == 2 and st.cameras().iloc[0]["pending"] == 1
    with pytest.raises(ValueError):
        st.review(aid, "maybe")


def make_stream(tmp_path, name="gatchina-03-20261009T143000", n_chunks=2):
    d = tmp_path / "streams" / name
    d.mkdir(parents=True)
    for i in range(n_chunks):
        (d / f"c{i + 1}.mp4").write_bytes(b"x")
    return d


def synthetic_frames(chunk, offset, fps, dur=30.0):
    for i in range(int(dur * fps)):
        yield np.full((120, 160, 3), (i * 5) % 255, np.uint8), offset + i / fps


def fake_engine_factory(gestures, keep_frames=True):
    def f(profile, cam, fps):
        return StreamEngine(profile.cfg(), cam, make_pose_fn(gestures), keep_frames=keep_frames)

    return f


PROF = PR.Profile("t", config={"cycles.th_in": 0.35, "cycles.th_out": 0.55, "events.confidence_threshold": 0.5})


def test_worker_processes_chunks_writes_alert_artifacts_and_resumes(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    streams = make_stream(tmp_path).parent
    store = AlertStore(tmp_path / "m.db")
    logs = []
    stream = S.scan_streams(streams)[0]
    eng = fake_engine_factory(G2)(PROF, stream.camera_id, 10.0)
    frames = lambda chunk, off, fps: synthetic_frames(chunk, off, fps)                     # noqa: E731
    r = W.process_stream(stream, store, eng, PROF, frames_fn=frames, probe_fn=lambda c: SimpleNamespace(duration=30.0), artifacts_dir=tmp_path / "art", log=logs.append)
    assert r["chunks"] == 2 and r["alerts"] == 1
    al = store.list_alerts()
    assert len(al) == 1
    a = al.iloc[0]
    assert a["camera_id"] == "gatchina-03" and a["district"] == "gatchina" and a["t_abs"].startswith("2026-10-09T14:30:0") and a["status"] == "new" and a["state"] == "closed"
    assert a["chunk"].endswith("c1.mp4") and a["chunk_offset"] == 0.0 and a["fingerprint"] == PROF.fingerprint()
    assert (tmp_path / "art" / str(a["id"]) / "thumb.jpg").exists() and (tmp_path / "art" / str(a["id"]) / "clip.mp4").stat().st_size > 0
    assert any("ТРЕВОГА" in x for x in logs) and store.cameras().iloc[0]["processed_sec"] == 60.0 and store.cameras().iloc[0]["status"] == "idle"
    # повторный запуск: все фрагменты уже обработаны — повторных тревог нет
    r2 = W.process_stream(stream, store, fake_engine_factory(G2)(PROF, "gatchina-03", 10.0), PROF, frames_fn=frames, probe_fn=lambda c: SimpleNamespace(duration=30.0),
                          artifacts_dir=tmp_path / "art", log=logs.append)
    assert r2["chunks"] == 0 and len(store.list_alerts()) == 1
    # новый фрагмент в папке подхватывается с правильным смещением
    (stream.path / "c3.mp4").write_bytes(b"x")
    r3 = W.process_stream(stream, store, fake_engine_factory([])(PROF, "gatchina-03", 10.0), PROF, frames_fn=frames, probe_fn=lambda c: SimpleNamespace(duration=30.0),
                          artifacts_dir=tmp_path / "art", log=logs.append)
    assert r3["chunks"] == 1 and store.chunks_done("gatchina-03")["c3.mp4"] == (30.0, 60.0)


def test_worker_survives_broken_chunk_and_artifact_failure(tmp_path):
    streams = make_stream(tmp_path).parent
    store = AlertStore(tmp_path / "m.db")
    stream = S.scan_streams(streams)[0]
    calls = {"n": 0}

    def probe(c):
        calls["n"] += 1
        if c.name == "c1.mp4":
            raise OSError("битый файл")
        return SimpleNamespace(duration=45.0)

    logs = []
    eng = fake_engine_factory(G2, keep_frames=False)(PROF, "gatchina-03", 10.0)                # без кольца кадров: миниатюры нет, тревога всё равно создаётся
    r = W.process_stream(stream, store, eng, PROF, frames_fn=lambda c, o, f: synthetic_frames(c, o, f, 45.0), probe_fn=probe, artifacts_dir=tmp_path / "a", log=logs.append)
    assert r["chunks"] == 1 and any("не открывается" in x for x in logs) and "c1.mp4" in store.chunks_done("gatchina-03")


def test_run_monitor_stop_event_and_camera_filter(tmp_path):
    streams = make_stream(tmp_path).parent
    make_stream(tmp_path, "pavlovsk-01-20261009T100000", 1)
    store = AlertStore(tmp_path / "m.db")
    stop = threading.Event()
    seen = []

    def frames(c, o, f):
        seen.append(c.parent.name)
        yield from synthetic_frames(c, o, f, 5.0)

    tot = W.run_monitor(streams, PROF, store, cameras=["pavlovsk-01"], engine_factory=fake_engine_factory([], keep_frames=False), frames_fn=frames,
                        probe_fn=lambda c: SimpleNamespace(duration=5.0), artifacts_dir=tmp_path / "a", stop=stop, log=lambda m: None)
    assert set(seen) == {"pavlovsk-01-20261009T100000"} and tot["chunks"] == 1
    stop.set()
    assert W.run_monitor(streams, PROF, store, watch=True, stop=stop, engine_factory=fake_engine_factory([]), log=lambda m: None) == dict(chunks=0, alerts=0, updates=0)


def test_feedback_exports_decisions_to_gt_once(tmp_path):
    st = AlertStore(tmp_path / "m.db")
    base = dict(district="d", cam_index="1", chunk="$ROOT/streams/gatchina-03-x/c2.mp4", chunk_offset=30.0, fingerprint="fp")
    a1, _ = st.apply_update(alert("k1", start=40.0, end=55.0, box=(10, 20, 110, 220)), {**base, "t_abs": "2026-10-09T10:00:40"})
    a2, _ = st.apply_update(alert("k2", start=70.0, end=80.0), {**base, "t_abs": "2026-10-09T10:01:10"})
    a3, _ = st.apply_update(alert("k3", start=90.0, end=99.0), {**base, "t_abs": "2026-10-09T10:01:30"})        # без решения
    st.review(a1, "confirmed", "иван")
    st.review(a2, "false", "иван")
    f = tmp_path / "gt.csv"
    out = FB.export_reviewed(st, f)
    assert out["added"] == 2 and out["already"] == 0
    gt = GT.load(f).set_index("note")
    pos, neg = gt.loc[f"alert:{a1}"], gt.loc[f"alert:{a2}"]
    assert (pos.label, pos.start_sec, pos.end_sec, pos.x2, pos.person_gt_id, pos.labeler) == ("POSITIVE", 10.0, 25.0, 110.0, "1", "иван")      # время — от начала фрагмента
    assert neg.label == "NEGATIVE" and pd.isna(neg.x1) and pos.clip_id == "gatchina-03-x__c2"
    assert FB.export_reviewed(st, f) == dict(added=0, skipped=0, already=2)
    st.review(a3, "confirmed")                                             # подтверждена, но рамки нет — в POSITIVE превратить нельзя
    assert FB.alert_to_gt_row({**st.get(a3), "box": None}) is None


def _ffmpeg():
    try:
        from sd.proxy import ffmpeg_exe

        return ffmpeg_exe()
    except Exception:
        return None


@pytest.mark.skipif(_ffmpeg() is None, reason="нет ffmpeg")
def test_end_to_end_with_real_video_chunks(tmp_path):
    """Настоящие файлы: два фрагмента по 25 с (ffmpeg) → чтение кадров → поза подставная → тревога на стыке фрагментов → миниатюра и клип на диске."""
    import subprocess

    pytest.importorskip("cv2")
    d = tmp_path / "streams" / "gatchina-03-20261009T143000"
    d.mkdir(parents=True)
    for i in (1, 2):
        subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=25", "-pix_fmt", "yuv420p", str(d / f"part{i}.mp4")], check=True)
    store = AlertStore(tmp_path / "m.db")
    # жесты: первый в конце 1-го фрагмента (18–23 с), второй — во 2-м (28–33 с): событие собирается через границу фрагментов
    gest = [(18.0, 2.0, 1.5, 1.5), (28.0, 2.0, 1.5, 1.5)]
    tot = W.run_monitor(tmp_path / "streams", PROF, store, engine_factory=lambda p, cam, fps: StreamEngine(p.cfg(), cam, make_pose_fn(gest)), process_fps=10.0, speed=0.0,
                        artifacts_dir=tmp_path / "art", log=lambda m: None)
    assert tot["chunks"] == 2
    al = store.list_alerts()
    assert len(al) == 1
    a = al.iloc[0]
    assert a["camera_id"] == "gatchina-03" and 16.0 <= a["start_sec"] <= 18.6 and a["end_sec"] > 30.0       # время потока непрерывно через стык фрагментов
    assert a["t_abs"].startswith("2026-10-09T14:30:1")                                                      # 14:30:00 + ≈17 с
    art = tmp_path / "art" / str(a["id"])
    assert (art / "thumb.jpg").stat().st_size > 500
    assert store.chunks_done("gatchina-03")["part2.mp4"][1] == pytest.approx(25.0, abs=0.3)
