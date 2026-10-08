"""Регрессионные тесты на найденное аудитом (docs/AUDIT_AND_CHANGES.md, раздел «Аудит 2»)."""
import os
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from rt_helpers import make_pose_fn, run_stream
from sd import gt as GT
from sd import library as LIB
from sd import paths as P
from sd import profiles as PR
from sd import runner as R
from sd.config import load_config
from sd.realtime.engine import StreamEngine


def test_concurrent_writers_do_not_lose_rows(tmp_path):
    f = tmp_path / "gt.csv"
    errs = []

    def worker(k):
        try:
            for i in range(8):
                GT.add(f"clip{k}", i, i + 1, "IGNORE", path=f)
        except Exception as e:                                         # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(GT.load(f)) == 32 and GT.load(f).id.is_unique and not list(tmp_path.glob("*.lock"))


def test_stale_lock_is_removed_and_busy_lock_times_out(tmp_path):
    f = tmp_path / "gt.csv"
    lock = tmp_path / "gt.csv.lock"
    lock.write_text("")
    old = time.time() - 120
    os.utime(lock, (old, old))
    GT.add("c", 0, 1, "IGNORE", path=f)                               # брошенный замок (процесс упал) не блокирует навсегда
    assert len(GT.load(f)) == 1
    lock.write_text("")
    with pytest.raises(TimeoutError):
        with GT.locked(f, timeout=0.2):
            pass
    lock.unlink()


def test_portable_token_cannot_escape_root(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DATA", tmp_path / "data")
    assert P.from_portable("$DATA/../../etc/passwd") == tmp_path / "data"
    assert P.from_portable("$DATA/a/b.mp4") == tmp_path / "data" / "a" / "b.mp4"


def test_discover_dirs_limits_depth_and_skips_service_dirs(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g"
    deep.mkdir(parents=True)
    (deep / "x.mp4").write_bytes(b"x")                                 # глубже 5 уровней — не ищем
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "v.mp4").write_bytes(b"x")
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "y.mp4").write_bytes(b"x")
    df = LIB.discover_dirs(roots=[tmp_path])
    assert [Path_.name for Path_ in map(__import__("pathlib").Path, df.path)] == ["ok"]


def test_clip_that_cannot_be_probed_still_counts_as_misses(tmp_path):
    d = tmp_path / "курение"
    d.mkdir()
    (d / "s1.mp4").write_bytes(b"x")
    from sd.paths import video_id

    f = tmp_path / "gt.csv"
    GT.add(video_id(d / "s1.mp4"), 8, 22, "POSITIVE", person=1, box=(0, 0, 100, 200), path=f)

    def bad_probe(v):
        raise OSError("битый контейнер")

    out = R.evaluate_dirs([d], PR.default_profile(), policy="fixed", gt_df=GT.load(f), recognize_fn=lambda *a: None, probe_fn=bad_probe, cache_root=tmp_path / "c", save=False, n_boot=10)
    m = out.report.metrics
    assert m["clips"] == 1 and m["fn"] == 1 and m["tp"] == 0 and m["f1"] == 0.0          # сбой не «исчезает» из метрики
    assert any("ошибк" in n for n in out.report.notes)


def test_engine_state_stays_bounded_over_long_stream():
    cfg = load_config(overrides=["cycles.th_in=0.35", "cycles.th_out=0.55", "events.confidence_threshold=0.5"])
    eng = StreamEngine(cfg, "c", make_pose_fn([(5.0 + 60 * k, 2.0, 1.5, 1.5) for k in range(40)] + [(16.0 + 60 * k, 2.0, 1.5, 1.5) for k in range(40)]), keep_frames=False, buffer_sec=30.0)
    ups = run_stream(eng, 1800.0, fps=5.0, flush=False)
    opens = [u for u in ups if u.kind == "open"]
    assert len(opens) >= 25                                             # события по-прежнему находятся через полчаса потока
    assert len(eng._emitted) < 15 and sum(len(v) for v in eng._scored.values()) < 20 and len(eng._frames) < 30 * 5 + 20
