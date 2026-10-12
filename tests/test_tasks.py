import json
import os
import time

import pytest

from sd import tasks as T


def test_command_builder_uses_only_declared_fields():
    c = T.command("enrich", {"vlm": "", "objects": "a,b", "evil": "; rm -rf /"})
    assert c[0] == "enrich" and "evil" not in " ".join(c) and "rm" not in c
    assert c[c.index("--objects") + 1] == "a,b"
    assert c[c.index("--vlm") + 1] == ""        # пусто = «не считать VLM», передаётся явно
    with pytest.raises(KeyError):
        T.command("rm", {})


def test_command_flags_and_lines():
    c = T.command("eval", {"dirs": "data/a\n\ndata/b", "profile": "p", "policy": "fixed"})
    assert c.count("--dir") == 2 and "data/b" in c
    c = T.command("oof_eval", {"by_scene": True, "sets": "fast", "repeats": 2})
    assert "--by-scene" in c and c[c.index("--repeats") + 1] == "2"
    assert "--no-calibrate" in T.command("train_bundle", {"calibrate": False, "name": "n"}) and T.command("train_bundle", {"name": "n"})[-1] == "n"


def test_status_states(tmp_path):
    d = tmp_path / "20260101_000000_doctor"
    d.mkdir()
    base = dict(id=d.name, task="doctor", title="x", args=["doctor"], started=time.time() - 5, pid=999999, finished=None, returncode=None)
    (d / "task.json").write_text(json.dumps(base), encoding="utf-8")
    assert T.status(d.name, tmp_path)["state"] == "прервана"
    (d / "task.json").write_text(json.dumps({**base, "finished": time.time(), "returncode": 0}), encoding="utf-8")
    assert T.status(d.name, tmp_path)["state"] == "готово"
    (d / "task.json").write_text(json.dumps({**base, "finished": time.time(), "returncode": 2}), encoding="utf-8")
    assert T.status(d.name, tmp_path)["state"] == "ошибка"
    assert T.stop(d.name, tmp_path) is False
    (d / "log.txt").write_bytes("строка 1\nстрока 2\n".encode("utf-8"))
    assert T.tail(d.name, 1, tmp_path) == "строка 2"


def test_start_runs_and_finishes(tmp_path):
    """Реальный процесс `python -m sd …`; берём команду без побочных эффектов (папки нет — быстро падает с ошибкой), чтобы тест не трогал данные и метки проекта."""
    tid = T.start("eval", {"dirs": str(tmp_path / "нет_такой_папки"), "profile": "default"}, tmp_path)
    for _ in range(120):
        s = T.status(tid, tmp_path)
        if s["state"] != "идёт":
            break
        time.sleep(1)
    assert s["state"] in ("готово", "ошибка") and s["returncode"] is not None
    assert (tmp_path / tid / "log.txt").exists()


def _fake_launch(calls):
    def launch(d, meta):
        calls.append(d.name)
        meta.update(pid=os.getpid(), started=time.time(), queued=False)      # «живой» pid — сам тест
        T._write_json(d / "task.json", meta)
    return launch


def test_heavy_tasks_queue_and_pump(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(T, "_launch", _fake_launch(calls))
    a = T.start("ladder", {}, tmp_path)
    b = T.start("ladder", {}, tmp_path)
    light = T.start("doctor", {}, tmp_path)
    assert calls == [a, light]                                               # вторая тяжёлая ждёт, лёгкая — сразу
    assert T.status(b, tmp_path)["state"] == "в очереди" and T.status(a, tmp_path)["state"] == "идёт"
    assert T.pump(tmp_path) is None                                          # первая ещё идёт
    T._write_json(tmp_path / a / "state.json", dict(finished=time.time(), returncode=0))
    assert T.pump(tmp_path) == b and calls[-1] == b
    assert T.status(a, tmp_path)["state"] == "готово"


def test_stop_queued_task_and_rerun(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(T, "_launch", _fake_launch(calls))
    a = T.start("ladder", {"configs": "n960"}, tmp_path)
    b = T.start("ladder", {"configs": "n1280"}, tmp_path)
    assert T.stop(b, tmp_path) is True and T.status(b, tmp_path)["state"] == "остановлена"
    T._write_json(tmp_path / a / "state.json", dict(finished=time.time(), returncode=0))
    assert T.pump(tmp_path) is None                                          # остановленная из очереди не запускается
    c = T.rerun(a, tmp_path)
    assert T.status(c, tmp_path)["values"] == {"configs": "n960"} and calls[-1] == c


def test_progress_from_log(tmp_path):
    d = tmp_path / "t1"
    d.mkdir()
    (d / "log.txt").write_text("старт\n  [3/10] клип a\nшум\n  [4/10] клип b\n", encoding="utf-8")
    p = T.progress("t1", tmp_path)
    assert p["done"] == 4 and p["total"] == 10 and abs(p["frac"] - 0.4) < 1e-9 and p["message"] == "клип b"
    assert T.progress("нет", tmp_path) is None


def test_old_task_json_format_still_reads(tmp_path):
    """Задачи, запущенные до введения state.json (finished/returncode в task.json), показываются правильно."""
    d = tmp_path / "20260101_000000_doctor"
    d.mkdir()
    (d / "task.json").write_text(json.dumps(dict(id=d.name, task="doctor", title="x", args=["doctor"], started=time.time() - 5, pid=1, finished=time.time(), returncode=0)), encoding="utf-8")
    assert T.status(d.name, tmp_path)["state"] == "готово"


def test_new_task_types_build_commands():
    assert T.command("report", {"run": "r1", "baseline": ""}) == ["report", "r1"]
    c = T.command("clf_train", {"spec": "s", "bundle": "b", "run": "r", "overwrite": False, "attach": "mvp"})
    assert c[:2] == ["clf", "train"] and c[c.index("--attach") + 1] == "mvp" and "--overwrite" not in c
    c = T.command("eval", {"dirs": "data/a", "profile": "p", "sets": "video.process_fps=10\noptions.objects=[a]", "max_sec": 0.0})
    assert c.count("--set") == 2 and "--max-sec" not in c
    assert "--no-importance" in T.command("clf_diagnose", {"spec": "s", "importance": False})
