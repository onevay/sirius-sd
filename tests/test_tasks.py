import json
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
    tid = T.start("gt_from_gestures", {"labeler": "x"}, tmp_path)       # реальный процесс `python -m sd …`; результат нас не волнует, важно завершение и код
    for _ in range(90):
        s = T.status(tid, tmp_path)
        if s["state"] != "идёт":
            break
        time.sleep(1)
    assert s["state"] in ("готово", "ошибка") and s["returncode"] is not None
    assert (tmp_path / tid / "log.txt").exists()
