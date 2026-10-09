import importlib
import json
import os

import numpy as np
import pandas as pd

from sd import _env
from sd import paths as P
from sd.tracks import Tracks


def test_portable_roundtrip_and_foreign_windows_path(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DATA", tmp_path / "data")
    f = tmp_path / "data" / "курение" / "a.mp4"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    s = P.portable(f)
    assert s == "$DATA/курение/a.mp4" and P.from_portable(s) == f
    # путь из старого артефакта на Windows находится по хвосту внутри data/
    assert P.from_portable(r"C:\Users\me\proj\data\курение\a.mp4") == f
    assert P.portable("/somewhere/else.mp4").endswith("/somewhere/else.mp4") and "$DATA" not in P.portable("/somewhere/else.mp4")      # на Windows добавляется диск
    assert P.portable(None) is None and P.from_portable("") is None


def test_tracks_meta_video_survives_moving_data(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DATA", tmp_path / "A" / "data")
    v = tmp_path / "A" / "data" / "cam" / "x.mp4"
    v.parent.mkdir(parents=True)
    v.write_bytes(b"x")
    df = pd.DataFrame(dict(frame=[0], t=[0.0], tid=[1], x1=[0], y1=[0], x2=[1], y2=[1], score=[1.0], h=[1.0]))
    Tracks(df, np.zeros((1, 17, 3), np.float32), pd.DataFrame(dict(frame=[0], t=[0.0])), dict(video=P.portable(v))).save(tmp_path / "pose")
    assert json.loads((tmp_path / "pose" / "meta.json").read_text())["video"] == "$DATA/cam/x.mp4"
    monkeypatch.setattr(P, "DATA", tmp_path / "B" / "data")          # «другая машина»: данные лежат в другом месте
    w = tmp_path / "B" / "data" / "cam" / "x.mp4"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"x")
    import sd.tracks as T

    monkeypatch.setattr(T, "from_portable", P.from_portable)
    assert Tracks.load(tmp_path / "pose").meta["video"] == str(w)


def test_dotenv_sets_missing_variables_only(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('# комментарий\nSD_TEST_A=1\nSD_TEST_B="два"\n\nSD_TEST_C=x=y\n', encoding="utf-8")
    monkeypatch.setenv("SD_TEST_A", "заранее")
    monkeypatch.delenv("SD_TEST_B", raising=False)
    monkeypatch.delenv("SD_TEST_C", raising=False)
    got = _env.load_dotenv(tmp_path / ".env")
    assert os.environ["SD_TEST_A"] == "заранее" and os.environ["SD_TEST_B"] == "два" and os.environ["SD_TEST_C"] == "x=y" and "SD_TEST_A" not in got
    assert _env.load_dotenv(tmp_path / "нет") == {}


def test_data_dirs_can_be_moved_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SD_DATA", str(tmp_path / "big" / "data"))
    monkeypatch.setenv("SD_OUTPUTS", str(tmp_path / "out"))
    try:
        m = importlib.reload(P)
        assert m.DATA == tmp_path / "big" / "data" and m.OUTPUTS == tmp_path / "out" and m.MODELS.name == "models"
    finally:
        monkeypatch.undo()
        importlib.reload(P)
