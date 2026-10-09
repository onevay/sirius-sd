"""Запуск без единого человека (пустая сцена, клип без людей) не должен ронять этапы признаков, циклов и событий (`sd clips-run` падал с KeyError: 'tid')."""
import numpy as np
import pandas as pd

from sd import stages
from sd.tracks import Tracks


def _empty_tracks():
    df = pd.DataFrame(columns=["frame", "t", "tid", "x1", "y1", "x2", "y2", "score", "h"])
    grid = pd.DataFrame(dict(frame=np.arange(20), t=np.arange(20) / 10.0))
    meta = dict(video_info=dict(width=320, height=240, fps=10.0), stride=1, start=0.0, end=None)
    return Tracks(df, np.zeros((0, 17, 3), np.float32), grid, meta)


def test_features_cycles_events_without_people(cfg, tmp_path):
    tr = _empty_tracks()
    ser = stages.stage_features(tr, cfg, tmp_path)
    assert ser.empty and {"tid", "frame", "t"} <= set(ser.columns)
    cyc, rej, st = stages.stage_cycles(ser, cfg, tmp_path)
    assert cyc.empty and rej.empty and st.empty and "tid" in cyc.columns
    ev, events = stages.stage_events(tr, ser, cyc, cfg, tmp_path, "cam", "clip")
    assert ev.empty and events == []


def test_cached_series_without_columns_is_repaired(cfg, tmp_path):
    """Кэш `series.parquet`, записанный до исправления (пустая таблица без колонок), читается как пустая таблица с ключами."""
    tr = _empty_tracks()
    d = tmp_path / "features" / stages._feat_hash(cfg)
    d.mkdir(parents=True)
    pd.DataFrame().to_parquet(d / "series.parquet", index=False)
    ser = stages.stage_features(tr, cfg, tmp_path)
    assert ser.empty and "tid" in ser.columns
    cyc, _, _ = stages.stage_cycles(pd.DataFrame(), cfg, tmp_path / "other")          # и пустая таблица вообще без колонок
    assert cyc.empty
