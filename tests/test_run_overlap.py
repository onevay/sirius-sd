"""Запуски одного видео с пересекающимися окнами не считаются вместе (двойной учёт циклов), а побеждает самый ранний запуск."""
import json
import os

from sd.stages import drop_overlapping_runs, run_window


def _run(tmp_path, video, name, start, end, dur=60.0, created=1000.0):
    rd = tmp_path / video / name
    (rd / "pose").mkdir(parents=True)
    (rd / "run.json").write_text(json.dumps({"meta": {"start": start, "end": end, "video_info": {"duration": dur}}}), encoding="utf-8")
    tracks = rd / "pose" / "tracks.parquet"
    tracks.write_bytes(b"x")
    os.utime(tracks, (created, created))                                       # время создания запуска
    return video, rd


def test_run_window_end_none_means_until_video_end(tmp_path):
    _, rd = _run(tmp_path, "v", "0-ends", 0.0, None, dur=25.6)
    assert run_window(rd) == (0.0, 25.6)
    assert run_window(tmp_path / "нет такого") == (0.0, 0.0)


def test_earliest_run_wins_over_later_overlapping_one(tmp_path):
    main = _run(tmp_path, "v", "36-66s", 36.0, 66.0, dur=66.0, created=1000.0)          # основной запуск: по его циклам построены метки и признаки
    ui = _run(tmp_path, "v", "0-40s", 0.0, 40.0, dur=66.0, created=5000.0)               # пробный запуск в интерфейсе позже, окно пересекается на 4 с и длиннее
    other = _run(tmp_path, "w", "0-30s", 0.0, 30.0, dur=30.0)                            # другое видео не затрагивается
    assert drop_overlapping_runs([ui, main, other]) == [main, other]                    # порядок как у входа; длинное позднее окно основной не вытесняет


def test_same_window_rerun_does_not_replace_original(tmp_path):
    full = _run(tmp_path, "v", "0-ends", 0.0, None, dur=25.6, created=1000.0)
    ui = _run(tmp_path, "v", "0-25.6s", 0.0, 25.6, dur=25.6, created=9000.0)
    assert drop_overlapping_runs([ui, full]) == [full]


def test_disjoint_windows_of_same_video_are_both_kept(tmp_path):
    a = _run(tmp_path, "v", "0-30s", 0.0, 30.0, dur=100.0, created=1000.0)
    b = _run(tmp_path, "v", "60-90s", 60.0, 90.0, dur=100.0, created=2000.0)
    assert set(drop_overlapping_runs([a, b])) == {a, b}


def test_tiny_overlap_is_not_a_conflict(tmp_path):
    a = _run(tmp_path, "v", "0-30s", 0.0, 30.0, dur=100.0)
    b = _run(tmp_path, "v", "29.5-60s", 29.5, 60.0, dur=100.0, created=2000.0)           # перекрытие 0.5 с < порога 1 с
    assert len(drop_overlapping_runs([a, b])) == 2
