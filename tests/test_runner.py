import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sd import experiments as XP
from sd import gt as G
from sd import library as LIB
from sd import profiles as PR
from sd import runner as R
from sd.paths import video_id

COLS = R.EVENT_COLUMNS


def ev_row(s, e, conf, box=(0, 0, 100, 200), tid=1, clip="x"):
    return dict(camera_id="cam", clip_id=clip, event_id=f"{clip}_{s}", start_sec=s, end_sec=e, confidence=conf, label="smoking_like", person_track_id=tid, peak_sec=(s + e) / 2,
                x1=box[0], y1=box[1], x2=box[2], y2=box[3])


def make_world(tmp_path):
    d1, d2 = tmp_path / "курение", tmp_path / "лжекурение"
    d1.mkdir(); d2.mkdir()
    for n in ("s1.mp4", "s2.mp4"):
        (d1 / n).write_bytes(b"x" * 10)
    (d2 / "f1.mp4").write_bytes(b"x" * 10)
    (d2 / "notes.txt").write_text("не видео")
    return d1, d2


class Fake:
    """Подставной распознаватель: события по имени клипа; считает вызовы."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, video, start, end, profile):
        self.calls.append((Path(video).stem, start, end))
        ev = pd.DataFrame([ev_row(*r) for r in self.table.get(Path(video).stem, [])], columns=COLS)
        return SimpleNamespace(events=ev, meta=dict(run_dir="/rd", out_dir="/od", counts=dict(cycles=3, people=1), warnings=[]))


def probe(v):
    return SimpleNamespace(duration=100.0)


def test_discover_and_select_dirs(tmp_path):
    d1, d2 = make_world(tmp_path)
    df = LIB.discover_dirs(roots=[tmp_path])
    assert sorted(df.clips) == [1, 2] and set(df.weak) == {"smoking", "fake"}
    assert [v.name for v in LIB.videos_in([d2])] == ["f1.mp4"]
    assert len(LIB.videos_in([d1, d1, d2])) == 3                       # дубли папок не удваивают клипы
    other = tmp_path / "другая" / "курение"
    other.mkdir(parents=True)
    (other / "s1.mp4").write_bytes(b"x")
    # папки разные, но video_id у них совпадает по «папка__имя» только если папки одноимённые
    with pytest.raises(ValueError, match="совпал"):
        LIB.clip_ids(LIB.videos_in([d1, other]))


def test_run_clips_caches_by_profile_and_isolates_errors(tmp_path):
    d1, d2 = make_world(tmp_path)
    fake = Fake({"s1": [(10, 20, 0.9)]})
    vids = LIB.videos_in([d1])
    kw = dict(recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "cache")
    prof = PR.default_profile()
    r1 = R.run_clips(vids, prof, **kw)
    assert [r.cached for r in r1] == [False, False] and len(fake.calls) == 2 and len(r1[0].events) == 1 and r1[0].events.clip_id.iloc[0] == video_id(vids[0])
    r2 = R.run_clips(vids, prof, **kw)
    assert [r.cached for r in r2] == [True, True] and len(fake.calls) == 2 and len(r2[0].events) == 1      # из кэша, модель не вызывалась
    # смена порога не инвалидирует кэш, смена модели — инвалидирует
    R.run_clips(vids, PR.Profile("t", config={"events.confidence_threshold": 0.9}), **kw)
    assert len(fake.calls) == 2
    R.run_clips(vids, PR.Profile("m", config={"pose.weights": "yolo11m-pose"}), **kw)
    assert len(fake.calls) == 4

    def boom(video, *a):
        raise RuntimeError("нет кадров")

    bad = R.run_clips(vids, PR.Profile("z", config={"pose.imgsz": 480}), recognize_fn=boom, probe_fn=probe, cache_root=tmp_path / "cache")
    assert all(r.error and "нет кадров" in r.error and r.events.empty for r in bad)


def test_window_for_caps_long_clips():
    assert R.window_for(100.0) == (0.0, None) and R.window_for(100.0, max_sec=30) == (0.0, 30.0) and R.window_for(30.4, max_sec=30) == (0.0, None)
    assert R.window_for(100.0, (10.0, 50.0), 5) == (10.0, 50.0)


def _gt(tmp_path, d1, d2):
    f = tmp_path / "gt.csv"
    G.add(video_id(d1 / "s1.mp4"), 8, 22, "POSITIVE", person=1, box=(0, 0, 100, 200), path=f)
    G.add(video_id(d1 / "s2.mp4"), 30, 45, "POSITIVE", person=1, box=(0, 0, 100, 200), path=f)
    G.mark_clean(video_id(d2 / "f1.mp4"), 100.0, path=f)
    return G.load(f)


def test_evaluate_dirs_events_mode_end_to_end(tmp_path):
    d1, d2 = make_world(tmp_path)
    gt = _gt(tmp_path, d1, d2)
    fake = Fake({"s1": [(10, 20, 0.9)], "s2": [(31, 44, 0.3)], "f1": [(5, 12, 0.8)]})
    out = R.evaluate_dirs([d1, d2], PR.heuristic_profile(), mode="events", policy="fixed", gt_df=gt, recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c",
                          exp_root=tmp_path / "exp", n_boot=20, target_f1=0.8)
    m = out.report.metrics
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1) and m["clips"] == 3 and m["neg_clips"] == 1
    assert set(out.report.errors.cause) == {"below_threshold", "hard_negative"}
    runs = XP.list_runs(tmp_path / "exp")
    assert len(runs) == 1 and runs.f1.iloc[0] == pytest.approx(m["f1"]) and runs.reached.iloc[0] in (False, np.False_)
    rep, meta, ev, gt_saved = XP.load_run(out.run_id, tmp_path / "exp")
    assert len(ev) == 3 and len(gt_saved) == 3 and meta["gt_fingerprint"] and meta["describe"]["pose"] and rep.metrics["tp"] == 1
    assert XP.delete_run(out.run_id, tmp_path / "exp") and XP.list_runs(tmp_path / "exp").empty
    # порог можно пересчитать по сохранённым событиям без нового прогона: при 0.25 событие s2 находится
    from sd import evaluation as EV
    preds = [p for c in ev.clip_id.unique() for p in EV.to_preds(ev[ev.clip_id == c], c)]
    gts = G.to_eval_gts(gt_saved, set(ev.clip_id) | {video_id(d2 / "f1.mp4")})
    dur = {c["clip_id"]: c["duration"] for c in meta["clips"]}
    assert EV.metrics_at(preds, gts, EV.Settings(threshold=0.25), dur)[0]["tp"] == 2


def test_only_selected_dirs_and_only_labeled_clips_are_evaluated(tmp_path):
    d1, d2 = make_world(tmp_path)
    gt = _gt(tmp_path, d1, d2)
    G.delete(gt[gt.clip_id == video_id(d1 / "s2.mp4")].id, path=tmp_path / "gt.csv")
    gt = G.load(tmp_path / "gt.csv")
    fake = Fake({"s1": [(10, 20, 0.9)]})
    out = R.evaluate_dirs([d1], PR.heuristic_profile(), policy="fixed", gt_df=gt, recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c", save=False, n_boot=10)
    assert [c[0] for c in fake.calls] == ["s1"]                                      # лишние папки и неразмеченный s2 не запускались
    assert out.report.metrics["clips"] == 1 and any("неразмеченных" in n for n in out.report.notes)
    with pytest.raises(ValueError, match="нет размеченных"):
        R.evaluate_dirs([d2], PR.heuristic_profile(), gt_df=G.empty(), recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c", save=False)
    with pytest.raises(ValueError, match="нет видео"):
        R.evaluate_dirs([tmp_path / "пусто"], PR.heuristic_profile(), gt_df=gt, recognize_fn=fake, probe_fn=probe, save=False)


def test_hidden_role_freezes_threshold(tmp_path):
    d1, d2 = make_world(tmp_path)
    gt = _gt(tmp_path, d1, d2)
    fake = Fake({"s1": [(10, 20, 0.9)]})
    with pytest.raises(ValueError, match="фиксируется"):
        R.evaluate_dirs([d1], PR.heuristic_profile(), role="hidden", policy="plateau", gt_df=gt, recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c", save=False)
    out = R.evaluate_dirs([d1], PR.Profile("h", config={"events.confidence_threshold": 0.77}, options={"allow_heuristic": True}), role="hidden", policy="fixed", gt_df=gt, recognize_fn=fake, probe_fn=probe,
                          cache_root=tmp_path / "c", save=False, n_boot=10)
    assert out.report.settings["threshold"] == 0.77 and any("не подбирался" in n for n in out.report.notes)


def test_weak_mode_by_folder_labels(tmp_path):
    d1, d2 = make_world(tmp_path)
    fake = Fake({"s1": [(10, 20, 0.9)], "s2": [], "f1": [(5, 12, 0.8)]})
    out = R.evaluate_dirs([d1, d2], PR.heuristic_profile(), mode="clips", policy="fixed", recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c", save=False, n_boot=10)
    m = out.report.metrics
    assert out.report.mode == "clips" and (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    with pytest.raises(ValueError, match="обоих классов"):
        R.evaluate_dirs([d1], PR.heuristic_profile(), mode="clips", recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c", save=False)


def test_gt_outside_window_is_not_a_miss(tmp_path):
    d1, d2 = make_world(tmp_path)
    f = tmp_path / "gt.csv"
    cid = video_id(d1 / "s1.mp4")
    G.add(cid, 8, 22, "POSITIVE", person=1, box=(0, 0, 100, 200), path=f)
    G.add(cid, 70, 85, "POSITIVE", person=1, box=(0, 0, 100, 200), path=f)           # за пределами окна 0–30 с
    fake = Fake({"s1": [(10, 20, 0.9)]})
    out = R.evaluate_dirs([d1], PR.heuristic_profile(), policy="fixed", gt_df=G.load(f), recognize_fn=fake, probe_fn=probe, max_sec=30, cache_root=tmp_path / "c", save=False, n_boot=10)
    assert out.report.metrics["fn"] == 0 and out.report.metrics["tp"] == 1 and out.runs[0].duration == 30.0


def test_roi_filters_predictions_before_scoring(tmp_path):
    d1, d2 = make_world(tmp_path)
    gt = _gt(tmp_path, d1, d2)
    roi = tmp_path / "roi.json"
    roi.write_text(json.dumps({"polygon": [[500, 500], [600, 500], [600, 600]]}))      # событие в рамке (0..100, 0..200) вне зоны
    fake = Fake({"s1": [(10, 20, 0.9)]})
    out = R.evaluate_dirs([d1], PR.heuristic_profile(), policy="fixed", gt_df=gt, roi_path=roi, recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c", save=False, n_boot=10)
    assert out.report.metrics["tp"] == 0 and out.report.metrics["fn"] >= 1


def test_all_clips_failed_is_an_error_not_f1_zero(tmp_path):
    d1, d2 = make_world(tmp_path)
    gt = _gt(tmp_path, d1, d2)

    def boom(video, start, end, profile):
        raise RuntimeError("CUDA недоступна")

    with pytest.raises(RuntimeError, match="ни один клип не обработан.*CUDA"):
        R.evaluate_dirs([d1, d2], PR.heuristic_profile(), mode="events", policy="fixed", gt_df=gt, recognize_fn=boom, probe_fn=probe, cache_root=tmp_path / "c", exp_root=tmp_path / "exp",
                        n_boot=5)
    assert XP.list_runs(tmp_path / "exp").empty          # в журнал такой прогон не попадает
