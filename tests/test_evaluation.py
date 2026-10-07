import numpy as np
import pandas as pd
import pytest

from sd import evaluation as E
from sd.evaluate import GT, Pred, evaluate

B = (0.0, 0.0, 100.0, 200.0)


def P(clip, s, e, conf=0.9, box=B):
    return Pred(clip, s, e, conf, (s + e) / 2, box)


def test_overlapping_ignore_intervals_are_not_double_counted():
    # два перекрывающихся IGNORE по 6 с покрывают 8 с из 20 -> 40% < 50%: предсказание остаётся FP
    gts = [GT("c", "1", 0, 6, "IGNORE"), GT("c", "1", 2, 8, "IGNORE")]
    r = evaluate([Pred("c", 0, 20, 0.9, 10, B)], gts)
    assert r.fp == 1 and r.ignored_preds == 0
    # а при реальном покрытии >= 50% — подавляется
    r2 = evaluate([Pred("c", 0, 10, 0.9, 5, B)], gts)
    assert r2.ignored_preds == 1 and r2.fp == 0


def test_error_budget_matches_f1_formula():
    b = E.error_budget(tp=10, fp=2, fn=3, target=0.8)
    # F1 = 20/25 = 0.8 -> цель достигнута ровно, допустимо 5 ошибок
    assert b["allowed"] == 5 and b["errors"] == 5 and b["f1"] == pytest.approx(0.8) and b["reached"] and b["over"] == 0
    c = E.error_budget(tp=10, fp=4, fn=3, target=0.8)
    assert not c["reached"] and c["over"] == 2 and c["gain_fix_fp"] > 0 and c["cost_extra_fn"] > 0
    z = E.error_budget(tp=0, fp=1, fn=1, target=0.8)
    assert z["allowed"] == 0 and z["cost_extra_fn"] == 0.0


def test_plateau_prefers_center_over_sharp_peak():
    th = np.round(np.arange(0.1, 0.91, 0.1), 2)
    f1 = np.array([.5, .6, .8, .81, .8, .79, .5, .4, .3])
    cur = pd.DataFrame(dict(threshold=th, f1=f1))
    p = E.plateau(cur, tol=0.02)
    assert p["best_threshold"] == pytest.approx(0.4) and p["lo"] == pytest.approx(0.3) and p["hi"] == pytest.approx(0.6) and p["center"] == pytest.approx(0.4)
    assert E.plateau(pd.DataFrame(columns=["threshold", "f1"]))["center"] is None


def _scene():
    dur = {"a": 600.0, "b": 3600.0, "n": 1800.0}
    gts = [GT("a", "1", 10, 25, "POSITIVE", box=B), GT("a", "2", 100, 115, "POSITIVE", box=(300, 0, 400, 200)), GT("b", "1", 50, 60, "NEGATIVE"),
           GT("a", "3", 200, 210, "POSITIVE", box=(500, 0, 600, 200)), GT("a", "9", 300, 310, "POSITIVE", box=(700, 0, 800, 200))]
    preds = [P("a", 12, 24, 0.95), P("a", 14, 22, 0.90),                     # TP + дубль (duplicate)
             P("a", 101, 114, 0.40, box=(300, 0, 400, 200)),                 # ниже порога -> FN below_threshold
             P("a", 201, 209, 0.9, box=(0, 0, 50, 50)),                      # время совпало, рамка нет -> wrong_person FP + FN wrong_box
             P("b", 52, 58, 0.9),                                            # на размеченном негативе
             P("n", 10, 20, 0.8)]                                            # фон
    return preds, gts, dur


def test_report_counts_and_error_causes():
    preds, gts, dur = _scene()
    rep = E.build_report(preds, gts, dur, E.Settings(threshold=0.5, target_f1=0.8), n_boot=40)
    m = rep.metrics
    assert (m["tp"], m["fp"], m["fn"]) == (1, 4, 3) and m["n_gt"] == 4
    causes = sorted(zip(rep.errors.kind, rep.errors.cause))
    assert causes == sorted([("FP", "duplicate"), ("FP", "wrong_person"), ("FP", "hard_negative"), ("FP", "background"),
                             ("FN", "below_threshold"), ("FN", "wrong_box"), ("FN", "missed")])
    # ложные тревоги в час считаются по клипам без позитивов (b, n): 2 FP на 1.5 часа
    assert m["fp_neg"] == 2 and m["fp_per_hour"] == pytest.approx(2 / 1.5)
    assert rep.budget["errors"] == 7 and not rep.budget["reached"]
    pc = rep.per_clip.set_index("clip_id")
    assert pc.loc["a", "status"] == "FP+FN" and pc.loc["n", "status"] == "FP"
    assert rep.errors.set_index("cause").loc["below_threshold", "confidence"] == pytest.approx(0.40)


def test_merged_events_are_classified():
    gts = [GT("a", "1", 10, 20, "POSITIVE", box=B), GT("a", "1", 24, 34, "POSITIVE", box=B)]
    preds = [P("a", 10, 34, 0.9)]
    rep = E.build_report(preds, gts, {"a": 100.0}, E.Settings(threshold=0.5), n_boot=20)
    assert rep.metrics["tp"] == 1 and rep.errors.cause.tolist() == ["merged"]


def test_threshold_curve_moves_precision_recall_and_finalize_picks_plateau():
    preds, gts, dur = _scene()
    cur = E.sweep(preds, gts, E.Settings(), dur, thresholds=[0.05, 0.99])
    lo, hi = cur.iloc[0], cur.iloc[1]
    assert lo.n_pred > hi.n_pred and hi.n_pred == 0 and hi.recall == 0
    rep = E.finalize_report(preds, gts, dur, E.Settings(threshold=0.95), policy="plateau", n_boot=20)
    assert rep.settings["threshold"] == pytest.approx(rep.plateau["center"]) and any("оптимистична" in n for n in rep.notes)
    fixed = E.finalize_report(preds, gts, dur, E.Settings(threshold=0.95), policy="fixed", n_boot=20)
    assert fixed.settings["threshold"] == 0.95 and not any("оптимистична" in n for n in fixed.notes)


def test_bootstrap_interval_contains_point_and_is_deterministic():
    preds, gts, dur = _scene()
    st = E.Settings(threshold=0.5)
    a = E.bootstrap(preds, gts, st, dur, n=60, seed=1)
    b = E.bootstrap(preds, gts, st, dur, n=60, seed=1)
    assert a == b and a["f1"][0] <= a["f1"][1]
    assert E.bootstrap(preds, gts, st, {"a": 1.0, "b": 1.0}, n=60)["n"] == 0        # меньше трёх клипов — интервала нет


def test_weak_clip_level_report():
    clips = pd.DataFrame(dict(clip_id=["s1", "s2", "s3", "f1", "f2"], y=[1, 1, 1, 0, 0], duration=[60.0] * 5,
                              events=[[0.9], [0.7, 0.6], [], [0.8], []]))
    rep = E.build_weak_report(clips, E.Settings(threshold=0.5), n_boot=30)
    m = rep.metrics
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1) and rep.mode == "clips" and m["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert set(rep.errors.cause) == {"clip_missed", "clip_false"} and m["fp_per_hour"] == pytest.approx(1 / (120 / 3600))


def test_report_save_load_roundtrip(tmp_path):
    preds, gts, dur = _scene()
    rep = E.build_report(preds, gts, dur, E.Settings(threshold=0.5), n_boot=20)
    rep.save(tmp_path / "r")
    back = E.Report.load(tmp_path / "r")
    assert back.metrics["tp"] == rep.metrics["tp"] and len(back.errors) == len(rep.errors) and back.mode == "events" and back.budget["target"] == 0.8


def test_no_gt_gives_zero_metrics_and_notes():
    rep = E.build_report([], [], {"a": 10.0}, E.Settings(), n_boot=10)
    assert rep.metrics["f1"] == 0.0 and rep.errors.empty and rep.notes
