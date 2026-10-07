import pytest

from sd.evaluate import GT, Pred, evaluate, iou, threshold_curve

BOX = (100.0, 100.0, 200.0, 300.0)


def gt(start=10, end=30, label="POSITIVE", box=BOX, clip="c1", pid="p1"):
    return GT(clip, pid, start, end, label, box=box)


def pred(start=10, end=30, peak=20, box=BOX, conf=0.9, clip="c1"):
    return Pred(clip, start, end, conf, peak, box)


def test_iou_basics():
    assert iou(BOX, BOX) == pytest.approx(1.0)
    assert iou(BOX, (300, 300, 400, 400)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_perfect_match():
    r = evaluate([pred()], [gt()])
    assert (r.tp, r.fp, r.fn) == (1, 0, 0) and r.f1 == 1.0 and r.median_latency == 0.0


def test_temporal_tolerance_plus_minus_3s():
    # предсказание целиком ПОСЛЕ эталона: эталон 10–30 расширяется до 7–33; пересечение 33–36 -> 0 <1 с -> FP+FN
    assert evaluate([pred(33.5, 40, 36)], [gt()]).tp == 0
    # пересечение с расширенной частью ровно >= 1 с (32–40 пересекает 32–33 = 1 с) -> совпало
    assert evaluate([pred(32.0, 40, 36)], [gt()]).tp == 1
    # слишком ранняя метка: 0–7.5 пересекает расширенный эталон на 0.5 с (<1) -> нет
    assert evaluate([pred(0, 7.5, 5)], [gt()]).tp == 0


def test_min_overlap_one_second():
    assert evaluate([pred(29.0, 31.0, 30)], [gt()]).tp == 1
    assert evaluate([pred(33.0, 33.9, 33.4)], [gt()]).tp == 0   # пересечение 0.0..: за границей расширенного окна


def test_iou_gate_at_peak():
    far = (400.0, 100.0, 500.0, 300.0)
    r = evaluate([pred(box=far)], [gt()])
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)
    shifted = (130.0, 100.0, 230.0, 300.0)   # IoU = 70/130 = 0.54 >= 0.30
    assert evaluate([pred(box=shifted)], [gt()]).tp == 1
    barely = (180.0, 100.0, 280.0, 300.0)    # IoU = 20/180 = 0.11 < 0.30
    assert evaluate([pred(box=barely)], [gt()]).tp == 0


def test_fragmentation_is_one_tp_one_fp():
    r = evaluate([pred(10, 18, 14), pred(22, 30, 26)], [gt()])
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)


def test_merging_two_people_is_one_tp_one_fn():
    g1 = gt(box=(100, 100, 200, 300), pid="a")
    g2 = gt(box=(500, 100, 600, 300), pid="b")
    r = evaluate([pred(box=(100, 100, 200, 300))], [g1, g2])
    assert (r.tp, r.fp, r.fn) == (1, 0, 1)


def test_one_to_one_matching_picks_best_overlap():
    g = gt(10, 30)
    r = evaluate([pred(10, 30), pred(12, 28)], [g])
    assert (r.tp, r.fp) == (1, 1) and r.matches == [(0, 0)]


def test_ignore_suppresses_false_positive():
    ign = GT("c1", "x", 50, 80, "IGNORE")
    r = evaluate([pred(55, 75, 65, box=(900, 900, 950, 1000))], [gt(), ign])
    assert r.fp == 0 and r.ignored_preds == 1 and r.fn == 1


def test_ignore_does_not_suppress_if_less_than_half_inside():
    ign = GT("c1", "x", 50, 60, "IGNORE")
    r = evaluate([pred(55, 85, 70, box=(900, 900, 950, 1000))], [gt(), ign])   # 5/30 внутри IGNORE
    assert r.fp == 1 and r.ignored_preds == 0


def test_ignore_does_not_suppress_if_touching_positive():
    ign = GT("c1", "x", 0, 100, "IGNORE")
    r = evaluate([pred(12, 28, 20)], [gt(), ign])
    assert r.tp == 1 and r.ignored_preds == 0


def test_fp_in_negative_clip_and_fp_per_hour():
    neg = GT("neg1", "n", 0, 60, "NEGATIVE", box=BOX)
    r = evaluate([Pred("neg1", 10, 20, 0.7, 15, BOX), Pred("neg1", 30, 40, 0.6, 35, BOX)], [gt(), neg], negative_hours=0.5)
    assert r.fp == 2 and r.fp_per_hour == pytest.approx(4.0)


def test_clips_do_not_mix():
    r = evaluate([pred(clip="other")], [gt()])
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)


def test_median_latency():
    gts = [gt(10, 30, pid="a", box=(0, 0, 100, 200)), gt(50, 70, pid="b", box=(300, 0, 400, 200))]
    preds = [pred(14, 30, 20, box=(0, 0, 100, 200)), pred(53, 70, 60, box=(300, 0, 400, 200))]
    r = evaluate(preds, gts)
    assert r.tp == 2 and r.median_latency == pytest.approx(3.5)


def test_threshold_curve_monotone_recall():
    preds = [pred(conf=0.9), Pred("c1", 100, 110, 0.4, 105, BOX)]
    rows = threshold_curve(preds, [gt()], [0.3, 0.5, 0.95])
    assert [r["tp"] for r in rows] == [1, 1, 0]
    assert [r["fp"] for r in rows] == [1, 0, 0]
