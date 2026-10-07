import numpy as np
import pandas as pd

from sd import start_eval as SE


def _cyc():
    def row(gen, start, end, peak, tid=1, video="v", run="r"):
        return dict(video=video, run=run, gen=gen, tid=tid, start=start, mouth_in=start + 0.5, mouth_out=end - 0.5, end=end, peak_t=peak, hold=end - start - 1, hand=0, d_min=0.3)

    return pd.DataFrame([
        row("main", 10.0, 13.0, 11.5), row("guide", 10.2, 12.8, 11.5), row("lenient", 9.8, 13.2, 11.5),    # один и тот же жест
        row("lenient", 20.0, 22.0, 21.0),                                                                  # найден только мягким порогом
        row("main", 30.0, 33.0, 31.5, tid=2), row("main", 30.0, 33.0, 31.5, tid=1, video="w"),            # другой трек / другое видео — не склеиваются
    ])


def test_pool_merges_generators_per_track_and_keeps_main_as_reference():
    pool = SE.build_pool(_cyc())
    assert len(pool) == 4
    a = pool[(pool.video == "v") & (pool.tid == 1) & (pool.peak_t < 15)].iloc[0]
    assert a.found_by == "main,guide,lenient" and a.ref_gen == "main" and a.n_cycles == 3 and a.start == 10.0
    b = pool[(pool.video == "v") & (pool.tid == 1) & (pool.peak_t > 15)].iloc[0]
    assert b.found_by == "lenient"
    assert set(pool.groupby(["video", "tid"]).size().index) == {("v", 1), ("v", 2), ("w", 1)}


def test_extra_trigger_candidates_join_the_pool():
    extra = pd.DataFrame([dict(video="v", tid=1, gen="photo", start=10.1, end=12.9, peak_t=11.4), dict(video="v", tid=1, gen="photo", start=40.0, end=41.0, peak_t=40.5)])
    pool = SE.build_pool(_cyc(), extra=extra)
    assert "photo" in pool[(pool.video == "v") & (pool.peak_t < 15) & (pool.tid == 1)].iloc[0].found_by.split(",")
    assert (pool.found_by == "photo").sum() == 1                                     # чисто «спасательный» кандидат


def test_generator_table_recall_precision():
    pool = SE.build_pool(_cyc())
    gt = pd.DataFrame([dict(video="v", tid=1, peak_t=11.5, label="smoke", start_quality="ok", note=""),
                       dict(video="v", tid=1, peak_t=21.0, label="drink", start_quality="late", note=""),
                       dict(video="v", tid=2, peak_t=31.5, label="no_gesture", start_quality=None, note=""),
                       dict(video="w", tid=1, peak_t=31.5, label="touch_face", start_quality="ok", note="")])
    p = SE.attach_gt(pool, gt)
    assert p.label.notna().all()
    t = SE.generator_table(p, ["main", "lenient"], n_boot=60).set_index("generator")
    # настоящих жестов 3 (smoke A, drink B, touch_face W): main находит A и W, lenient — A и B (W есть только в другом видео, где lenient нет)
    assert t.loc["main", "recall_gesture"] == 2 / 3 and t.loc["lenient", "recall_gesture"] == 2 / 3
    assert t.loc["main", "precision_gesture"] == 2 / 3                               # 3 кандидата main: smoke, no_gesture, touch_face
    assert t.loc["lenient", "cand"] == 2 and t.loc["lenient", "precision_gesture"] == 1.0
    assert t.loc["main", "recall_puff"] == 1.0
    sq = SE.start_quality_table(p, "main").set_index("start_quality")
    assert sq.loc["ok", "n"] == 2 and abs(sq.share.sum() - 1) < 1e-9


def test_wilson_interval_sane():
    lo, hi = SE.wilson(8, 10)
    assert 0.45 < lo < 0.5 and 0.9 < hi < 0.97
    assert np.isnan(SE.wilson(0, 0)[0])
    assert SE.wilson(0, 5)[0] == 0.0 and SE.wilson(5, 5)[1] == 1.0
