import pandas as pd

from sd import gt as GT
from sd import gt_pool as GP


def _puffs():
    return pd.DataFrame([
        dict(video="a", tid=1, start=10.0, end=12.0, peak_t=11.0), dict(video="a", tid=1, start=20.0, end=22.0, peak_t=21.0),      # зазор 8 с → один эпизод
        dict(video="a", tid=1, start=60.0, end=62.0, peak_t=61.0),                                                                # далеко → одиночная
        dict(video="a", tid=2, start=11.0, end=13.0, peak_t=12.0),                                                                # другой человек → отдельно
        dict(video="b", tid=1, start=5.0, end=7.0, peak_t=6.0), dict(video="b", tid=1, start=8.0, end=9.0, peak_t=8.5),
    ])


def test_episodes_merge_by_gap_and_single_puff_is_ignore():
    e = GP.puff_episodes(_puffs()).sort_values(["video", "tid", "start"]).reset_index(drop=True)
    assert list(e.label) == ["POSITIVE", "IGNORE", "IGNORE", "POSITIVE"] and list(e.n) == [2, 1, 1, 2]
    assert abs(e.start[0] - 9.5) < 1e-9 and abs(e.end[0] - 22.5) < 1e-9


def test_write_gt_replaces_own_rows_only_and_marks_clean_clips(tmp_path):
    p = tmp_path / "gt.csv"
    GT.add("a", 1, 2, "NEGATIVE", labeler="user", path=p)                    # чужая строка не должна пропасть
    e = GP.puff_episodes(_puffs())
    box = lambda v, t, s: (0, 0, 10, 20)                                       # noqa: E731
    r = GP.write_gt(e, {"a": 100.0, "b": 50.0, "c": 30.0}, box, reviewed_clips=["c"], path=p)
    assert r == dict(positive=2, ignore=2, clean_clips=1)
    df = GT.load(p)
    assert (df.labeler == "user").sum() == 1 and set(df[df.clip_id == "c"].label) == {"NEGATIVE"}
    GP.write_gt(e, {"a": 100.0, "b": 50.0, "c": 30.0}, box, reviewed_clips=["c"], path=p)      # повтор не множит строки
    assert len(GT.load(p)) == len(df)


def test_rebuild_gt_policies_and_idempotent(tmp_path):
    p = tmp_path / "gt.csv"
    pool = pd.DataFrame([
        dict(video="курение__a", tid=1, start=10.0, end=12.0, peak_t=11.0, label="smoke"), dict(video="курение__a", tid=1, start=20.0, end=22.0, peak_t=21.0, label="smoke"),
        dict(video="курение__a", tid=1, start=80.0, end=82.0, peak_t=81.0, label="unsure"),
        dict(video="курение__b", tid=1, start=5.0, end=7.0, peak_t=6.0, label="unsure"),                       # папка «курение», затяжек нет → клип вне оценки
        dict(video="лжекурение__c", tid=1, start=5.0, end=7.0, peak_t=6.0, label="touch_face"),               # чистый клип
        dict(video="лжекурение__d", tid=1, start=5.0, end=7.0, peak_t=6.0, label="smoke"), dict(video="лжекурение__d", tid=1, start=9.0, end=11.0, peak_t=10.0, label="smoke"),
    ])
    dur = {"курение__a": 100.0, "курение__b": 30.0, "лжекурение__c": 50.0, "лжекурение__d": 40.0}
    box = lambda v, t, s: (0, 0, 10, 20)                                                                       # noqa: E731
    r = GP.rebuild_gt(pool, dur, box, path=p)
    assert r == dict(positive=2, ignore_episodes=0, ignore_unsure=1, clean_clips=1, neutral_clips=1)
    df = GT.load(p)
    by = df.groupby("clip_id").label.apply(lambda s: sorted(set(s))).to_dict()
    assert by == {"курение__a": ["IGNORE", "POSITIVE"], "курение__b": ["IGNORE"], "лжекурение__c": ["NEGATIVE"], "лжекурение__d": ["POSITIVE"]}
    GP.rebuild_gt(pool, dur, box, path=p)
    assert len(GT.load(p)) == len(df) and GT.validate(GT.load(p)) == []
