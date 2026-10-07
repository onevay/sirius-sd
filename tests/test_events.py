import pytest

from sd.cycles import Cycle
from sd.events import ScoredCycle, build_events, events_to_frame, heuristic_cycle_score


def sc(start, hold=2.0, tid=1, score=0.8, obj=False, end=None):
    end = start + hold + 2.0 if end is None else end
    return ScoredCycle(Cycle(tid, start, start + 1.0, start + 1.0 + hold, end, hold, 1, 0.2), score=score, has_object=obj)


def test_one_cycle_without_object_is_not_an_event(cfg):
    assert build_events([sc(10)], cfg["events"]) == []


def test_one_cycle_with_object_is_event(cfg):
    ev = build_events([sc(10, obj=True)], cfg["events"])
    assert len(ev) == 1 and ev[0].rule == "cycle+object"


def test_two_cycles_in_window_is_event_with_proper_boundaries(cfg):
    a, b = sc(10), sc(20)   # b.end = 24 -> окно 14 с <= 20
    ev = build_events([b, a], cfg["events"])   # порядок на входе не важен
    assert len(ev) == 1
    e = ev[0]
    assert e.rule == "two_cycles" and e.n_cycles == 2
    assert e.start == a.cycle.start and e.end == b.cycle.end
    assert e.peak in (a.cycle.peak_t, b.cycle.peak_t)


def test_two_cycles_outside_20s_window_is_not_event_contain_mode(cfg):
    a, b = sc(10), sc(31)    # b.end - a.start = 25 > 20, но разрыв end->start = 31-14=17 > 15 -> разные группы
    assert build_events([a, b], cfg["events"]) == []
    a, b = sc(10), sc(26)    # разрыв end->start = 26-14 = 12 <= 15 (склеены), но b.end-a.start = 18 <= 20 -> событие
    assert len(build_events([a, b], cfg["events"])) == 1
    a, b = sc(10, hold=3.0, end=15.0), sc(27, hold=3.0, end=33.0)  # разрыв 12 <= 15; b.end - a.start = 23 > 20 -> нет
    assert build_events([a, b], cfg["events"]) == []


def test_window_mode_start_gap_is_more_permissive(cfg):
    e = dict(cfg["events"], window_mode="start_gap")
    a, b = sc(10, hold=3.0, end=15.0), sc(27, hold=3.0, end=33.0)   # старт–старт = 17 <= 20
    assert len(build_events([a, b], e)) == 1


def test_merge_gap_15s_keeps_single_event_no_fragmentation(cfg):
    cycles = [sc(0), sc(10), sc(22)]    # разрывы 6 и 6 с -> один эпизод
    ev = build_events(cycles, cfg["events"])
    assert len(ev) == 1 and ev[0].n_cycles == 3


def test_gap_over_15s_splits_into_two_events(cfg):
    cycles = [sc(0), sc(8), sc(40), sc(48)]  # серия 1 и серия 2 разделены паузой > 15 с
    ev = build_events(cycles, cfg["events"])
    assert len(ev) == 2


def test_people_are_never_merged(cfg):
    with pytest.raises(ValueError):
        build_events([sc(0, tid=1), sc(8, tid=2)], cfg["events"])


def test_low_score_cycles_are_dropped(cfg):
    assert build_events([sc(0, score=0.3), sc(8, score=0.3)], cfg["events"]) == []


def test_confidence_is_monotone_in_cycles_and_object(cfg):
    two = build_events([sc(0), sc(8)], cfg["events"])[0].confidence
    three = build_events([sc(0), sc(8), sc(16)], cfg["events"])[0].confidence
    with_obj = build_events([sc(0, obj=True), sc(8)], cfg["events"])[0].confidence
    assert three > two and with_obj > two


def test_events_to_frame_format(cfg):
    ev = build_events([sc(0), sc(8)], cfg["events"])
    df = events_to_frame(ev, "cam_x", "clip_001", boxes={1: lambda t: (10, 20, 110, 220)})
    assert list(df.columns)[:3] == ["camera_id", "clip_id", "event_id"]
    assert df.iloc[0].event_id == "cam_x_clip_001_0001" and df.iloc[0].label == "smoking_like"
    assert (df.iloc[0].x1, df.iloc[0].y2) == (10, 220)


def test_heuristic_cycle_score_plateau():
    cyc = lambda h: Cycle(1, 0, 1, 1 + h, 2 + h, h, 1, 0.2)
    assert heuristic_cycle_score(cyc(2.0)) == 1.0
    assert heuristic_cycle_score(cyc(0.5)) == pytest.approx(0.5)
    assert heuristic_cycle_score(cyc(5.0)) == pytest.approx(0.5)
    assert heuristic_cycle_score(cyc(2.0), quality=0.4) == pytest.approx(0.4)
