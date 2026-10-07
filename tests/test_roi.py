import pandas as pd
import pytest

from sd import roi as R

SQ = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_point_in_polygon_inside_outside_and_border():
    assert R.point_in_polygon(50, 50, SQ) and not R.point_in_polygon(150, 50, SQ) and R.point_in_polygon(0, 50, SQ) and R.point_in_polygon(100, 100, SQ)
    tri = [(0, 0), (10, 0), (0, 10)]
    assert R.point_in_polygon(2, 2, tri) and not R.point_in_polygon(8, 8, tri)


def test_filter_events_uses_feet_point_at_peak():
    ev = pd.DataFrame(dict(x1=[40, 200, float("nan")], y1=[0, 0, 0], x2=[60, 220, 1], y2=[90, 90, 1], start_sec=[0, 1, 2]))
    assert R.filter_events(ev, [SQ]).start_sec.tolist() == [0]          # у второго стопы вне полигона, у третьего нет рамки
    assert len(R.filter_events(ev, None)) == 3 and R.filter_events(ev.iloc[0:0], [SQ]).empty


def test_load_polygon_and_zones(tmp_path):
    f = tmp_path / "roi.json"
    f.write_text('{"polygon": [[0,0],[10,0],[10,10]]}')
    assert len(R.load(f)) == 1
    f.write_text('{"zones": [[[0,0],[10,0],[10,10]], [[20,20],[30,20],[30,30]]]}')
    assert len(R.load(f)) == 2
    f.write_text('{"polygon": [[0,0],[1,1]]}')
    with pytest.raises(ValueError):
        R.load(f)
    f.write_text("{}")
    with pytest.raises(ValueError):
        R.load(f)
