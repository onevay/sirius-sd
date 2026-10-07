import cv2
import numpy as np
import pandas as pd

from sd import photo_video as PV


def test_cycle_dir_is_filesystem_safe_and_unique():
    a = PV.cycle_dir("курение__smoking_Jar (1)", 3, 12.3456)
    b = PV.cycle_dir("курение__smoking_Jar (1)", 3, 12.4)
    assert a != b and " " not in a.name and "(" not in a.name and a.name.endswith("12.346")
    assert PV.cycle_dir("v", 1, 12.3456) == PV.cycle_dir("v", 1, 12.3461)   # ключи start в таблицах округлены до 1 мс


def test_frame_table_and_aggregate(tmp_path):
    rows = []
    for k, start in enumerate((1.0, 5.0)):
        d = tmp_path / f"c{k}"
        d.mkdir()
        for j in range(3):
            cv2.imwrite(str(d / f"{j}.jpg"), np.zeros((8, 8, 3), np.uint8))
        rows.append(dict(video="v", tid=1, start=start, n=3, dir=str(d)))
    ft = PV.frame_table(pd.DataFrame(rows))
    assert len(ft) == 6 and set(ft.start) == {1.0, 5.0}
    agg = PV.aggregate(ft, {"p": np.array([0.1, 0.9, 0.5, 0.2, np.nan, 0.4])})
    a = agg[agg.start == 1.0].iloc[0]
    assert a.photo_p_max == 0.9 and abs(a.photo_p_mean - 0.5) < 1e-9 and abs(a.photo_p_top2 - 0.7) < 1e-9 and a.photo_n_frames == 3
    b = agg[agg.start == 5.0].iloc[0]
    assert b.photo_n_frames == 2 and abs(b.photo_p_mean - 0.3) < 1e-9
