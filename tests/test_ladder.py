import pandas as pd

from sd import ladder as LD


def test_match_cycles_counts_found_and_extra():
    cyc = pd.DataFrame(dict(peak_t=[10.0, 30.0, 50.5]))
    r = LD.match_cycles(cyc, [10.5, 50.0, 90.0])                 # затяжки на 10.5, 50, 90; цикл на 30 — лишний; 90 — не найдена
    assert r == dict(smoke=3, found=2, extra=1, cycles=3)
    assert LD.match_cycles(cyc.iloc[0:0], [5.0]) == dict(smoke=1, found=0, extra=0, cycles=0)
    assert LD.match_cycles(cyc, []) == dict(smoke=0, found=0, extra=3, cycles=3)


def test_summarize_recall_and_errors():
    df = pd.DataFrame([dict(config="a", clip="x", smoke=4, found=3, extra=1, cycles=4, tracks=1, wrist_conf=0.5, ms_per_frame=100.0, frames=10, error=""),
                       dict(config="a", clip="y", smoke=2, found=2, extra=0, cycles=2, tracks=1, wrist_conf=0.7, ms_per_frame=120.0, frames=10, error=""),
                       dict(config="b", clip="x", error="MemoryError: x")])
    s = LD.summarize(df).set_index("config")
    assert abs(s.loc["a", "recall"] - 5 / 6) < 1e-9 and s.loc["a", "errors"] == 0
    assert s.loc["b", "errors"] == 1 and pd.isna(s.loc["b", "recall"])               # упавшая конфигурация видна в таблице


def test_scaled_copy_keeps_frame_size_with_padding(tmp_path):
    """Дальняя камера: содержимое уменьшено, размер кадра прежний (поля), повторный вызов не пересчитывает."""
    import cv2
    import numpy as np

    from sd.video_io import probe

    src = tmp_path / "a.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(12):
        w.write(np.full((180, 320, 3), 200, np.uint8))
    w.release()
    out = LD.scaled_copy(src, 0.5, tmp_path / "o")
    i = probe(out)
    assert (i.width, i.height) == (320, 180) and out.name.startswith("a_z0500")
    assert LD.scaled_copy(src, 0.5, tmp_path / "o") == out
    cap = cv2.VideoCapture(str(out))
    ok, fr = cap.read()
    cap.release()
    assert ok and fr[5, 5].mean() < 150 and fr[90, 160].mean() > 170          # углы — серые поля, центр — светлое содержимое
