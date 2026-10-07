from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sd import photo_data as PD
from sd import photo_fetch as PF


def _yolo(tmp_path: Path) -> Path:
    root = tmp_path / "rf"
    ok, buf = cv2.imencode(".jpg", np.full((20, 30, 3), 90, np.uint8))
    for split, n in (("train", 3), ("valid", 2), ("test", 1)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            (root / split / "images" / f"{split}{i}.jpg").write_bytes(buf.tobytes())
    (root / "train" / "labels" / "train0.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (root / "train" / "labels" / "train1.txt").write_text("1 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    (root / "valid" / "labels" / "valid0.txt").write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (root / "data.yaml").write_text("names: [cigarette_in_hand, cigarette_near_mouth, pen]\n", encoding="utf-8")
    return root


def test_labels_from_yolo_any_box_and_class_filter(tmp_path):
    root = _yolo(tmp_path)
    rows = PF.labels_from_yolo(root)
    by = {r["file"].split("/")[-1]: r for r in rows}
    assert len(rows) == 6 and by["train0.jpg"]["label"] == 1 and by["train2.jpg"]["label"] == 0 and by["train2.jpg"]["cls"] == "background"
    assert by["train1.jpg"]["cls"] == "cigarette_in_hand+cigarette_near_mouth" and by["valid0.jpg"]["cls"] == "pen" and by["valid0.jpg"]["split"] == "val"
    rows2 = PF.labels_from_yolo(root, {"cigarette_in_hand", "cigarette_near_mouth"})          # «ручка» — не курение: трудный негатив
    by2 = {r["file"].split("/")[-1]: r for r in rows2}
    assert by2["valid0.jpg"]["label"] == 0 and by2["train1.jpg"]["label"] == 1
    df = PD.scan_photos(root)
    assert len(df) == 6 and df.label.sum() == 2 and set(df.split) == {"train", "val", "test"}   # в labels.csv — вариант с фильтром классов (ручка = 0)


def test_montage_shape_and_captions(tmp_path):
    root = _yolo(tmp_path)
    PF.labels_from_yolo(root)
    df = PD.scan_photos(root)
    m = PD.montage(df, n=6, cols=3, size=40)
    assert m.shape == (80, 120, 3) and m.max() > 0
    assert PD.montage(df.iloc[:0], n=3).shape[2] == 3
