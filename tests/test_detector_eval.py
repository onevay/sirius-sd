from pathlib import Path

import cv2
import numpy as np
import pytest

from sd import detector_eval as DE


def _dataset(tmp_path: Path):
    root = tmp_path / "ds"
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    ok, buf = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
    for i in range(6):
        (root / "valid" / "images" / f"i{i}.jpg").write_bytes(buf.tobytes())
    (root / "valid" / "labels" / "i0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "valid" / "labels" / "i1.txt").write_text("1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    (root / "valid" / "labels" / "i2.txt").write_text("", encoding="utf-8")            # пустой файл = негатив
    (root / "data.yaml").write_text("names: [cigarette, vape]\n", encoding="utf-8")
    return root


def test_split_dirs_accepts_roboflow_layout_and_reports_missing(tmp_path):
    root = _dataset(tmp_path)
    im, lb = DE.split_dirs(root / "data.yaml", "val")
    assert im == root / "valid" / "images" and lb == root / "valid" / "labels"
    with pytest.raises(FileNotFoundError):
        DE.split_dirs(root / "data.yaml", "test")


def test_image_labels_counts_boxes_per_class_filter(tmp_path):
    root = _dataset(tmp_path)
    im, lb = DE.split_dirs(root / "data.yaml", "val")
    all_cls = DE.image_labels(im, lb)
    assert list(all_cls.positive) == [1, 1, 0, 0, 0, 0] and list(all_cls.n_boxes) == [1, 2, 0, 0, 0, 0]
    only1 = DE.image_labels(im, lb, {1})
    assert list(only1.positive) == [0, 1, 0, 0, 0, 0]


def test_image_level_auc_and_points(tmp_path):
    root = _dataset(tmp_path)
    df = DE.image_labels(*DE.split_dirs(root / "data.yaml", "val"))
    scores = {"i0": 0.9, "i1": 0.6, "i2": 0.1, "i3": 0.0, "i4": 0.3, "i5": 0.0}
    res = DE.image_level(df, lambda paths: [scores[Path(p).stem] for p in paths], thresholds=(0.25, 0.5))
    assert res["auc"] == 1.0 and res["positives"] == 2 and res["n"] == 6
    assert res["points"][1] == dict(conf=0.5, recall=1.0, fpr=0.0)
    assert res["points"][0]["fpr"] == 0.25                               # один из четырёх негативов (i4) выше 0.25
