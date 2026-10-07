from pathlib import Path

import cv2
import numpy as np
import pytest

from sd import photo_data as pdta


def _img(path: Path, seed: int, size=(96, 96), quality=90, noise=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    im = cv2.resize(base, size, interpolation=cv2.INTER_CUBIC)
    if noise:
        im = np.clip(im.astype(int) + rng.integers(-noise, noise, im.shape), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, quality])


def test_label_from_path_negation_first(tmp_path):
    r = tmp_path
    cases = {"Training/smoking_0001.jpg": 1, "Training/notsmoking_0001.jpg": 0, "train/not_smoking/a.jpg": 0,
             "train/smoking/a.jpg": 1, "x/non_smoking/a.jpg": 0, "other/abc.jpg": None, "cigarette/z.jpg": 1}
    for rel, want in cases.items():
        assert pdta.label_from_path(r / rel, r) == want, rel


def test_split_from_path(tmp_path):
    assert pdta.split_from_path(tmp_path / "Validation" / "a.jpg", tmp_path) == "val"
    assert pdta.split_from_path(tmp_path / "Testing" / "a.jpg", tmp_path) == "test"
    assert pdta.split_from_path(tmp_path / "x" / "a.jpg", tmp_path) == "all"


def test_scan_and_duplicates_across_splits(tmp_path):
    _img(tmp_path / "Training" / "smoking_0001.jpg", 1)
    _img(tmp_path / "Testing" / "smoking_0002.jpg", 1, quality=60)   # тот же снимок, другое сжатие → дубликат
    _img(tmp_path / "Training" / "notsmoking_0001.jpg", 2)
    _img(tmp_path / "Training" / "notsmoking_0002.jpg", 3)
    df = pdta.scan_photos(tmp_path, source="t")
    assert len(df) == 4 and df.label.tolist().count(1) == 2 and set(df.split) == {"train", "test"}
    assert (df.w == 96).all() and (df.h == 96).all()
    out, rep = pdta.audit(df, with_shortcut=False)
    assert rep["dup_groups"] == 1 and rep["dup_images"] == 2 and rep["dup_groups_across_splits"] == 1
    assert rep["dup_groups_label_conflict"] == 0
    assert out.dup_group.nunique() == 3


def test_duplicate_label_conflict_is_reported(tmp_path):
    _img(tmp_path / "Training" / "smoking_0001.jpg", 7)
    _img(tmp_path / "Training" / "notsmoking_0001.jpg", 7, quality=70)
    _, rep = pdta.audit(pdta.scan_photos(tmp_path), with_shortcut=False)
    assert rep["dup_groups_label_conflict"] == 1


def test_shortcut_detected_when_class_is_visible_in_metadata(tmp_path):
    for i in range(40):  # курящие — крупные и шумные, остальные — маленькие: чистый шорткат
        _img(tmp_path / "train" / "smoking" / f"s{i}.jpg", 100 + i, size=(160, 120), noise=40)
        _img(tmp_path / "train" / "not_smoking" / f"n{i}.jpg", 200 + i, size=(64, 64))
    _, rep = pdta.audit(pdta.scan_photos(tmp_path))
    assert rep["shortcut"]["auc_meta_only"] > 0.9


def test_no_shortcut_when_metadata_matches(tmp_path):
    for i in range(40):
        _img(tmp_path / "train" / "smoking" / f"s{i}.jpg", 100 + i)
        _img(tmp_path / "train" / "not_smoking" / f"n{i}.jpg", 200 + i)
    _, rep = pdta.audit(pdta.scan_photos(tmp_path))
    assert rep["shortcut"]["auc_meta_only"] < 0.8


def test_hamming_and_groups():
    h = np.array([0, 1, 3, 2 ** 40], dtype=np.uint64)
    d = pdta.hamming_matrix(h)
    assert d[0, 1] == 1 and d[0, 2] == 2 and d[0, 3] == 1 and d[1, 2] == 1
    # single-linkage: 0–1, 0–3 и 1–2 — на расстоянии 1, поэтому все четыре хэша попадают в одну группу
    assert len(set(pdta.duplicate_groups([int(x) for x in h], max_dist=1))) == 1
    assert len(set(pdta.duplicate_groups([int(x) for x in h], max_dist=0))) == 4
