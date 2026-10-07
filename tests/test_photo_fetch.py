import zipfile
from pathlib import Path

import cv2
import numpy as np

from sd import photo_data as pdta
from sd import photo_fetch as pf


def _zip_with(tmp_path: Path, counts: dict[str, int]) -> Path:
    zp = tmp_path / "Stanford40.zip"
    ok, buf = cv2.imencode(".jpg", np.full((32, 32, 3), 127, np.uint8))
    assert ok
    with zipfile.ZipFile(zp, "w") as z:
        for cls, n in counts.items():
            for i in range(1, n + 1):
                z.writestr(f"JPEGImages/{cls}_{i:03d}.jpg", buf.tobytes())
        z.writestr("ImageSplits/train.txt", "smoking_001.jpg\ndrinking_001.jpg\n")
        z.writestr("ImageSplits/test.txt", "smoking_002.jpg\n")
        z.writestr("__MACOSX/JPEGImages/._smoking_001.jpg", b"junk")
    return zp


def test_extract_stanford40_picks_smoking_hard_and_few_easy(tmp_path):
    zp = _zip_with(tmp_path, {"smoking": 6, "drinking": 5, "phoning": 4, "climbing": 20, "jumping": 20})
    dest = tmp_path / "out"
    rows = pf.extract_stanford40(zp, dest, hard=("drinking", "phoning"), easy_per_class=3)
    by = {}
    for r in rows:
        by.setdefault(r["cls"], []).append(r)
    assert len(by["smoking"]) == 6 and len(by["drinking"]) == 5 and len(by["phoning"]) == 4
    assert len(by["climbing"]) == 3 and len(by["jumping"]) == 3          # лёгкие негативы — только по 3
    assert all(r["label"] == 1 for r in by["smoking"]) and all(r["label"] == 0 for r in rows if r["cls"] != "smoking")
    assert next(r for r in by["smoking"] if r["file"].endswith("smoking_001.jpg"))["split"] == "train"
    assert next(r for r in by["smoking"] if r["file"].endswith("smoking_002.jpg"))["split"] == "test"
    assert (dest / "smoking" / "smoking_001.jpg").exists()


def test_extract_is_deterministic_and_limited(tmp_path):
    zp = _zip_with(tmp_path, {"smoking": 10, "drinking": 10, "climbing": 30})
    a = pf.extract_stanford40(zp, tmp_path / "a", hard=("drinking",), easy_per_class=5, max_per_class=4)
    b = pf.extract_stanford40(zp, tmp_path / "b", hard=("drinking",), easy_per_class=5, max_per_class=4)
    assert [r["file"] for r in a] == [r["file"] for r in b]
    assert sum(1 for r in a if r["cls"] == "smoking") == 4 and sum(1 for r in a if r["cls"] == "climbing") == 4


def test_extract_yolo_zip_flat_layout_to_ultralytics(tmp_path):
    """CigDet: `train/a.jpg` + `train/a.txt` в одном каталоге → `<split>/images|labels` + data.yaml; лишнее и пути из архива не используются."""
    import yaml

    ok, buf = cv2.imencode(".jpg", np.full((16, 16, 3), 90, np.uint8))
    assert ok
    zp = tmp_path / "cig.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for sp, n in (("train", 3), ("test", 2)):
            for i in range(n):
                z.writestr(f"CigDet_dataset/{sp}/smoking_{sp}{i}.jpg", buf.tobytes())
                z.writestr(f"CigDet_dataset/{sp}/smoking_{sp}{i}.txt", "0 0.5 0.5 0.1 0.1\n")
        z.writestr("CigDet_dataset/test/run.exe", b"MZ")                                  # не картинка и не метка
        z.writestr("CigDet_dataset/train/../../escape.jpg", buf.tobytes())               # обход каталога: берётся только имя файла
        z.writestr("CigDet_dataset/other/loose.jpg", buf.tobytes())                      # вне train/test/val — пропускается
        z.writestr("CigDet_dataset/test/huge.jpg", b"\0" * 2_000_000)                    # крупнее лимита
    dest = tmp_path / "out"
    res = pf.extract_yolo_zip(zp, dest, ["cigarette"], max_file_mb=1.0)
    assert res["splits"]["train"] == dict(images=4, labels=3)                            # + escape.jpg внутри train/images, без метки
    assert res["splits"]["test"] == dict(images=2, labels=2)
    assert not (tmp_path / "escape.jpg").exists() and not (dest / "test" / "images" / "run.exe").exists()
    assert not (dest / "test" / "images" / "huge.jpg").exists() and not (dest / "other").exists()
    cfg = yaml.safe_load((dest / "data.yaml").read_text(encoding="utf-8"))
    assert cfg == {"train": "train/images", "test": "test/images", "val": "test/images", "names": {0: "cigarette"}}
    from sd import detector_eval as DE

    imgs, labs = DE.split_dirs(dest / "data.yaml", "test")
    assert imgs == dest / "test" / "images" and len(DE.image_labels(imgs, labs)) == 2


def test_cigdet_source_is_a_detector_set_not_a_photo_set():
    info = pf.SOURCES["cigdet"]
    assert info["type"] == "yolo_zip" and len(info["sha256"]) == 64 and info["names"] == ["cigarette"]
    assert "CC BY 4.0" in info["license"] and "labels.csv намеренно не создаётся" in info["license"]


def test_scan_photos_reads_labels_csv(tmp_path):
    zp = _zip_with(tmp_path, {"smoking": 3, "drinking": 3})
    dest = tmp_path / "ds"
    rows = pf.extract_stanford40(zp, dest, hard=("drinking",), easy_per_class=0)
    pf.SOURCES["_t"] = dict(license="тест", url="http://example")
    pf._write_meta(dest, "_t", rows)
    df = pdta.scan_photos(dest)
    assert len(df) == 6 and df.label.sum() == 3
    assert set(df.cls) == {"smoking", "drinking"}
    assert (dest / "LICENSE.txt").read_text(encoding="utf-8").startswith("тест")
    del pf.SOURCES["_t"]
