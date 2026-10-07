"""Проверка внешних датасетов, скачанных вручную (Roboflow / Kaggle / Mendeley / HMDB51 / Kinetics) в `data_external/`.

Ожидаемая раскладка (подробности — docs/DATA_AND_WEIGHTS.md):
  data_external/<источник>/<имя_датасета>/data.yaml            — YOLO-датасет (Roboflow: экспорт «YOLOv8»), рядом train|valid|test/{images,labels}
  data_external/<источник>/<имя_датасета>/LICENSE.txt           — лицензия (в README кейса лицензию каждого датасета записываем)
  data_external/<источник>/<имя_датасета>/**/*.mp4|avi|mkv      — видео-датасеты (HMDB51/Kinetics): считаем клипы
  data_external/<источник>/<имя_датасета>/**/*.jpg|png          — классификационные картинки (Kaggle «smoker detection»)
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import yaml

from .paths import EXTERNAL, VIDEO_EXT

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_external(root: Path = EXTERNAL) -> list[dict]:
    out = []
    if not root.exists():
        return out
    for src in sorted(p for p in root.iterdir() if p.is_dir()):
        for ds in sorted(p for p in src.iterdir() if p.is_dir()):
            files = [f for f in ds.rglob("*") if f.is_file()]
            imgs = [f for f in files if f.suffix.lower() in IMG_EXT]
            vids = [f for f in files if f.suffix.lower() in VIDEO_EXT]
            row = dict(source=src.name, name=ds.name, images=len(imgs), videos=len(vids), size_mb=round(sum(f.stat().st_size for f in files) / 1e6, 1),
                       yolo=False, classes=None, splits={}, license=None, issues=[])
            y = ds / "data.yaml"
            if y.exists():
                try:
                    d = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
                    names = d.get("names")
                    row["yolo"], row["classes"] = True, (list(names.values()) if isinstance(names, dict) else names)
                    for sp in ("train", "valid", "val", "test"):
                        im, lb = ds / sp / "images", ds / sp / "labels"
                        if im.exists():
                            n_i = sum(1 for f in im.iterdir() if f.suffix.lower() in IMG_EXT)
                            n_l = sum(1 for _ in lb.glob("*.txt")) if lb.exists() else 0
                            row["splits"][sp] = f"{n_i} img / {n_l} lbl"
                            if n_l == 0:
                                row["issues"].append(f"{sp}: нет разметки")
                    lic = (d.get("roboflow") or {}).get("license")
                    if lic:
                        row["license"] = lic
                except Exception as e:
                    row["issues"].append(f"data.yaml не читается: {e}")
            for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "README.roboflow.txt", "README.dataset.txt"):
                if (ds / name).exists():
                    txt = (ds / name).read_text(encoding="utf-8", errors="ignore")
                    row["license"] = row["license"] or next((ln.strip() for ln in txt.splitlines() if "license" in ln.lower()), name)
            if not row["license"]:
                row["issues"].append("нет лицензии: добавьте LICENSE.txt (требование кейса — записать лицензию каждого датасета)")
            out.append(row)
    return out
