"""Проверка детекторов предмета (сигарета/вейп) на скачанных YOLO-датасетах (Roboflow «YOLOv8», Ultralytics Platform, CigDet): `sd detector-eval`.

Два режима:
  * `box`  — стандартные P / R / mAP50 / mAP50-95 по рамкам (`ultralytics val`); осмысленно, когда классы детектора и датасета совпадают (у одноклассовых — обычно да);
  * `image` — уровень картинки: «есть ли на снимке хоть одна сигарета» (макс. уверенность детектора против наличия разметки) → AUC и точки на ROC. Не зависит от
    названий классов и ближе к тому, как предмет используется в конвейере (признак «предмет виден»).
Веса `.pt` перед загрузкой проходят сканирование (`model_tools.scan_model_path`). Сеть не используется, все вычисления локальные.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    """(каталог картинок, каталог меток) выбранного сплита YOLO-датасета с data.yaml (раскладка Roboflow: <split>/images, <split>/labels)."""
    root = Path(data_yaml).parent
    names = {"val": ["valid", "val"], "test": ["test"], "train": ["train"]}[split]
    for n in names:
        if (root / n / "images").exists():
            return root / n / "images", root / n / "labels"
    raise FileNotFoundError(f"в {root} нет сплита {split} (ожидается {names[0]}/images и {names[0]}/labels)")


def image_labels(images_dir: Path, labels_dir: Path, class_ids: set[int] | None = None) -> pd.DataFrame:
    """Таблица path, n_boxes, positive (1, если в файле меток есть хотя бы одна рамка нужного класса)."""
    rows = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        f = labels_dir / (p.stem + ".txt")
        ids = []
        if f.exists():
            for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = ln.split()
                if len(parts) >= 5:
                    ids.append(int(float(parts[0])))
        ids = [i for i in ids if class_ids is None or i in class_ids]
        rows.append(dict(path=str(p), n_boxes=len(ids), positive=int(len(ids) > 0)))
    return pd.DataFrame(rows)


def image_level(df: pd.DataFrame, predict_max_conf: Callable[[list[str]], list[float]], thresholds=(0.15, 0.25, 0.4, 0.5)) -> dict:
    """AUC уровня картинки и recall/FPR при порогах; `predict_max_conf(paths)` → максимальная уверенность детектора на каждой картинке."""
    from sklearn.metrics import roc_auc_score

    s = np.asarray(predict_max_conf(df.path.tolist()), float)
    y = df.positive.to_numpy(int)
    out = dict(n=int(len(y)), positives=int(y.sum()), auc=float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan"))
    out["points"] = [dict(conf=t, recall=float((s[y == 1] >= t).mean()) if (y == 1).any() else float("nan"), fpr=float((s[y == 0] >= t).mean()) if (y == 0).any() else float("nan"))
                     for t in thresholds]
    out["scores"] = s
    return out


def photo_table(df: pd.DataFrame, predict_max_conf: Callable[[list[str]], list[float]], per_source: int = 300, seed: int = 0) -> dict:
    """Детектор на фото-наборах «курит / не курит» (таблица `photos_audit`: path, source, label, cls): AUC по максимальной уверенности для каждого набора
    и против каждого негативного класса (питьё, телефон…). Выборка — не больше `per_source` снимков на набор, поровну курящих и нет (детерминированно)."""
    from sklearn.metrics import roc_auc_score

    parts = []
    for src, g in df[df.label.notna()].groupby("source"):
        k = min(per_source // 2, int(g.label.sum()), int((1 - g.label).sum()))
        parts.append(pd.concat([g[g.label == 1].sample(k, random_state=seed), g[g.label == 0].sample(k, random_state=seed)]))
    d = pd.concat(parts, ignore_index=True)
    d["score"] = np.asarray(predict_max_conf(d.path.tolist()), float)
    out: dict = dict(n=int(len(d)), per_source={}, per_negative=[])
    for src, g in d.groupby("source"):
        out["per_source"][src] = dict(n=int(len(g)), auc=float(roc_auc_score(g.label.astype(int), g.score)), recall_015=float((g[g.label == 1].score >= 0.15).mean()),
                                      fpr_015=float((g[g.label == 0].score >= 0.15).mean()))
        pos = g[g.label == 1]
        for c, gc in g[g.label == 0].groupby("cls"):
            if len(gc) >= 15:
                y = np.r_[np.ones(len(pos)), np.zeros(len(gc))]
                out["per_negative"].append(dict(source=src, neg_class=c or "—", n=int(len(gc)), auc=float(roc_auc_score(y, np.r_[pos.score, gc.score])), fpr_015=float((gc.score >= 0.15).mean())))
    out["table"] = d
    return out


def yolo_predict_fn(weights: Path, imgsz: int = 640, conf: float = 0.05) -> Callable[[list[str]], list[float]]:
    from ultralytics import YOLO, settings

    from .model_tools import scan_model_path

    sc = scan_model_path(weights)
    if Path(weights).suffix.lower() in (".pt", ".pth") and sc["verdict"] != "ok":
        raise RuntimeError("веса не прошли проверку безопасности: " + "; ".join(sc["notes"]))
    settings.update({"sync": False})
    m = YOLO(str(weights))

    def fn(paths: list[str]) -> list[float]:
        out = []
        for i in range(0, len(paths), 16):
            for r in m.predict(paths[i:i + 16], imgsz=imgsz, conf=conf, verbose=False, device="cpu"):
                out.append(float(r.boxes.conf.max()) if r.boxes is not None and len(r.boxes) else 0.0)
        return out

    return fn


def box_metrics(weights: Path, data_yaml: Path, split: str = "val", imgsz: int = 640) -> dict:
    """P / R / mAP50 / mAP50-95 через `ultralytics val` (CPU)."""
    from ultralytics import YOLO, settings

    from .model_tools import scan_model_path

    sc = scan_model_path(weights)
    if Path(weights).suffix.lower() in (".pt", ".pth") and sc["verdict"] != "ok":
        raise RuntimeError("веса не прошли проверку безопасности: " + "; ".join(sc["notes"]))
    settings.update({"sync": False})
    m = YOLO(str(weights))
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sd_val_") as tmp:      # служебные файлы val не должны оседать в корне проекта (runs/detect/val*)
        r = m.val(data=str(data_yaml), split=split, imgsz=imgsz, device="cpu", verbose=False, plots=False, project=tmp, name="val", exist_ok=True)
    return dict(precision=float(r.box.mp), recall=float(r.box.mr), map50=float(r.box.map50), map50_95=float(r.box.map), detector_classes=dict(m.names))
