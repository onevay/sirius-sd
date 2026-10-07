"""Фото-датасеты курения («курит / не курит»): список снимков, метки по именам папок/файлов, аудит данных.

Зачем аудит: у открытых наборов фото часто «пробивают» шорткаты — класс угадывается по размеру файла, яркости,
пропорциям кадра, а не по содержимому; а одни и те же снимки попадают и в train, и в test. Оба случая завышают метрики,
поэтому перед обучением считаем (1) группы почти-дубликатов (dHash) — по ним потом делим на фолды;
(2) AUC классификатора, которому разрешено смотреть ТОЛЬКО на метаданные (размер, яркость, резкость).
Если он заметно выше 0.5 — набор нельзя использовать «как есть» без решения этой проблемы.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_NEG = re.compile(r"(not[_\- ]?smok|non[_\- ]?smok|no[_\- ]?smok|negative|\bneg\b|нет|не[_\- ]?кур)", re.I)
_POS = re.compile(r"(smok|cigar|vape|tobacco|кур)", re.I)
_SPLITS = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
           "test": "test", "testing": "test"}


def label_from_path(p: Path, root: Path | None = None) -> int | None:
    """1 = курит, 0 = не курит, None = не удалось определить по имени. Сначала отрицание: «notsmoking» содержит «smoking»."""
    parts = list(p.relative_to(root).parts) if root and p.is_relative_to(root) else list(p.parts[-3:])
    parts[-1] = Path(parts[-1]).stem
    text = "/".join(parts)
    if _NEG.search(text):
        return 0
    if _POS.search(text):
        return 1
    return None


def split_from_path(p: Path, root: Path) -> str:
    for part in p.relative_to(root).parts[:-1]:
        s = _SPLITS.get(part.lower())
        if s:
            return s
    return "all"


def scan_photos(root: Path, source: str | None = None, with_size: bool = True) -> pd.DataFrame:
    """Таблица path, source, split, label, cls, w, h, bytes.

    Метка: `labels.csv` в корне набора (file,label,cls,split — пишет photo_fetch), иначе по именам папок/файлов.
    Файлы без определимой метки остаются с label=NaN (аудит их покажет).
    """
    root = Path(root)
    given: dict[str, dict] = {}
    if (root / "labels.csv").exists():
        given = {r["file"].replace("\\", "/"): r for r in pd.read_csv(root / "labels.csv", dtype=str).to_dict("records")}
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        h, w = (_image_size(p) or (0, 0)) if with_size else (0, 0)
        g = given.get(p.relative_to(root).as_posix())
        if g is not None:
            lab, split, cls = int(g["label"]), g.get("split") or "all", g.get("cls") or ""
        else:
            lab, split, cls = label_from_path(p, root), split_from_path(p, root), ""
        rows.append(dict(path=str(p), source=source or root.name, split=split, label=np.nan if lab is None else lab,
                         cls=cls, w=w, h=h, bytes=p.stat().st_size))
    return pd.DataFrame(rows)


def _image_size(p: Path) -> tuple[int, int] | None:
    """(h, w) из заголовка файла без полного декодирования; None — если файл не читается."""
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size[1], im.size[0]
    except Exception:
        return None


def dhash(path: str | Path, size: int = 8) -> int:
    """64-битный dHash: устойчив к перекодированию и небольшому масштабу."""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return 0
    im = cv2.resize(im, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (im[:, 1:] > im[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Матрица расстояний Хэмминга между uint64-хэшами (по 8 байтам через таблицу битов)."""
    b = a if b is None else b
    x = a[:, None] ^ b[None, :]
    out = np.zeros(x.shape, dtype=np.uint8)
    for k in range(8):
        out += _POP[((x >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)]
    return out


def duplicate_groups(hashes: list[int], max_dist: int = 4) -> np.ndarray:
    """id группы почти-дубликатов для каждого снимка (union-find по парам с расстоянием ≤ max_dist)."""
    h = np.array(hashes, dtype=np.uint64)
    n = len(h)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for start in range(0, n, 512):  # блоками — чтобы матрица N×N не разрасталась
        d = hamming_matrix(h[start:start + 512], h)
        ii, jj = np.nonzero(d <= max_dist)
        for i, j in zip(ii + start, jj):
            if i < j:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    _, inv = np.unique(roots, return_inverse=True)
    return inv


def image_stats(path: str | Path) -> dict:
    """Дешёвые «метаданные-признаки», по которым нельзя разделять классы: яркость, насыщенность, резкость, доля тёмных/светлых."""
    im = cv2.imread(str(path))
    if im is None:
        return dict(bright=np.nan, sat=np.nan, sharp=np.nan, dark=np.nan, light=np.nan)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return dict(bright=float(g.mean()), sat=float(hsv[..., 1].mean()), sharp=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                dark=float((g < 40).mean()), light=float((g > 215).mean()))


def audit(df: pd.DataFrame, max_dist: int = 4, with_shortcut: bool = True, progress=None) -> tuple[pd.DataFrame, dict]:
    """Добавляет dhash, dup_group, метаданные-признаки; возвращает (таблица, отчёт-словарь)."""
    df = df.copy().reset_index(drop=True)
    hs, stats = [], []
    for i, p in enumerate(df.path):
        hs.append(dhash(p))
        stats.append(image_stats(p))
        if progress and i % 200 == 0:
            progress(i, len(df))
    df["dhash"] = [f"{h:016x}" for h in hs]
    df = pd.concat([df, pd.DataFrame(stats)], axis=1)
    df["dup_group"] = duplicate_groups(hs, max_dist)
    sizes = df.groupby("dup_group").size()
    rep: dict = dict(n=len(df), n_labeled=int(df.label.notna().sum()), classes=df.label.value_counts(dropna=False).to_dict(),
                     dup_groups=int((sizes > 1).sum()), dup_images=int(sizes[sizes > 1].sum()))
    # дубликаты между сплитами — настоящая утечка
    cross = 0
    for _, g in df.groupby("dup_group"):
        if len(g) > 1 and g.split.nunique() > 1:
            cross += 1
    rep["dup_groups_across_splits"] = cross
    # дубликаты с разными метками — ошибка разметки
    rep["dup_groups_label_conflict"] = int(sum(1 for _, g in df.groupby("dup_group") if len(g) > 1 and g.label.nunique() > 1))
    if with_shortcut and df.label.nunique() == 2:
        rep["shortcut"] = shortcut_auc(df)
    return df, rep


def shortcut_auc(df: pd.DataFrame, cols=("w", "h", "bytes", "bright", "sat", "sharp", "dark", "light")) -> dict:
    """AUC классификатора, который видит ТОЛЬКО метаданные (grouped CV по дублям). ≈0.5 — норма; ≫0.5 — шорткат в данных."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    d = df[df.label.notna()].copy()
    d["aspect"] = d.w / d.h.clip(lower=1)
    d["bpp"] = d.bytes / (d.w * d.h).clip(lower=1)
    feats = list(cols) + ["aspect", "bpp"]
    X, y, g = d[feats].fillna(0).to_numpy(float), d.label.to_numpy(int), d.dup_group.to_numpy()
    oof = np.zeros(len(d))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y, g):
        m = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    per = {}
    for f in feats:
        v = d[f].to_numpy(float)
        a = roc_auc_score(y, np.nan_to_num(v))
        per[f] = round(max(a, 1 - a), 3)
    return dict(auc_meta_only=round(float(roc_auc_score(y, oof)), 3), per_feature_auc=per)


def montage(df: pd.DataFrame, n: int = 24, cols: int = 6, size: int = 200, seed: int = 0, caption: str = "cls") -> np.ndarray:
    """Сетка из n случайных снимков таблицы (для проверки разметки глазами); подпись — значение колонки `caption` (по умолчанию исходный класс)."""
    d = df.sample(min(n, len(df)), random_state=seed)
    tiles = []
    for r in d.itertuples():
        im = cv2.imread(r.path)
        if im is None:
            continue
        h, w = im.shape[:2]
        k = size / max(h, w)
        im = cv2.resize(im, (max(1, int(w * k)), max(1, int(h * k))))
        tile = np.zeros((size, size, 3), np.uint8)
        tile[:im.shape[0], :im.shape[1]] = im
        cap = str(getattr(r, caption, "") or ("курит" if getattr(r, "label", 0) == 1 else "нет"))[:26].encode("ascii", "replace").decode()
        cv2.putText(tile, f"{int(getattr(r, 'label', -1)) if getattr(r, 'label', None) == getattr(r, 'label', None) else -1}:{cap}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]) if tiles else np.zeros((size, size, 3), np.uint8)
