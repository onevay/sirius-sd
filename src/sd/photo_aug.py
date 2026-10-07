"""Аугментации фото «под камеру наблюдения»: уменьшение объекта в кадре (сигарета мелкая), сжатие JPEG, размытие, шум.

Зачем: обычная фото-модель обучена на крупных планах с хорошо видимой сигаретой (AUC 0.97–0.99 на фото), а на кропах рта из записей камер падает почти до случайности
(`docs/ANALYSIS.md`, раздел 9). Эти аугментации приближают фото к кропам видео по масштабу и качеству; помогает ли это — проверяется на размеченных циклах
(`sd photos eval-video`), а не предполагается.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import cv2
import numpy as np


def zoom_out(img: np.ndarray, rng: np.random.Generator, lo: float = 0.4, hi: float = 0.8) -> np.ndarray:
    """Объект становится в `s` раз меньше: исходный кадр уменьшается и вклеивается на размытый фон из него же (без зеркальных двойников по краям)."""
    h, w = img.shape[:2]
    s = float(rng.uniform(lo, hi))
    bg = cv2.GaussianBlur(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), max(h, w) / 30)
    fh, fw = max(8, int(h * s)), max(8, int(w * s))
    fg = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
    y0 = int(np.clip((h - fh) / 2 + rng.uniform(-0.15, 0.15) * h * (1 - s), 0, h - fh))
    x0 = int(np.clip((w - fw) / 2 + rng.uniform(-0.15, 0.15) * w * (1 - s), 0, w - fw))
    out = bg.copy()
    out[y0:y0 + fh, x0:x0 + fw] = fg
    return out


def degrade(img: np.ndarray, rng: np.random.Generator, jpeg=(25, 70), blur=(0.0, 1.2), noise=(0.0, 6.0)) -> np.ndarray:
    """Сжатие JPEG, гауссово размытие, шум — типичные потери видеопотока камеры."""
    out = img
    sig = float(rng.uniform(*blur))
    if sig > 0.15:
        out = cv2.GaussianBlur(out, (0, 0), sig)
    n = float(rng.uniform(*noise))
    if n > 0.5:
        out = np.clip(out.astype(np.float32) + rng.normal(0, n, out.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(jpeg[0], jpeg[1] + 1))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out


def cctv_like(img: np.ndarray, rng: np.random.Generator, zoom=(0.4, 0.8)) -> np.ndarray:
    return degrade(zoom_out(img, rng, *zoom), rng)


def aug_key(key: str, i: int) -> str:
    return f"{key}|aug{i}"
