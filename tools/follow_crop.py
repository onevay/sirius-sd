"""Крупный план выбранных людей для видео «с рук» (камера дрожит и поворачивается): окно кадрирования следует за людьми, а не стоит на месте.

Отличие от `zoom_video.py` (один неподвижный вырез на весь ролик): здесь люди ищутся на КАЖДОМ кадре (YOLO с входом 1920), окно держится на группе
людей рядом с заданной точкой, его центр сглаживается, размер подбирается так, чтобы рост человека в результате был ≈ `--fill` высоты кадра.
Запуск:  .venv\\Scripts\\python tools\\follow_crop.py data\\custom\\IMG_1319.mp4 --point 0.69,0.51
         (--point — доли ширины и высоты исходного кадра, где стоят нужные люди; по умолчанию — самый крупный человек первого кадра)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sd.models import local_weights  # noqa: E402
from sd.video_io import FFmpegWriter, iter_frames, probe  # noqa: E402


def track_group(video: Path, point: tuple[float, float] | None, radius: float, imgsz: int) -> tuple[np.ndarray, np.ndarray, float]:
    """По кадрам: центр группы (x, y) и рост человека в ней; NaN — группу не нашли. Группа = люди, чьи центры ближе `radius`·роста к центру группы на прошлом кадре."""
    from ultralytics import YOLO

    m = YOLO(str(local_weights("yolo26n-pose")))
    info = probe(video)
    W, H = int(info.width), int(info.height)
    cur = None if point is None else np.array([point[0] * W, point[1] * H])
    cs, hs = [], []
    for i, fr in enumerate(iter_frames(video)):
        r = m.predict(fr.img, imgsz=imgsz, conf=0.25, classes=[0], verbose=False, device="cpu")[0]
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        if cur is None and len(b):
            k = int(np.argmax(b[:, 3] - b[:, 1]))
            cur = np.array([(b[k, 0] + b[k, 2]) / 2, (b[k, 1] + b[k, 3]) / 2])
        if cur is None or not len(b):
            cs.append([np.nan, np.nan]); hs.append(np.nan)
            continue
        c = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
        h = b[:, 3] - b[:, 1]
        near = np.hypot(*(c - cur).T) < radius * np.median(h)
        if not near.any():
            cs.append([np.nan, np.nan]); hs.append(np.nan)
            continue
        g = b[near]
        cur = np.array([(g[:, 0].min() + g[:, 2].max()) / 2, (g[:, 1].min() + g[:, 3].max()) / 2])
        cs.append(cur.tolist()); hs.append(float(np.median(h[near])))
        if i % 60 == 0:
            print(f"  кадр {i}: людей в группе {int(near.sum())}, рост {hs[-1]:.0f} px", flush=True)
    return np.array(cs, float), np.array(hs, float), info.fps


def smooth(a: np.ndarray, win: int) -> np.ndarray:
    """Заполнение пропусков (ближайшим известным) и скользящее среднее: окно не дёргается вслед за шумом детектора."""
    a = a.copy()
    idx = np.arange(len(a))
    ok = np.isfinite(a)
    if not ok.any():
        raise SystemExit("нужные люди не найдены ни на одном кадре: задайте --point точнее")
    a = np.interp(idx, idx[ok], a[ok])
    k = np.ones(win) / win
    pad = np.r_[np.full(win // 2, a[0]), a, np.full(win - 1 - win // 2, a[-1])]
    return np.convolve(pad, k, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--point", help="x,y в долях исходного кадра, где стоят нужные люди")
    ap.add_argument("--fill", type=float, default=0.55, help="рост человека в долях высоты результата")
    ap.add_argument("--radius", type=float, default=2.5, help="кто входит в группу: расстояние до центра в ростах человека")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--size", default="1920x1080")
    a = ap.parse_args()
    pt = tuple(float(x) for x in a.point.split(",")) if a.point else None
    print("ищу людей на каждом кадре…")
    cs, hs, fps = track_group(a.video, pt, a.radius, a.imgsz)
    cx, cy = smooth(cs[:, 0], 15), smooth(cs[:, 1], 15)
    hmed = float(np.nanmedian(hs))
    ow, oh = (int(x) for x in a.size.split("x"))
    info = probe(a.video)
    W, H = int(info.width), int(info.height)
    ch = min(H, hmed / a.fill)
    cw = min(W, ch * ow / oh)
    ch = cw * oh / ow
    print(f"рост человека {hmed:.0f} px → окно {cw:.0f}×{ch:.0f}, увеличение ×{oh / ch:.1f} (человек ≈ {hmed * oh / ch:.0f} px)")
    out = a.out or a.video.with_name(a.video.stem + "_follow.mp4")
    with FFmpegWriter(out, ow, oh, fps) as wr:
        for i, fr in enumerate(iter_frames(a.video)):
            j = min(i, len(cx) - 1)
            x0 = float(np.clip(cx[j] - cw / 2, 0, W - cw))
            y0 = float(np.clip(cy[j] - ch / 2, 0, H - ch))
            M = np.array([[ow / cw, 0, -x0 * ow / cw], [0, oh / ch, -y0 * oh / ch]], np.float32)       # субпиксельный сдвиг: окно плывёт плавно
            wr.write(cv2.warpAffine(fr.img, M, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
    print("готово:", out)


if __name__ == "__main__":
    main()
