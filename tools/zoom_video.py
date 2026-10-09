"""Увеличить дальних людей в видео: найти область, где они есть, вырезать её и растянуть до 1920×1080.

Детектор людей не видит людей ниже ≈ 60 px на входе модели (вертикальное видео 720×1280 сжимается до 544×960). Скрипт один раз ищет людей
на увеличенных кадрах, берёт общую рамку всех найденных людей с запасом, приводит её к 16:9 и перекодирует ролик (H.264, звук убран).
Запуск:  .venv\\Scripts\\python tools\\zoom_video.py data\\custom\\IMG_1319.mp4   → data\\custom\\IMG_1319_zoom.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sd.models import local_weights  # noqa: E402
from sd.video_io import iter_frames, probe  # noqa: E402


def people_box(video: Path, every_sec: float = 1.0, imgsz: int = 1920) -> tuple[int, int, int, int]:
    from ultralytics import YOLO

    m = YOLO(str(local_weights("yolo26n-pose")))
    info = probe(video)
    boxes = []
    for fr in iter_frames(video, 0, None, max(1, int(round(info.fps * every_sec)))):
        r = m.predict(fr.img, imgsz=imgsz, conf=0.25, classes=[0], verbose=False, device="cpu")[0]
        boxes += r.boxes.xyxy.tolist()
    if not boxes:
        raise SystemExit("людей не найдено даже на увеличенных кадрах")
    b = np.array(boxes)
    return int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())


def crop_16x9(box, W: int, H: int, margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h = (y2 - y1) * (1 + 2 * margin) + 1
    w = max((x2 - x1) * (1 + 2 * margin), h * 16 / 9)
    w = min(w, W)
    h = w * 9 / 16
    if h > H:
        h, w = H, H * 16 / 9
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x = int(np.clip(cx - w / 2, 0, W - w)) // 2 * 2
    y = int(np.clip(cy - h / 2, 0, H - h)) // 2 * 2
    return x, y, int(w) // 2 * 2, int(h) // 2 * 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--margin", type=float, default=0.6, help="запас вокруг людей (доля высоты рамки)")
    ap.add_argument("--size", default="1920:1080")
    a = ap.parse_args()
    info = probe(a.video)
    box = people_box(a.video)
    x, y, w, h = crop_16x9(box, int(info.width), int(info.height), a.margin)
    out = a.out or a.video.with_name(a.video.stem + "_zoom.mp4")
    k = int(a.size.split(":")[1]) / h
    print(f"люди: {box}; вырез {w}×{h} с ({x},{y}); увеличение ×{k:.2f} (человек {box[3] - box[1]} px → ≈ {int((box[3] - box[1]) * k)} px)")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(a.video), "-vf", f"crop={w}:{h}:{x}:{y},scale={a.size}:flags=lanczos",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
    subprocess.run(cmd, check=True)
    print("готово:", out)


if __name__ == "__main__":
    main()
