"""Демонстрационные тревоги для разработки интерфейса оператора БЕЗ моделей и видео. Все записи помечены `demo=1` и в интерфейсе подписаны «ДЕМО»; удаляются командой `sd monitor-demo --clear`.
Это синтетика (градиент и прямоугольник), а не результаты распознавания — не использовать как данные для оценки."""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..paths import portable
from .engine import AlertUpdate
from .store import AlertStore
from .worker import ARTIFACTS

DISTRICTS = ["gatchina", "pavlovsk", "tsarskoe-selo"]


def _frame(i: int, w: int = 480, h: int = 270):
    import cv2

    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (70 + (i * 3) % 40, 90, 110)
    x = 80 + (i * 5) % 260
    cv2.rectangle(img, (x, 60), (x + 70, 250), (60, 60, 255), 3)
    cv2.putText(img, "DEMO", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def seed_demo(store: AlertStore, n: int = 14, seed: int = 7, root: Path | None = None, with_clips: bool = True, now: datetime | None = None) -> int:
    rnd = random.Random(seed)
    now = now or datetime.now().replace(microsecond=0)
    root = root or ARTIFACTS
    made = 0
    for i in range(n):
        dist = rnd.choice(DISTRICTS)
        idx = f"{rnd.randint(1, 4):02d}"
        cam = f"{dist}-{idx}"
        start_at = (now - timedelta(hours=3)).isoformat(timespec="seconds")
        store.upsert_camera(cam, dist, idx, f"demo:{cam}", start_at, "idle")
        t0 = rnd.uniform(60, 10000)
        conf = round(rnd.uniform(0.45, 0.97), 2)
        u = AlertUpdate("open", f"demo:{cam}:{i}:{t0:.2f}", cam, rnd.randint(1, 6), t0, t0 + rnd.uniform(14, 40), t0 + 8, conf, rnd.choice(["two_cycles", "cycle+object"]),
                        rnd.choice([2, 2, 3]), rnd.choice(["2 цикла за 14 с", "2 цикла за 18 с, предмет у рта", "3 цикла за 25 с"]), (120.0, 40.0, 200.0, 250.0), t0 + 20)
        t_abs = (now - timedelta(minutes=rnd.randint(1, 170))).isoformat(timespec="seconds")
        aid, _ = store.apply_update(u, dict(district=dist, cam_index=idx, chunk=None, chunk_offset=0.0, t_abs=t_abs, fingerprint="demo", demo=1))
        try:
            import cv2

            d = root / f"demo_{aid}"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "thumb.jpg"), _frame(aid * 7))
            clip = None
            if with_clips:
                from ..video_io import FFmpegWriter

                with FFmpegWriter(d / "clip.mp4", 480, 270, 10.0) as wr:
                    for k in range(40):
                        wr.write(_frame(aid * 7 + k))
                clip = portable(d / "clip.mp4")
            store.set_artifacts(aid, portable(d / "thumb.jpg"), clip)
        except Exception:
            pass
        made += 1
    return made
