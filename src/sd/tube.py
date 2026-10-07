"""«Трубка» человека для видео-моделей и их запуск: X-CLIP (zero-shot видео-текст) и VideoMAE (Kinetics-400, класс smoking).

Подавать модели весь кадр бессмысленно: курящий занимает 1–3% площади (руководство §3.3). Поэтому по треку вырезаем кроп верхней части
тела (рамка + запас), центрируем, берём 16 кадров за ~2.5 с и ресайзим до 224×224. Вероятности — не решение, а колонки вектора признаков.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .models import hf_dir, load_registry
from .tracks import Tracks
from .video_io import iter_frames


def _interp_box(rows: pd.DataFrame, t: float) -> np.ndarray | None:
    """Рамка трека на момент t (линейная интерполяция по времени; вне диапазона трека — None)."""
    ts = rows.t.values
    if t < ts.min() - 0.3 or t > ts.max() + 0.3:
        return None
    return np.array([np.interp(t, ts, rows[c].values) for c in ("x1", "y1", "x2", "y2")])


def tube_frames(video: str | Path, tr: Tracks, tid: int, t0: float, t1: float, cfg: dict, n: int | None = None
                ) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:
    """Возвращает (n кадров RGB 224×224, их времена, исходные кропы BGR для показа). Пустой список, если трек не покрывает окно."""
    tc = cfg["tube"]
    n = n or tc["frames"]
    rows, _ = tr.of(tid)
    targets = np.linspace(t0, t1, n)
    pick: dict[int, list[int]] = {}
    frames_buf: dict[int, tuple[float, np.ndarray]] = {}
    for fr in iter_frames(video, max(0.0, t0 - 0.05), t1 + 0.05, 1):
        frames_buf[fr.idx] = (fr.t, fr.img)
    if not frames_buf:
        return [], [], []
    idxs = np.array(sorted(frames_buf))
    times = np.array([frames_buf[i][0] for i in idxs])
    sel = [int(idxs[np.abs(times - t).argmin()]) for t in targets]
    boxes = [_interp_box(rows, frames_buf[i][0]) for i in sel]
    if any(b is None for b in boxes):
        return [], [], []
    b = np.stack(boxes)
    h_up = (b[:, 3] - b[:, 1]) * tc["upper_body_frac"]
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = b[:, 1] + h_up / 2
    side = float(np.median(np.maximum(b[:, 2] - b[:, 0], h_up))) * (1 + tc["margin"])  # постоянный размер кропа на окне
    k = max(3, int(len(cx) // 4) * 2 + 1)
    cx = pd.Series(cx).rolling(k, min_periods=1, center=True).median().values  # сглаживаем центр: меньше дрожания фона
    cy = pd.Series(cy).rolling(k, min_periods=1, center=True).median().values
    out, raw, tt = [], [], []
    for i, fi in enumerate(sel):
        t, img = frames_buf[fi]
        H, W = img.shape[:2]
        x1, y1 = int(round(cx[i] - side / 2)), int(round(cy[i] - side / 2))
        x2, y2 = int(round(x1 + side)), int(round(y1 + side))
        pad = max(0, -x1, -y1, x2 - W, y2 - H)
        src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img
        crop = src[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
        if crop.size == 0:
            return [], [], []
        raw.append(crop)
        out.append(cv2.cvtColor(cv2.resize(crop, (tc["size"], tc["size"]), interpolation=cv2.INTER_CUBIC if side < tc["size"] else cv2.INTER_AREA),
                                cv2.COLOR_BGR2RGB))
        tt.append(t)
    return out, tt, raw


def _snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download

    reg = {s.hf_repo: s for s in load_registry().values() if s.hf_repo}
    pats = reg[repo].hf_patterns if repo in reg else None
    return snapshot_download(repo, allow_patterns=pats or None, local_files_only=True)


class XCLIPScorer:
    """X-CLIP base/32, 16 кадров (MIT): сходство «трубки» с текстовыми промптами -> softmax по промптам."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import XCLIPModel, XCLIPProcessor

        self.cfg = cfg
        self.prompts = list(cfg["tube"]["prompts"])
        path = _snapshot(cfg["tube"]["xclip_weights"])
        self.proc = XCLIPProcessor.from_pretrained(path)
        self.model = XCLIPModel.from_pretrained(path, use_safetensors=True).eval()
        # XCLIPProcessor.__call__ в transformers 4.5x молча игнорирует `videos=` (video_processor не входит в attributes) и отдаёт только токены
        # текста -> модель получала pixel_values=None. Поэтому текст и видео готовим раздельно: токенизируем один раз, кадры — image_processor'ом.
        self.text_inp = dict(self.proc.tokenizer(self.prompts, padding=True, return_tensors="pt"))
        self.torch = torch
        self.timing: list[float] = []

    def score(self, frames: list[np.ndarray]) -> dict[str, float]:
        t0 = time.perf_counter()
        pix = self.proc.image_processor(list(frames), return_tensors="pt")["pixel_values"]   # (1, T, 3, 224, 224)
        with self.torch.no_grad():
            p = self.model(pixel_values=pix, **self.text_inp).logits_per_video.softmax(-1)[0]
        self.timing.append((time.perf_counter() - t0) * 1000)
        return {f"p_{k}": float(v) for k, v in zip(self.prompts, p.tolist())}


class VideoMAEScorer:
    """VideoMAE base, Kinetics-400 (CC-BY-NC-4.0, только некоммерческое): вероятность класса smoking и родственных (drinking, eating ...)."""

    RELEVANT = ("smoking", "drinking", "drinking beer", "drinking shots", "tasting beer", "eating burger", "eating cake", "eating carrots",
                "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon",
                "brushing teeth", "blowing nose", "yawning", "texting", "talking on cell phone", "answering questions", "applauding")

    def __init__(self, cfg: dict):
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        path = _snapshot(cfg["tube"]["videomae_weights"])
        self.proc = VideoMAEImageProcessor.from_pretrained(path)
        self.model = VideoMAEForVideoClassification.from_pretrained(path, use_safetensors=True).eval()
        self.id2label = self.model.config.id2label
        self.rel_ids = {i: l for i, l in self.id2label.items() if l in self.RELEVANT}
        self.torch = torch
        self.timing: list[float] = []

    def score(self, frames: list[np.ndarray]) -> dict[str, float]:
        t0 = time.perf_counter()
        inp = self.proc(list(frames), return_tensors="pt")
        with self.torch.no_grad():
            pr = self.model(**inp).logits.softmax(-1)[0]
        self.timing.append((time.perf_counter() - t0) * 1000)
        out = {f"k_{l}": float(pr[i]) for i, l in self.rel_ids.items()}
        top = self.torch.topk(pr, 3)
        out["top1"] = self.id2label[int(top.indices[0])]
        out["top1_p"] = float(top.values[0])
        out["top3"] = "; ".join(f"{self.id2label[int(i)]} {float(v):.2f}" for v, i in zip(top.values, top.indices))
        return out


def run_tube(video: str | Path, tr: Tracks, windows: list[tuple[int, int, float, float]], cfg: dict, models: tuple[str, ...] = ("xclip",),
             progress=None, save_dir: Path | None = None) -> pd.DataFrame:
    """Прогон видео-моделей по окнам (индекс, tid, t0, t1). Результат — по строке на окно со всеми вероятностями."""
    scorers = {}
    if "xclip" in models:
        scorers["xclip"] = XCLIPScorer(cfg)
    if "videomae" in models:
        scorers["videomae"] = VideoMAEScorer(cfg)
    rows = []
    for wi, (ci, tid, t0, t1) in enumerate(windows):
        frames, ts, raw = tube_frames(video, tr, tid, t0, t1, cfg)
        if not frames:
            continue
        row = dict(window=ci, tid=tid, t0=t0, t1=t1)
        for name, sc in scorers.items():
            res = sc.score(frames)
            row.update({(f"{name}_{k}" if name != "xclip" else k): v for k, v in res.items()})
        rows.append(row)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            strip = np.hstack([cv2.resize(r, (112, 112)) for r in raw[:: max(1, len(raw) // 8)][:8]])
            cv2.imwrite(str(save_dir / f"tube_w{ci:03d}_tid{tid}_{t0:.1f}s.jpg"), strip)
        if progress:
            progress(wi + 1, len(windows))
    df = pd.DataFrame(rows)
    df.attrs["ms"] = {k: (float(np.mean(s.timing)) if s.timing else None) for k, s in scorers.items()}
    return df
