"""Эмбеддинги изображений для фото-классификатора «курит / не курит» (замороженные backbone'ы, только safetensors).

Backbone'ы (веса — реестр `sd models`):
  * `clip`     — CLIP ViT-B/32 (LAION-2B), 512-мерный нормированный эмбеддинг; умеет ещё и текст (zero-shot);
  * `convnext` — ConvNeXt-B (ImageNet-22k, Meta), 1024-мерный пулинг.
Два рантайма с ОДНОЙ предобработкой (numpy), поэтому признаки совпадают:
  * `torch` — как есть (на машине с NVIDIA это путь по умолчанию);
  * `ov`    — OpenVINO IR (models/ov/…), iGPU/CPU; на этом ноутбуке в разы быстрее (torch упирается в ОЗУ и 2 ядра).
Эмбеддинги кэшируются по файлам (outputs/photo/emb/<backbone>.npz): прогон можно прервать и продолжить.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import os
import time
from pathlib import Path

import cv2
import numpy as np

from .models import hf_dir, load_registry
from .paths import EXTERNAL, MODELS, OUTPUTS, ROOT

EMB_DIR = OUTPUTS / "photo" / "emb"
SIZE = 224
OV_BATCH = 8

ZERO_SHOT_PROMPTS = {
    "smoke": "a photo of a person smoking a cigarette",
    "hold": "a photo of a person holding a lit cigarette in their hand",
    "drink": "a photo of a person drinking from a cup or a bottle",
    "phone": "a photo of a person talking on a phone",
    "none": "a photo of a person with no cigarette",
}
NORM = {  # (mean, std) в долях 0..1
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "convnext": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
DIM = {"clip": 512, "convnext": 1024}


def to_square_rgb(img_bgr: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Дополняем до квадрата отражением краёв (кадр целиком, без обрезки сбоку) и масштабируем до size×size, RGB uint8."""
    h, w = img_bgr.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(img_bgr, top, s - h - top, left, s - w - left, cv2.BORDER_REFLECT_101) if h != w else img_bgr
    interp = cv2.INTER_AREA if s > size else cv2.INTER_CUBIC
    return cv2.cvtColor(cv2.resize(sq, (size, size), interpolation=interp), cv2.COLOR_BGR2RGB)


def normalize(rgb: list[np.ndarray], name: str) -> np.ndarray:
    """(N,3,224,224) float32 для backbone `name`: один и тот же код для torch и OpenVINO."""
    mean, std = (np.array(v, np.float32)[None, :, None, None] for v in NORM[name])
    x = np.stack([np.asarray(r, np.uint8) for r in rgb]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return np.ascontiguousarray((x - mean) / std)


def _threads() -> int:
    return int(os.environ.get("SD_THREADS", "2"))


def _local(spec_id: str) -> Path:
    d = hf_dir(load_registry()[spec_id])
    if d is None:
        raise FileNotFoundError(f"веса «{spec_id}» не скачаны: sd models get {spec_id}")
    return d


# ------------------------------------------------------------------------------------------ torch
def torch_device(torch) -> str:
    """Устройство torch-эмбеддеров: переменная окружения SD_TORCH_DEVICE (cpu | cuda | cuda:0). Нет CUDA — cpu, даже если просили cuda."""
    want = os.environ.get("SD_TORCH_DEVICE", "cpu").strip().lower() or "cpu"
    return want if want == "cpu" or (want.startswith("cuda") and torch.cuda.is_available()) else "cpu"


def build_torch(name: str):
    """(torch-модуль «картинка → эмбеддинг», torch). Для clip эмбеддинг нормирован по L2."""
    import torch

    torch.set_num_threads(_threads())
    if name == "clip":
        from transformers import CLIPModel

        full = CLIPModel.from_pretrained(_local("clip-vit-b32"), use_safetensors=True).eval()

        class ImgTower(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                e = self.m.get_image_features(pixel_values=x)
                return e / e.norm(dim=-1, keepdim=True)

        return ImgTower(full).eval(), torch, full
    if name == "convnext":
        import timm

        _local("convnext-b-in22k")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")   # веса уже в models/hf: сеть не нужна
        return timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True, num_classes=0).eval(), torch, None
    raise KeyError(name)


class TorchEmbedder:
    def __init__(self, name: str):
        self.name, self.dim = name, DIM[name]
        self.model, self.torch, self._full = build_torch(name)
        self.device = torch_device(self.torch)
        if self.device != "cpu":
            self.model.to(self.device)
            self._full.to(self.device)
        self._text: np.ndarray | None = None

    def embed(self, rgb: list[np.ndarray]) -> np.ndarray:
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(normalize(rgb, self.name)).to(self.device)).cpu().numpy().astype(np.float32)

    def text_features(self, prompts: dict[str, str]) -> np.ndarray:
        from transformers import CLIPTokenizer

        tok = CLIPTokenizer.from_pretrained(_local("clip-vit-b32"))(list(prompts.values()), padding=True, return_tensors="pt")
        with self.torch.no_grad():
            t = self._full.get_text_features(**{k: v.to(self.device) for k, v in tok.items()})
        return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------------------------------ OpenVINO
def ov_path(name: str) -> Path:
    return MODELS / "ov" / f"photo_{name}_{SIZE}_openvino_model" / f"photo_{name}.xml"


def export_ov(name: str, force: bool = False) -> Path:
    """Конвертирует backbone в OpenVINO IR (fp16-веса, статический батч OV_BATCH). Один раз; нужен torch."""
    import openvino as ov

    dst = ov_path(name)
    if dst.exists() and not force:
        return dst
    model, torch, _ = build_torch(name)
    ex = torch.zeros(OV_BATCH, 3, SIZE, SIZE)
    ovm = ov.convert_model(model, example_input=ex)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ovm, dst, compress_to_fp16=True)
    return dst


def ov_parity(name: str, n: int = 8, images: list[Path] | None = None) -> dict:
    """Проверка конвертации: косинус между эмбеддингами OpenVINO-IR и torch на n снимках (реальные из `data/Smoker Detection` и `data_external`, иначе синтетика с фиксированным зерном).
    Возвращает {n, cos_mean, cos_min, real}: после экспорта в fp16 ожидается cos_mean ≥ 0.99 (измерено на этом ноутбуке: CLIP 0.998, ConvNeXt 0.99998)."""
    if images is None:
        roots = [ROOT / "data" / "Smoker Detection", *sorted(EXTERNAL.glob("*/*"))]
        images = [q for r in roots if r.exists() for q in sorted(r.rglob("*.jpg"))[:n]][:n]
    rgb = []
    for q in images:
        im = cv2.imread(str(q))
        if im is not None:
            rgb.append(to_square_rgb(im))
    real = bool(rgb)
    if not rgb:
        rng = np.random.default_rng(0)
        rgb = [np.clip(np.linspace(0, 255, SIZE)[None, :, None] + rng.normal(0, 30, (SIZE, SIZE, 3)), 0, 255).astype(np.uint8) for _ in range(n)]
    a, b = TorchEmbedder(name).embed(rgb), OVEmbedder(name).embed(rgb)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return dict(n=len(rgb), cos_mean=float(cos.mean()), cos_min=float(cos.min()), real=real)


class OVEmbedder:
    def __init__(self, name: str, device: str | None = None):
        import openvino as ov

        from .pose_track import enable_openvino_cache

        enable_openvino_cache()
        self.name, self.dim = name, DIM[name]
        core = ov.Core()
        dev = device or ("GPU" if "GPU" in core.available_devices else "CPU")
        self.device = dev
        self.compiled = core.compile_model(core.read_model(ov_path(name)), dev)
        self.out = self.compiled.output(0)

    def embed(self, rgb: list[np.ndarray]) -> np.ndarray:
        res = []
        for i in range(0, len(rgb), OV_BATCH):
            chunk = rgb[i:i + OV_BATCH]
            x = normalize(chunk + [chunk[-1]] * (OV_BATCH - len(chunk)), self.name)   # хвост добиваем копией
            res.append(self.compiled(x)[self.out][:len(chunk)])
        return np.concatenate(res).astype(np.float32)


def make_embedder(name: str, backend: str = "auto"):
    """backend: ov | torch | auto (ov, если IR уже экспортирован и есть openvino, иначе torch)."""
    if backend in ("ov", "auto") and ov_path(name).exists() and not (backend == "auto" and os.environ.get("SD_TORCH_DEVICE", "cpu").lower().startswith("cuda")):
        try:
            return OVEmbedder(name)
        except Exception:
            if backend == "ov":
                raise
    if backend == "ov":
        raise FileNotFoundError(f"нет {ov_path(name)}: sd photos export-ov {name}")
    return TorchEmbedder(name)


def clip_text_embeddings(prompts: dict[str, str] = ZERO_SHOT_PROMPTS, cache_name: str = "clip_text") -> np.ndarray:
    """Эмбеддинги текстов промптов (CLIP text tower, один раз через torch, кэш в outputs/photo/emb/<cache_name>.npz; у каждого набора промптов свой файл)."""
    f = EMB_DIR / f"{cache_name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if z["prompts"].tolist() == list(prompts.values()):
            return z["emb"]
    t = TorchEmbedder("clip").text_features(prompts)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(f, prompts=np.array(list(prompts.values()), dtype=str), emb=t)
    return t


def zero_shot(emb: np.ndarray, text: np.ndarray, prompts: dict[str, str] = ZERO_SHOT_PROMPTS) -> dict[str, np.ndarray]:
    """Косинусные сходства нормированных эмбеддингов CLIP с текстом промптов."""
    sim = emb @ text.T
    return {k: sim[:, i] for i, k in enumerate(prompts)}


# ------------------------------------------------------------------------------------------ кэш
def _key(p: str | Path) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def load_cache(name: str) -> dict[str, np.ndarray]:
    f = EMB_DIR / f"{name}.npz"
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=False)
    return dict(zip(z["keys"].tolist(), z["emb"].astype(np.float32)))


def save_cache(name: str, cache: dict[str, np.ndarray]) -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(cache), dtype=str)
    np.savez(EMB_DIR / f"{name}.npz", keys=keys, emb=np.stack([cache[k] for k in keys]).astype(np.float16))


def embed_files(emb, paths: list[str], batch: int = 16, progress=None, save_every: int = 10) -> np.ndarray:
    """Эмбеддинги файлов с кэшем (по пути относительно корня проекта). Порядок результата = порядку paths."""
    cache = load_cache(emb.name)
    keys = [_key(p) for p in paths]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    t0 = time.perf_counter()
    for b, start in enumerate(range(0, len(todo), batch)):
        idx = todo[start:start + batch]
        imgs, ok = [], []
        for i in idx:
            im = cv2.imread(str(paths[i]))
            if im is None:
                continue
            imgs.append(to_square_rgb(im))
            ok.append(i)
        if imgs:
            for i, v in zip(ok, emb.embed(imgs)):
                cache[keys[i]] = v
        if progress:
            progress(min(start + batch, len(todo)), len(todo), time.perf_counter() - t0)
        if (b + 1) % save_every == 0:
            save_cache(emb.name, cache)
    if todo:
        save_cache(emb.name, cache)
    out = np.full((len(paths), emb.dim), np.nan, np.float32)
    for i, k in enumerate(keys):
        if k in cache:
            out[i] = cache[k]
    return out
