"""`sd doctor` — проверка окружения перед запуском на ДРУГОЙ машине (или в контейнере): пакеты против lock-файла, устройства, веса, пакеты моделей, Ollama, данные.

Ничего не скачивает и не запускает моделей: только читает версии и файлы; поэтому безопасно и быстро (< 10 с). Результат — словарь (для UI/JSON) и текстовый отчёт.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import importlib.metadata as md
import platform
import re
import shutil
import sys
from pathlib import Path

from .paths import MODELS, ROOT, VIDEO_EXT

KEY_PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "torch", "torchvision", "timm", "transformers", "ultralytics", "openvino",
                "onnxruntime", "opencv-python", "streamlit", "plotly", "typer", "rich", "psutil", "httpx", "huggingface-hub", "safetensors", "imageio-ffmpeg", "PyYAML", "rtmlib"]


def _norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def parse_lock(path: Path) -> dict[str, str]:
    """name==version из requirements*.txt (комментарии и пустые строки пропускаются)."""
    out = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if "==" in ln:
                n, v = ln.split("==", 1)
                out[_norm(n.split("[")[0])] = v.strip()
    return out


def installed_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def check(root: Path = ROOT) -> dict:
    from . import hw
    from .models import load_registry, status

    lock = parse_lock(root / "requirements.lock.txt") or parse_lock(root / "requirements.txt")
    pk = {}
    for n in KEY_PACKAGES:
        v = installed_version(n)
        want = lock.get(_norm(n))
        pk[n] = dict(installed=v, pinned=want, ok=v is not None and (want is None or v == want))
    info = hw.collect(bench=False)
    reg = load_registry()
    weights = [status(s) for s in reg.values()]
    need = {"yolo26n-pose", "smoking_yolo11m_beehzod", "clip-vit-b32", "convnext-b-in22k"}          # минимум для полного цикла распознавания
    from .model_tools import list_bundles

    bundles = list_bundles()
    try:
        from . import ollama_ctl as O

        ollama = O.status()
    except Exception as e:
        ollama = dict(up=False, error=str(e))
    data = {d: sum(1 for f in (root / "data" / d).iterdir() if f.suffix.lower() in VIDEO_EXT) for d in ("курение", "лжекурение") if (root / "data" / d).exists()}   # не только .mp4: есть .mkv/.wmv
    checks = [
        ("Python ≥ 3.11", sys.version_info >= (3, 11), platform.python_version()),
        ("пакеты совпадают с lock-файлом", all(p["ok"] for p in pk.values()), ", ".join(f"{n} {p['installed']}≠{p['pinned']}" for n, p in pk.items() if not p["ok"]) or "все ключевые пакеты"),
        ("ffmpeg (imageio-ffmpeg) c libx264", bool(info.get("ffmpeg", {}).get("libx264")), str(info.get("ffmpeg", {}).get("exe", info.get("ffmpeg")))),
        ("ускоритель: OpenVINO GPU / CUDA", bool(info.get("openvino", {}).get("devices", {}).get("GPU") or info.get("torch", {}).get("cuda")),
         f"openvino {list(info.get('openvino', {}).get('devices', {}))}, cuda {info.get('torch', {}).get('cuda')}; без ускорителя всё работает на CPU, но медленнее"),
        ("веса для полного цикла скачаны", all(w["present"] for w in weights if w["id"] in need), ", ".join(w["id"] for w in weights if w["id"] in need and not w["present"]) or "да"),
        ("пакеты моделей (photo/cycle) есть", any(b["kind"] == "photo" for b in bundles) and any(b["kind"] == "cycle" for b in bundles),
         ", ".join(f"{b['kind']}/{b['name']}" for b in bundles) or "нет — sd photos train / sd train-bundle или импорт из zip (вкладка «Модели»)"),
        ("свободная память ≥ 3 ГБ", info["ram"]["available_gb"] >= 3.0, f"{info['ram']['available_gb']} ГБ из {info['ram']['total_gb']}"),
        ("свободно на диске ≥ 5 ГБ", info["disk"]["drive_free_gb"] >= 5.0, f"{info['disk']['drive_free_gb']} ГБ"),
        ("Ollama (для VLM) запущен", bool(ollama.get("up")), "нужен только для режима VLM" + (f"; модели: {[m['name'] for m in ollama.get('models', [])]}" if ollama.get("up") else "")),
        ("видео данных найдены", bool(data), str(data) or "папок data/курение, data/лжекурение нет (для распознавания нужно только своё видео)"),
    ]
    return dict(packages=pk, hw=info, weights=weights, bundles=bundles, ollama=ollama, data=data, checks=[dict(name=n, ok=ok, note=note) for n, ok, note in checks],
                critical_ok=all(ok for n, ok, _ in checks if n.startswith(("Python", "пакеты", "ffmpeg"))))


def render_report(r: dict) -> str:
    lines = ["Проверка окружения (sd doctor):"]
    for c in r["checks"]:
        lines.append(f"  [{'OK' if c['ok'] else '!!'}] {c['name']}: {c['note']}")
    return "\n".join(lines)
