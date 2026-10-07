"""Отчёт о железе и оценка применимости VLM (руководство, раздел 7)."""
from __future__ import annotations

from . import _env  # noqa: F401

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import psutil

from .paths import ROOT


def _cpu_name() -> str:
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        return platform.processor() or "unknown"


def _gpus_windows() -> list[dict]:
    if os.name != "nt":
        return []
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        data = json.loads(out) if out else []
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def _nvidia() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        rows = []
        for line in out.strip().splitlines():
            n, mem, drv = [x.strip() for x in line.split(",")]
            rows.append(dict(name=n, vram_mb=int(float(mem)), driver=drv))
        return rows
    except Exception:
        return []


def _bench_matmul(seconds: float = 2.0, n: int = 1024) -> dict:
    """Реальные GFLOPS fp32/bf16 на этом CPU (torch) — основа для оценок скорости моделей."""
    import torch

    out = {"threads": torch.get_num_threads()}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        a = torch.randn(n, n, dtype=dt)
        b = torch.randn(n, n, dtype=dt)
        for _ in range(2):
            a @ b
        t0, k = time.perf_counter(), 0
        while time.perf_counter() - t0 < seconds:
            a @ b
            k += 1
        dt_s = time.perf_counter() - t0
        out[f"gflops_{name}"] = round(2 * n ** 3 * k / dt_s / 1e9, 1)
    return out


def collect(bench: bool = False) -> dict:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    disk = shutil.disk_usage(ROOT)
    info: dict = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": {
            "name": _cpu_name(),
            "cores_physical": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "max_mhz": getattr(psutil.cpu_freq(), "max", None),
        },
        "ram": {"total_gb": round(vm.total / 2**30, 2), "available_gb": round(vm.available / 2**30, 2),
                "swap_total_gb": round(sw.total / 2**30, 2)},
        "disk": {"drive_free_gb": round(disk.free / 2**30, 1), "drive_total_gb": round(disk.total / 2**30, 1)},
        "gpu_windows": _gpus_windows(),
        "gpu_nvidia": _nvidia(),
    }
    try:
        import torch
        info["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available(),
                         "cpu_capability": torch.backends.cpu.get_cpu_capability(), "threads": torch.get_num_threads()}
    except Exception as e:  # pragma: no cover
        info["torch"] = {"error": str(e)}
    try:
        import openvino as ov
        core = ov.Core()
        info["openvino"] = {"version": ov.__version__.split("-")[0],
                            "devices": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}}
    except Exception as e:
        info["openvino"] = {"error": str(e)}
    try:
        import onnxruntime as ort
        info["onnxruntime"] = {"version": ort.__version__, "providers": ort.get_available_providers()}
    except Exception as e:
        info["onnxruntime"] = {"error": str(e)}
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        info["ffmpeg"] = {"exe": exe, "libx264": "libx264" in enc, "h264_qsv": "h264_qsv" in enc}
    except Exception as e:
        info["ffmpeg"] = {"error": str(e)}
    if bench:
        info["bench"] = _bench_matmul()
    return info


# Ориентиры по памяти (ГБ): веса + визуальный энкодер/mmproj. Источники: карточки HF/ggml-org и руководство (раздел 7.1);
# для моделей, которых нет в руководстве, — пересчёт по числу параметров (отмечено ~). Точные цифры даёт `sd vlm-bench`.
VLM_CANDIDATES = [
    # name, params_b, weights_gb by quant, license, video?
    dict(name="Qwen3.5-0.8B", params_b=0.9, q4=0.6, q8=0.9, bf16=1.7, license="Apache-2.0", video=True),
    dict(name="Qwen3.5-2B", params_b=2.0, q4=1.5, q8=2.4, bf16=4.5, license="Apache-2.0", video=True),
    dict(name="Qwen3.5-4B", params_b=4.7, q4=3.0, q8=5.0, bf16=9.5, license="Apache-2.0", video=True),
    dict(name="Qwen3-VL-2B", params_b=2.1, q4=1.9, q8=2.8, bf16=4.5, license="Apache-2.0", video=True),
    dict(name="Qwen3-VL-4B", params_b=4.4, q4=3.5, q8=6.0, bf16=10.0, license="Apache-2.0", video=True),
    dict(name="Qwen3-VL-8B", params_b=8.8, q4=5.5, q8=9.5, bf16=17.0, license="Apache-2.0", video=True),
    dict(name="SmolVLM2-500M-Video", params_b=0.5, q4=0.4, q8=0.6, bf16=1.0, license="Apache-2.0", video=True),
    dict(name="SmolVLM2-2.2B-Video", params_b=2.2, q4=1.6, q8=2.5, bf16=4.5, license="Apache-2.0", video=True),
]


def vlm_budget(info: dict, os_overhead_gb: float = 3.5, kv_and_runtime_gb: float = 1.0) -> list[dict]:
    """Какие VLM помещаются в память этой машины (без дискретного GPU — только RAM)."""
    ram = info["ram"]["total_gb"]
    avail_now = info["ram"]["available_gb"]
    vram = max([g["vram_mb"] for g in info.get("gpu_nvidia", [])] or [0]) / 1024
    usable = vram if vram >= 4 else max(ram - os_overhead_gb, 0)
    where = "VRAM" if vram >= 4 else "RAM (CPU)"
    rows = []
    for m in VLM_CANDIDATES:
        for q in ("q4", "q8", "bf16"):
            need = m[q] + kv_and_runtime_gb
            rows.append(dict(model=m["name"], quant=q.upper(), weights_gb=m[q], need_gb=round(need, 1),
                             fits=need <= usable, fits_now=need <= avail_now, memory=where, usable_gb=round(usable, 1),
                             avail_now_gb=avail_now))
    return rows


def render_report(info: dict) -> str:
    c, r = info["cpu"], info["ram"]
    gpu = ", ".join(g.get("Name", "?") for g in info["gpu_windows"]) or "—"
    nv = ", ".join(f"{g['name']} {g['vram_mb']} МБ" for g in info["gpu_nvidia"]) or "нет (CUDA недоступна)"
    lines = [
        f"ОС: {info['os']} | Python {info['python']}",
        f"CPU: {c['name']} — {c['cores_physical']} ядра / {c['threads']} потока, до {c['max_mhz']} МГц",
        f"ОЗУ: всего {r['total_gb']} ГБ, доступно сейчас {r['available_gb']} ГБ, своп {r['swap_total_gb']} ГБ",
        f"Диск (проект): свободно {info['disk']['drive_free_gb']} из {info['disk']['drive_total_gb']} ГБ",
        f"GPU (Windows): {gpu}",
        f"NVIDIA: {nv}",
        f"torch: {info.get('torch')}",
        f"OpenVINO: {info.get('openvino')}",
        f"ONNX Runtime: {info.get('onnxruntime')}",
        f"ffmpeg: {info.get('ffmpeg')}",
    ]
    if "bench" in info:
        lines.append(f"Бенчмарк matmul 1024²: {info['bench']}")
    return "\n".join(lines)
