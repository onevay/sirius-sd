"""Первичная установка на любой ОС одной командой:  python scripts/bootstrap.py [--dev] [--cpu-torch]

Создаёт .venv, ставит зависимости из requirements.txt (torch — CPU-сборкой, если нужно), проект в режиме разработки (`pip install -e . --no-deps`), копирует .env.example → .env
и запускает `sd doctor`. Ничего не качает, кроме пакетов; веса моделей — отдельно: `sd models get all`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="ещё и pytest")
    ap.add_argument("--cpu-torch", action="store_true", help="torch/torchvision из CPU-индекса (на Linux без этого pip тянет CUDA-колёса)")
    ap.add_argument("--minimal", action="store_true", help="только лёгкое ядро (оценка, разметка, мониторинг): без torch/ultralytics/openvino/transformers")
    a = ap.parse_args()
    if sys.version_info < (3, 11):
        print("нужен Python 3.11+")
        return 2
    vdir = ROOT / ".venv"
    if not vdir.exists():
        venv.create(vdir, with_pip=True)
    py = str(vdir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
    run(py, "-m", "pip", "install", "-q", "--upgrade", "pip")
    heavy = ("torch", "torchvision", "timm", "ultralytics", "openvino", "rtmlib", "transformers", "onnxruntime", "lap")
    reqs = [ln.strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    reqs = [r.split("#")[0].strip() for r in reqs]
    if a.minimal:
        reqs = [r for r in reqs if r.split("==")[0].lower() not in heavy]
    elif a.cpu_torch:
        run(py, "-m", "pip", "install", "torch==2.7.1", "torchvision==0.22.1", "--index-url", "https://download.pytorch.org/whl/cpu")
        reqs = [r for r in reqs if r.split("==")[0].lower() not in ("torch", "torchvision")]
    tmp = ROOT / ".requirements.install.txt"
    tmp.write_text("\n".join(r for r in reqs if not r.startswith("rtmlib")) + "\n", encoding="utf-8")
    try:
        run(py, "-m", "pip", "install", "-r", str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    if not a.minimal:
        run(py, "-m", "pip", "install", "--no-deps", "rtmlib==0.0.16")   # rtmlib тянет opencv-contrib, что ломает единственную сборку cv2
    run(py, "-m", "pip", "install", "--no-deps", "-e", ".")
    if a.dev:
        run(py, "-m", "pip", "install", "-q", "pytest")
    if not (ROOT / ".env").exists() and (ROOT / ".env.example").exists():
        shutil.copyfile(ROOT / ".env.example", ROOT / ".env")
    print("+ sd doctor", flush=True)
    subprocess.call((py, "-m", "sd", "doctor"), cwd=ROOT)       # отчёт, а не условие успеха: без весов и данных «[!!]» ожидаемы
    print("\nготово. Запуск:  sd.sh ui   |   sd.sh app   |   sd.sh --help   (Windows: sd.cmd)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
