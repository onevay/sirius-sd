"""Управление локальным Ollama для VLM-верификатора: запуск сервера, статус, загрузка и выгрузка моделей.

Сервер поднимается только на 127.0.0.1 и с отключёнными облачными функциями (`OLLAMA_NO_CLOUD=1`): кадры с записей объекта никуда не уходят,
`:cloud`-модели недоступны. В памяти держится одна модель (`OLLAMA_MAX_LOADED_MODELS=1`), один запрос за раз — на ноутбуке с 7.7 ГБ ОЗУ иначе своп.
Логи сервера — `outputs/ollama/server.*.log`.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from .paths import OUTPUTS

BASE = os.environ.get("SD_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")   # в контейнере — http://ollama:11434 (сервис docker compose, профиль vlm)
HOST = BASE.split("://", 1)[-1]
EXTERNAL = HOST not in ("127.0.0.1:11434", "localhost:11434")                    # сервер не на этой машине: ни запуск, ни остановку отсюда не делаем
LOG_DIR = OUTPUTS / "ollama"
SAFE_ENV = {"OLLAMA_HOST": HOST, "OLLAMA_NO_CLOUD": "1", "OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_CONTEXT_LENGTH": "4096"}


def find_ollama() -> Path | None:
    """ollama.exe: сначала PATH, затем каталог установки по умолчанию для текущего пользователя."""
    w = shutil.which("ollama")
    if w:
        return Path(w)
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return cand if cand.exists() else None


def is_up(base: str = BASE, timeout: float = 2.0) -> bool:
    try:
        return httpx.get(f"{base}/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def start(extra_env: dict[str, str] | None = None, wait_sec: float = 30.0) -> str:
    """Поднимает `ollama serve` (если не запущен) в отдельном процессе. Возвращает строку-итог."""
    if is_up():
        return "уже запущен"
    if EXTERNAL:
        raise RuntimeError(f"сервер Ollama внешний (SD_OLLAMA_URL={BASE}) и не отвечает: запустите его там (docker compose --profile vlm up -d ollama)")
    exe = find_ollama()
    if exe is None:
        raise FileNotFoundError("ollama.exe не найден: установите Ollama (https://github.com/ollama/ollama/releases, OllamaSetup.exe)")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **SAFE_ENV, **(extra_env or {})}
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" else 0
    with open(LOG_DIR / "server.out.log", "ab") as out, open(LOG_DIR / "server.err.log", "ab") as err:
        subprocess.Popen([str(exe), "serve"], env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    t0 = time.time()
    while time.time() - t0 < wait_sec:
        if is_up():
            return f"запущен за {time.time() - t0:.1f} с ({HOST}, без облака)"
        time.sleep(0.5)
    raise TimeoutError(f"сервер не ответил за {wait_sec:.0f} с, смотрите {LOG_DIR / 'server.err.log'}")


def stop() -> int:
    """Останавливает сервер и значок в трее (процессы `ollama` и `ollama app`). Возвращает число остановленных процессов."""
    n = 0
    if EXTERNAL:
        return 0
    for p in psutil.process_iter(["name"]):
        if (p.info["name"] or "").lower() in ("ollama.exe", "ollama app.exe", "ollama", "ollama app"):
            try:
                p.kill()
                n += 1
            except psutil.Error:
                pass
    return n


def status() -> dict:
    """Версия, установленные и загруженные в память модели, свободная ОЗУ."""
    mem = psutil.virtual_memory()
    out = dict(exe=str(find_ollama() or ""), up=is_up(), free_ram_gb=round(mem.available / 2**30, 2), total_ram_gb=round(mem.total / 2**30, 1), models=[], loaded=[])
    if not out["up"]:
        return out
    out["version"] = httpx.get(f"{BASE}/api/version", timeout=5).json().get("version")
    for m in httpx.get(f"{BASE}/api/tags", timeout=10).json().get("models", []):
        out["models"].append(dict(name=m["name"], size_gb=round(m["size"] / 1e9, 2), params=m.get("details", {}).get("parameter_size"), quant=m.get("details", {}).get("quantization_level")))
    for m in httpx.get(f"{BASE}/api/ps", timeout=10).json().get("models", []):
        out["loaded"].append(dict(name=m["name"], size_gb=round(m["size"] / 1e9, 2), in_vram_gb=round(m.get("size_vram", 0) / 1e9, 2), expires=m.get("expires_at")))
    return out


def pull(model: str, progress=None) -> None:
    """Загружает модель через API сервера (поток прогресса); сервер сам ходит в реестр Ollama."""
    with httpx.stream("POST", f"{BASE}/api/pull", json={"model": model, "stream": True}, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                raise RuntimeError(d["error"])
            if progress:
                progress(d.get("status", ""), d.get("completed"), d.get("total"))


def unload(model: str | None = None) -> list[str]:
    """Выгружает модели из памяти (keep_alive=0); без аргумента — все загруженные."""
    names = [model] if model else [m["name"] for m in httpx.get(f"{BASE}/api/ps", timeout=10).json().get("models", [])]
    for n in names:
        httpx.post(f"{BASE}/api/generate", json={"model": n, "keep_alive": 0}, timeout=60)
    return names
