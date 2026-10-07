"""Настройка окружения. Импортируется ПЕРВЫМ (до cv2 / ultralytics / transformers).

- глушит шум FFmpeg («non-existing SPS ... referenced in buffering period»);
- держит кэши и конфиги библиотек внутри проекта (ничего не пишется в профиль пользователя);
- отключает телеметрию: обработка локальная (требование регламента).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SD_ROOT") or Path(__file__).resolve().parents[2])


def setup() -> None:
    env = os.environ.setdefault
    env("PYTHONUTF8", "1")
    env("PYTHONIOENCODING", "utf-8")
    env("OPENCV_LOG_LEVEL", "ERROR")
    env("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET
    env("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    # Аналитика Ultralytics включается ПРИ ИМПОРТЕ (settings.sync и «есть сеть»), последующий settings.update(sync=False) её уже не выключает; YOLO_OFFLINE=1 гасит и её,
    # и DNS-проверку сети при импорте. Веса и данные проект получает сам (`sd models get`, `sd photos fetch`), автозагрузки Ultralytics не нужны.
    env("YOLO_OFFLINE", "1")
    try:   # Ultralytics уходит в запасной каталог C:\tmp\Ultralytics, если родитель своего каталога не существует: создаём его заранее
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass   # только чтение (контейнер с read_only): Ultralytics сам выберет запасной каталог
    env("HF_HOME", str(ROOT / "models" / "hf"))
    env("HF_HUB_DISABLE_TELEMETRY", "1")
    env("TOKENIZERS_PARALLELISM", "false")
    env("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")


setup()
