"""Как отдать видео в браузер: маленькие копии вшиваются в страницу (data-URI), большие отдаются файлом по HTTP через статическую раздачу Streamlit.

Почему не всегда data-URI: копия на десятки мегабайт, закодированная в base64, превращается в страницу на сотни мегабайт — браузер её не показывает или показывает пустое видео.
Статическая раздача (`server.enableStaticServing`, включена в `.streamlit/config.toml` и командах запуска) читает файл из `src/sd/ui/static/` по адресу `/app/static/…` с поддержкой
перемотки (Range). Файлы публикуются жёсткой ссылкой (копией, если ссылка невозможна); старые удаляются, чтобы каталог не рос.
"""
from __future__ import annotations

import base64
import hashlib
import os
import shutil
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static" / "media"
URL_PREFIX = "/app/static/media/"
EMBED_SMALL_MB = float(os.environ.get("SD_EMBED_SMALL_MB", "6"))   # до этого размера вшиваем (работает без статической раздачи); 0 — всегда файлом
EMBED_MAX_MB = 70.0           # без статической раздачи больше не вшиваем
KEEP_FILES = 40


def static_enabled() -> bool:
    try:
        import streamlit as st

        return bool(st.get_option("server.enableStaticServing"))
    except Exception:   # noqa: BLE001 — вне Streamlit (тесты, скрипты)
        return False


def _mime(p: Path) -> str:
    return "video/webm" if p.suffix.lower() == ".webm" else "video/mp4"


def publish(path: str | Path, static_dir: Path | None = None) -> str:
    """Кладёт файл в каталог статической раздачи и возвращает URL (с отметкой версии, чтобы браузер не держал старую копию)."""
    p = Path(path)
    d = static_dir or STATIC_DIR
    d.mkdir(parents=True, exist_ok=True)
    st = p.stat()
    name = hashlib.md5(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16] + p.suffix.lower()
    dst = d / name
    if not (dst.exists() and dst.stat().st_size == st.st_size):
        tmp = d / (name + ".part")
        tmp.unlink(missing_ok=True)
        try:
            os.link(p, tmp)
        except OSError:
            shutil.copy2(p, tmp)
        tmp.replace(dst)
    if dst.stat().st_nlink == 1:        # у жёсткой ссылки время общее с исходным файлом: его трогать нельзя (сменилось бы имя и URL при каждом обновлении страницы)
        os.utime(dst, None)
    files = sorted((f for f in d.iterdir() if f.is_file() and f.suffix != ".part"), key=lambda f: f.stat().st_mtime, reverse=True)
    for old in files[KEEP_FILES:]:
        old.unlink(missing_ok=True)
    return f"{URL_PREFIX}{name}?v={int(st.st_mtime)}"


def media_src(path: str | Path, static: bool | None = None, static_dir: Path | None = None) -> tuple[str | None, str | None]:
    """(src для <video>, предупреждение). Выбор способа — по размеру и доступности статической раздачи."""
    p = Path(path)
    mb = p.stat().st_size / 1e6
    static = static_enabled() if static is None else static
    if mb <= EMBED_SMALL_MB or (not static and mb <= EMBED_MAX_MB):
        return f"data:{_mime(p)};base64,{base64.b64encode(p.read_bytes()).decode()}", None
    if static:
        return publish(p, static_dir), None
    return None, (f"копия видео {mb:.0f} МБ слишком велика, чтобы вшить её в страницу, а статическая раздача выключена: запустите интерфейс командой `sd ui` "
                  f"(или `streamlit run src/sd/ui/app.py --server.enableStaticServing true`)")
