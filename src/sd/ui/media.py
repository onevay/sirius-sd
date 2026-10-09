"""Как отдать видео в браузер из Streamlit: файлом через менеджер медиафайлов Streamlit (`/media/…`, с перемоткой Range), а вне Streamlit или при сбое — data-URI до разумного размера.

Менеджер медиа встроен в Streamlit и не требует ни флагов запуска, ни записи в каталог пакета — работает одинаково при `sd ui`, `streamlit run` и в Docker. Крупный файл в base64 в страницу
не вшивается: браузер получает его по ссылке (раньше копии больше нескольких МБ вшивались и «видео нет»). Файл удаляется менеджером, когда сессия браузера закрывается.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

EMBED_MAX_MB = float(os.environ.get("SD_EMBED_MAX_MB", "70"))     # запасной путь (без Streamlit): больше этого не вшиваем


def _mime(p: Path) -> str:
    return "video/webm" if p.suffix.lower() == ".webm" else "video/mp4"


def _managed(p: Path) -> str | None:
    """URL файла в менеджере медиа Streamlit или None, если мы вне работающего приложения."""
    try:
        from streamlit.runtime import get_instance
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            return None
        return get_instance().media_file_mgr.add(str(p), _mime(p), coordinates=f"sd-video-{p.stem}")
    except Exception:   # noqa: BLE001 — внутреннее API Streamlit: при любой несовместимости уходим на запасной путь
        return None


def media_src(path: str | Path) -> tuple[str | None, str | None]:
    """(src для <video>, предупреждение)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None, f"файл видео не найден или пуст: {p}"
    url = _managed(p)
    if url:
        return url, None
    mb = p.stat().st_size / 1e6
    if mb <= EMBED_MAX_MB:
        return f"data:{_mime(p)};base64,{base64.b64encode(p.read_bytes()).decode()}", None
    return None, f"видео {mb:.0f} МБ нельзя вшить в страницу вне приложения Streamlit; откройте его отдельно: {p}"
