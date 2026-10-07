"""Видео-разметчик событий — двусторонний компонент Streamlit без сборки фронтенда (`frontend/labeler/index.html`).

В компонент уходит копия видео (data-URI), длительность, треки людей и текущие интервалы; обратно приходят действия `add` / `update` / `delete` / `clean`
с уникальным `nonce` (повторная отправка того же действия после перерисовки игнорируется).
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit.components.v1 as components

FRONTEND = Path(__file__).parent / "frontend" / "labeler"
MAX_EMBED_MB = 120.0

_component = components.declare_component("sd_labeler", path=str(FRONTEND))


def video_data_uri(path: str | Path) -> tuple[str | None, str | None]:
    """(data-URI, предупреждение). Видео больше лимита не вшивается."""
    p = Path(path)
    mb = p.stat().st_size / 1e6
    if mb > MAX_EMBED_MB:
        return None, f"копия видео {mb:.0f} МБ больше {MAX_EMBED_MB:.0f} МБ — разметка без картинки; уменьшите высоту копии или разметьте по фрагментам"
    mime = "video/webm" if p.suffix.lower() == ".webm" else "video/mp4"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}", None


def labeler(*, key: str, video: str | None, video_key: str, duration: float, fps: float, src_size: tuple[int, int], intervals: list[dict], tracks: dict, clip_id: str,
            theme: str | None = None):
    """Рисует разметчик и возвращает последнее действие пользователя (dict) или None. `key` — ключ виджета Streamlit, `video_key` — идентификатор видео для компонента
    (при его смене компонент перезагружает ролик, при прочих изменениях — нет, воспроизведение не прерывается)."""
    return _component(key=key, default=None, video=video, vkey=video_key, duration=float(duration), fps=float(fps), src_w=int(src_size[0]), src_h=int(src_size[1]),
                      intervals=intervals, tracks=tracks, clip_id=clip_id, theme=theme)
