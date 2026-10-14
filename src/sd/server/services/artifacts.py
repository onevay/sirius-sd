"""Файлы тревог (миниатюра, клип): путь из БД → реальный файл, только внутри каталога результатов (`outputs/`). Путь вне каталога считается отсутствующим."""
from __future__ import annotations

from pathlib import Path

from ... import paths as P
from ..models import Alert


def confined(portable_path: str | None) -> Path | None:
    p = P.from_portable(portable_path) if portable_path else None
    if not p or not Path(p).exists():
        return None
    try:
        Path(p).resolve().relative_to(P.OUTPUTS.resolve())
    except ValueError:
        return None
    return Path(p)


def file_for(a: Alert, kind: str) -> Path | None:
    return confined(a.thumb if kind == "thumb" else a.clip)


def flags(a: Alert) -> dict[str, bool]:
    return dict(has_thumb=file_for(a, "thumb") is not None, has_clip=file_for(a, "clip") is not None)
