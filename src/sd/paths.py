"""Пути проекта и идентификаторы видео."""
from __future__ import annotations

import re
from pathlib import Path

from ._env import ROOT

DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
CONFIGS = ROOT / "configs"
DOCS = ROOT / "docs"
LABELS = ROOT / "labels"

# папки данных пользователя -> слабая (клип-уровня) метка
WEAK_LABEL_BY_FOLDER = {"курение": "smoking", "лжекурение": "fake"}
EXTERNAL = ROOT / "data_external"   # скачанные вручную открытые датасеты: data_external/<источник>/<датасет>/<класс>/клип
# имена папок-классов внешних датасетов, означающие «курение» (HMDB51 `smoke`, Kinetics `smoking`); регистр, пробелы и дефисы не важны
SMOKING_CLASSES = {"smoking", "smoke", "smoking_hookah", "smoking_pipe", "курение"}
VIDEO_EXT = {".mp4", ".mkv", ".wmv", ".avi", ".mov", ".m4v", ".webm", ".ts"}


def slug(text: str) -> str:
    """Безопасное для файловой системы имя (кириллица сохраняется)."""
    return re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE).strip("_")


def external_parts(path: str | Path) -> list[str] | None:
    """Части пути ВНУТРИ `data_external/` (источник, датасет, класс..., имя без расширения) или None, если файл лежит не там."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(EXTERNAL.resolve())
    except (ValueError, OSError):
        return None
    return [*rel.parts[:-1], p.stem]


def video_id(path: str | Path) -> str:
    """`<папка>__<имя без расширения>` — уникальный и читаемый id видео. Для `data_external/` — `<датасет>__<класс>__<имя>`:
    одинаковые имена классов и файлов в разных датасетах не должны склеиваться в один запуск."""
    p = Path(path)
    ext = external_parts(p)
    if ext:
        return slug("__".join(ext[-3:]))
    parent = p.parent.name if p.parent.name and p.parent != p.parent.parent else ""
    return slug(f"{parent}__{p.stem}" if parent else p.stem)


def class_label(name: str, positive: set[str] | None = None) -> str:
    """Имя папки-класса -> `smoking` (курение) или `fake` (любой другой класс: питьё, еда, телефон…)."""
    key = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return "smoking" if key in (positive or SMOKING_CLASSES) else "fake"


def weak_label(path: str | Path) -> str | None:
    """Метка уровня клипа по имени папки данных (`smoking` / `fake`) или None. Во внешних датасетах класс = имя папки с клипом."""
    p = Path(path)
    lab = WEAK_LABEL_BY_FOLDER.get(p.parent.name)
    if lab:
        return lab
    return class_label(p.parent.name) if external_parts(p) else None


def list_videos(root: str | Path | None = None) -> list[Path]:
    """Все видео под `root` (по умолчанию `data/`), отсортированы по (папка, имя)."""
    base = Path(root) if root else DATA
    if base.is_file():
        return [base]
    return sorted((p for p in base.rglob("*") if p.suffix.lower() in VIDEO_EXT), key=lambda p: (p.parent.name, p.name))


def resolve_video(arg: str | Path) -> Path:
    """Принимает путь к видео ИЛИ имя файла/stem внутри data/ (удобно в CLI: `sd info sm_3`)."""
    p = Path(arg)
    if p.exists():
        return p.resolve()
    cands = [v for v in list_videos() if v.name == str(arg) or v.stem == str(arg) or video_id(v) == str(arg)]
    if len(cands) == 1:
        return cands[0].resolve()
    if len(cands) > 1:
        raise FileNotFoundError(f"«{arg}» неоднозначно, подходит: " + ", ".join(video_id(c) for c in cands))
    raise FileNotFoundError(f"Видео не найдено: {arg}")
