"""Пути проекта и идентификаторы видео."""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from ._env import ROOT, env_path

# Каталоги по умолчанию лежат внутри проекта; на другой машине их можно вынести переменными SD_DATA / SD_EXTERNAL / SD_MODELS / SD_OUTPUTS / SD_LABELS (или файлом .env).
DATA = env_path("SD_DATA", ROOT / "data")
MODELS = env_path("SD_MODELS", ROOT / "models")
OUTPUTS = env_path("SD_OUTPUTS", ROOT / "outputs")
CONFIGS = ROOT / "configs"
DOCS = ROOT / "docs"
LABELS = env_path("SD_LABELS", ROOT / "labels")
STREAMS = env_path("SD_STREAMS", ROOT / "streams")   # имитация камер для мониторинга: папки <район>-<индекс>-<время начала>

# папки данных пользователя -> слабая (клип-уровня) метка
WEAK_LABEL_BY_FOLDER = {"курение": "smoking", "лжекурение": "fake"}
EXTERNAL = env_path("SD_EXTERNAL", ROOT / "data_external")   # скачанные вручную открытые датасеты: data_external/<источник>/<датасет>/<класс>/клип
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


# ------------------------------------------------------------------------------------------ переносимые пути
# В артефактах (мета запусков, журнал экспериментов, треки) пути хранятся не абсолютными, а «от корня»: `$DATA/курение/a.mp4`, `$OUTPUTS/runs/…`.
# Тогда скопированная на другую машину папка outputs/ открывается без правок, даже если данные лежат в другом месте.
def _roots() -> list[tuple[str, Path]]:
    return [("$EXTERNAL", EXTERNAL), ("$DATA", DATA), ("$OUTPUTS", OUTPUTS), ("$MODELS", MODELS), ("$LABELS", LABELS), ("$ROOT", ROOT)]


def portable(p: str | Path | None) -> str | None:
    """Путь → строка с токеном корня (`$DATA/…`) или абсолютная POSIX-строка, если путь вне известных корней."""
    if p is None or str(p) == "":
        return None
    q = Path(p)
    if not q.is_absolute():
        q = Path.cwd() / q
    for tok, root in _roots():
        for base in (root, root.resolve()):
            try:
                return f"{tok}/{q.relative_to(base).as_posix()}"
            except ValueError:
                continue
    return q.as_posix()


def repo_path(p: str | Path) -> Path:
    """Путь из профиля/конфига → рабочий путь этой машины. Относительные `models/…`, `data/…`, `outputs/…`, `labels/…` разворачиваются от СООТВЕТСТВУЮЩИХ каталогов (с учётом SD_MODELS и др.),
    остальные относительные — от корня проекта; абсолютные не трогаются. Иначе профиль со ссылкой `models/cycle/x` не находил бы пакет при `SD_MODELS` вне проекта."""
    q = Path(p)
    if q.is_absolute():
        return q
    first, rest = (q.parts[0] if q.parts else ""), q.parts[1:]
    for name, root in (("models", MODELS), ("data", DATA), ("data_external", EXTERNAL), ("outputs", OUTPUTS), ("labels", LABELS)):
        if first == name:
            return root.joinpath(*rest)
    return ROOT / q


def from_portable(s: str | Path | None) -> Path | None:
    """Обратное к `portable`. Понимает и «чужие» абсолютные пути (в т.ч. Windows `C:\\…` из старых артефактов): если файла нет, ищет по хвосту пути (1–3 последних
    компонента) в `data/` и `data_external/`."""
    if s is None or str(s) == "":
        return None
    t = str(s)
    for tok, root in _roots():
        if t.startswith(tok + "/"):
            rel = PurePosixPath(t[len(tok) + 1:])
            return root / rel if ".." not in rel.parts else root     # `$DATA/../../etc` не выводит за корень
    p = Path(t)
    if p.exists():
        return p
    parts = PureWindowsPath(t).parts if re.match(r"^[A-Za-z]:[\\/]", t) or "\\" in t else PurePosixPath(t).parts
    for k in (3, 2, 1):
        if len(parts) < k:
            continue
        for base in (DATA, EXTERNAL):
            cand = base.joinpath(*parts[-k:])
            if cand.exists():
                return cand
    return p
