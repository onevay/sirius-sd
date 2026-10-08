"""Каталоги с видео для выбора в интерфейсе и CLI: «оценка только на выбранных папках».

Папка считается набором, если видео лежат в ней НЕПОСРЕДСТВЕННО (`data/курение`, `data_external/hmdb51/<набор>/smoke`). Длительности здесь не читаются
(это декодирование каждого файла) — только список файлов; метаданные берёт раннер при прогоне.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import DATA, EXTERNAL, ROOT, VIDEO_EXT, video_id, weak_label


def _is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXT


def videos_in_dir(d: str | Path) -> list[Path]:
    return sorted((p for p in Path(d).iterdir() if _is_video(p)), key=lambda p: p.name) if Path(d).is_dir() else []


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "outputs", "models", ".ultralytics"}


def _walk_dirs(root: Path, max_depth: int = 5):
    """Папки под `root` с ограничением глубины и без служебных каталогов: путь, который пользователь добавил по ошибке (например, корень диска), не должен подвесить интерфейс."""
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        yield d
        if depth >= max_depth:
            continue
        try:
            subs = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in SKIP_DIRS]
        except OSError:
            continue
        stack.extend((p, depth + 1) for p in subs)


def discover_dirs(roots: Iterable[Path] | None = None, extra: Iterable[str | Path] = ()) -> pd.DataFrame:
    """Все папки, содержащие видео напрямую. Колонки: path, name (путь от корня проекта), clips, weak (метка по имени папки или '')."""
    seen: dict[Path, int] = {}
    for root in [*(roots if roots is not None else (DATA, EXTERNAL)), *map(Path, extra)]:
        if not root.is_dir():
            continue
        for d in _walk_dirs(root):
            n = len(videos_in_dir(d))
            if n:
                seen[d.resolve()] = n
    rows = []
    for d, n in sorted(seen.items(), key=lambda kv: str(kv[0])):
        first = videos_in_dir(d)[0]
        try:
            name = str(d.relative_to(ROOT.resolve()))
        except ValueError:
            name = str(d)
        rows.append(dict(path=str(d), name=name, clips=n, weak=weak_label(first) or ""))
    return pd.DataFrame(rows, columns=["path", "name", "clips", "weak"])


def videos_in(dirs: Iterable[str | Path]) -> list[Path]:
    """Видео выбранных папок (без дублей, порядок: папка, имя)."""
    out, seen = [], set()
    for d in dirs:
        for v in videos_in_dir(d):
            if v.resolve() not in seen:
                seen.add(v.resolve())
                out.append(v)
    return out


def clip_ids(videos: Iterable[Path]) -> dict[str, Path]:
    """clip_id → путь. При совпадении id (разные папки с одинаковыми именами) бросает ValueError: такие клипы нельзя оценивать вместе."""
    out: dict[str, Path] = {}
    for v in videos:
        cid = video_id(v)
        if cid in out and out[cid].resolve() != v.resolve():
            raise ValueError(f"clip_id «{cid}» совпал у {out[cid]} и {v}: переименуйте один из файлов")
        out[cid] = v
    return out
