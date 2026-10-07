"""Загрузка выборки открытых ВИДЕО-наборов без входа в аккаунт в `data_external/<источник>/<набор>/<класс>/клип.mp4` (папка = класс — формат `sd clips-run`).

Пока: HMDB51 (зеркало Hugging Face `divm/hmdb51`, отдельные mp4; весь набор 868 МБ, нужные 6 классов ≈ 83 МБ). Нужен именно HMDB51, потому что в нём «курение» (`smoke`) и
жесты-двойники «рука ко рту» (`drink`, `eat`, `chew`) — из одного источника: в ваших видео затяжки и негативы из разных папок, и признаки частично выучивают сцену.
Не делает: не обходит входы/Cloudflare, не качает с YouTube (Kinetics, AVA — вручную, см. docs/NEXT_STEPS.md).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import csv
import random
import shutil
from pathlib import Path

from .paths import EXTERNAL

HMDB51_CLASSES = ("smoke", "drink", "eat", "chew", "talk", "smile")

VIDEO_SOURCES = {
    "hmdb51": dict(
        repo="divm/hmdb51", dest=("hmdb51", "hmdb51_org"), url="https://huggingface.co/datasets/divm/hmdb51", classes=HMDB51_CLASSES,
        license=("Карточка зеркала divm/hmdb51 на Hugging Face: license: other (у зеркала Serrelab/hmdb51 указано CC BY 4.0 — карточки расходятся). Оригинал — HMDB51 "
                 "(Kuehne et al., ICCV 2011, Brown University), клипы собраны из фильмов и YouTube: использовать для внутренней оценки, не распространять."),
    ),
}


def dest_dir(name: str, root: Path = EXTERNAL) -> Path:
    return root.joinpath(*VIDEO_SOURCES[name]["dest"])


def class_of(filename: str, classes: tuple[str, ...] | list[str]) -> str | None:
    """Класс HMDB51 по имени файла: `<класс>_<название ролика>_<класс>_<атрибуты>.mp4`. Длинные имена классов проверяются первыми (`brush_hair` раньше `brush`)."""
    base = Path(filename).name
    for c in sorted(classes, key=len, reverse=True):
        if base.startswith(c + "_"):
            return c
    return None


def select(files: list[str], classes: tuple[str, ...] | list[str], per_class: int | None = None, seed: int = 0) -> dict[str, list[str]]:
    """{класс: пути в репозитории}; не больше `per_class` на класс — детерминированная случайная выборка по отсортированному списку."""
    by: dict[str, list[str]] = {c: [] for c in classes}
    for f in sorted(files):
        if not f.lower().endswith(".mp4"):
            continue
        c = class_of(f, classes)
        if c is not None:
            by[c].append(f)
    rng = random.Random(seed)
    return {c: (sorted(rng.sample(v, per_class)) if per_class and len(v) > per_class else v) for c, v in by.items()}


def fetch(name: str = "hmdb51", classes: list[str] | None = None, per_class: int | None = None, seed: int = 0, root: Path = EXTERNAL, workers: int = 3, attempts: int = 5) -> dict:
    """Скачать отобранные клипы в `<dest>/<класс>/`, записать `LICENSE.txt` и `manifest.csv` (файл, класс, путь в зеркале)."""
    from huggingface_hub import HfApi, snapshot_download

    info = VIDEO_SOURCES[name]
    classes = list(classes or info["classes"])
    files = HfApi().list_repo_files(info["repo"], repo_type="dataset")
    chosen = select(files, classes, per_class, seed)
    paths = [p for v in chosen.values() for p in v]
    if not paths:
        raise RuntimeError(f"в {info['repo']} не найдено клипов классов {classes}")
    dest = dest_dir(name, root)
    tmp = dest / "_hf"
    tmp.mkdir(parents=True, exist_ok=True)
    err: Exception | None = None
    for _ in range(attempts):                     # обрывы соединения (на Windows часто 10053/10054 при многопоточной загрузке): докачка берёт только недостающее
        try:
            snapshot_download(info["repo"], repo_type="dataset", allow_patterns=paths, local_dir=tmp, max_workers=workers)
            err = None
            break
        except Exception as e:                    # noqa: BLE001 — сеть: повторяем, скачанное сохраняется
            err = e
    if err is not None:
        raise RuntimeError(f"загрузка не завершилась после {attempts} попыток ({type(err).__name__}); скачанное сохранено в {tmp}, повторите команду — докачает недостающее") from err
    rows = []
    for c, v in chosen.items():
        (dest / c).mkdir(parents=True, exist_ok=True)
        for p in v:
            src = tmp / p
            if src.exists():
                shutil.move(str(src), str(dest / c / Path(p).name))
                rows.append(dict(file=f"{c}/{Path(p).name}", cls=c, repo_path=p))
    shutil.rmtree(tmp, ignore_errors=True)
    with open(dest / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "cls", "repo_path"])
        w.writeheader()
        w.writerows(rows)
    (dest / "LICENSE.txt").write_text(f"{info['license']}\nИсточник: {info['url']}\n", encoding="utf-8")
    return dict(name=name, dest=str(dest), clips=len(rows), per_class={c: sum(1 for r in rows if r["cls"] == c) for c in classes}, missing=len(paths) - len(rows))
