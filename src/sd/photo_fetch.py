"""Загрузка публичных фото-датасетов (без входа в аккаунт) в `data_external/<источник>/<датасет>/`.

Каждый датасет получает `LICENSE.txt` (что известно о лицензии и откуда это известно) и `labels.csv`
(file, label, cls, split): `label` — 1 = курит, 0 = не курит; `cls` — исходный класс («drinking», «phoning»…),
чтобы потом смотреть качество отдельно на «трудных» негативах.

Что НЕ делает: не использует Roboflow/Kaggle/Ultralytics Platform (нужен вход, а их Cloudflare не обходим) — для них
в docs/NEXT_STEPS.md даны ссылки и раскладка, которую понимает `scan_photos`.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import csv
import hashlib
import random
import re
import zipfile
from pathlib import Path

from .paths import EXTERNAL

SMOKING_CLASS = "smoking"
# Действия Stanford-40, при которых рука/предмет у лица: именно их путают с курением
STANFORD_HARD = ("drinking", "phoning", "texting_message", "brushing_teeth", "blowing_bubbles", "taking_photos",
                 "playing_violin", "looking_through_a_telescope")

SOURCES = {
    "stanford40": dict(
        repo="zrchen03/Stanford40_Dataset", file="Stanford40.zip", dest=("stanford40", "stanford40_actions"),
        url="https://huggingface.co/datasets/zrchen03/Stanford40_Dataset",
        license=("Карточка зеркала на Hugging Face (загрузил zrchen03): MIT. Оригинал — Stanford 40 Actions "
                 "(Yao, Jiang, Khosla, Lin, Guibas, Fei-Fei; ICCV 2011), распространяется для исследовательских целей: "
                 "для коммерческого использования условия оригинала нужно проверить отдельно."),
    ),
    "mendeley_smoker_2400": dict(
        type="rar", url_file="https://data.mendeley.com/public-files/datasets/7b52hhzs3r/files/a3adcf62-1ef4-40cc-bd87-0111ac9260e2/file_downloaded",
        size_mb=651, dest=("mendeley", "smoking_vs_notsmoking_2400"), url="https://data.mendeley.com/datasets/7b52hhzs3r/1",
        license=("CC BY 4.0 (Mendeley Data, DOI 10.17632/7b52hhzs3r.1, Ali Khan, 2020-07-18): 2400 снимков, 1200 курящих и 1200 некурящих; в «некурящих» — похожие жесты: "
                 "вода, ингалятор, телефон, обкусывание ногтей. Нужно указывать авторство."),
    ),
    "cigdet": dict(
        type="yolo_zip", url_file="https://data.mendeley.com/public-files/datasets/6hyrr8typ7/files/24e3327e-155e-4565-ab44-8b45dc6445ee/file_downloaded",
        sha256="ef2ab181b62e93198654a333ae82b80bf3e08eafcc53e3c865340665936bae41", size_mb=35, names=["cigarette"], dest=("mendeley", "cigdet"),
        url="https://data.mendeley.com/datasets/6hyrr8typ7/1",
        license=("CC BY 4.0 (Mendeley Data, DOI 10.17632/6hyrr8typ7.1, Ali Khan, 2024-03-28): 557 снимков класса «курит» набора Smoker Detection (тот же автор) с рамками сигарет в формате YOLO "
                 "(train 446 / test 111). Нужно указывать авторство. Снимки пересекаются с положительными снимками «Smoker Detection», в наборе нет снимков без сигареты: "
                 "это тест ДЕТЕКТОРА предмета (`sd detector-eval --data … --split test`), а не фото-набор «курит / не курит» — labels.csv намеренно не создаётся."),
    ),
    "smoking_img_final_test": dict(
        repo="ccclllwww/smoking_img_final", patterns=["images/test/*"], dest=("huggingface", "ccclllwww_smoking_img_final_test"),
        url="https://huggingface.co/datasets/ccclllwww/smoking_img_final",
        license="В карточке лицензия не указана (публичный датасет без лицензии): только внутренняя оценка, не распространять.",
    ),
}


def dest_dir(name: str, root: Path = EXTERNAL) -> Path:
    return root.joinpath(*SOURCES[name]["dest"])


def _write_meta(dest: Path, name: str, rows: list[dict]) -> None:
    info = SOURCES[name]
    (dest / "LICENSE.txt").write_text(f"{info['license']}\nИсточник: {info['url']}\n", encoding="utf-8")
    with open(dest / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "cls", "split"])
        w.writeheader()
        w.writerows(rows)


def extract_stanford40(zip_path: Path, dest: Path, hard: tuple[str, ...] = STANFORD_HARD, easy_per_class: int = 8,
                       seed: int = 0, max_per_class: int | None = None) -> list[dict]:
    """Вынимает из zip класс `smoking`, «трудные» действия и немного случайных других (лёгкие негативы).

    Имена в Stanford-40: `<действие>_<номер>.jpg` (каталог внутри архива не важен).
    """
    pat = re.compile(r"^(?P<cls>.+?)_(?P<num>\d+)\.jpe?g$", re.I)
    rng = random.Random(seed)
    by_cls: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            m = pat.match(Path(n).name)
            if m and "__MACOSX" not in n:
                by_cls.setdefault(m["cls"].lower(), []).append(n)
        # официальные списки train/test, если они есть в архиве
        splits: dict[str, str] = {}
        for n in z.namelist():
            base = Path(n).name.lower()
            if base in ("train.txt", "test.txt"):
                for ln in z.read(n).decode("utf-8", "ignore").splitlines():
                    if ln.strip():
                        splits[Path(ln.strip()).name.lower()] = base.split(".")[0]
        wanted: dict[str, list[str]] = {SMOKING_CLASS: sorted(by_cls.get(SMOKING_CLASS, []))}
        for c in hard:
            wanted[c] = sorted(by_cls.get(c, []))
        others = sorted(c for c in by_cls if c != SMOKING_CLASS and c not in hard)
        for c in others:
            lst = sorted(by_cls[c])
            wanted[c] = rng.sample(lst, min(easy_per_class, len(lst)))
        rows: list[dict] = []
        for c, names in wanted.items():
            for n in names[:max_per_class] if max_per_class else names:
                out = dest / c / Path(n).name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(n))
                rows.append(dict(file=f"{c}/{Path(n).name}", label=int(c == SMOKING_CLASS), cls=c,
                                 split=splits.get(Path(n).name.lower(), "all")))
    return rows


YOLO_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
YOLO_SPLITS = {"train": "train", "test": "test", "val": "val", "valid": "val", "validation": "val"}


def extract_yolo_zip(zip_path: Path, dest: Path, names: list[str], max_file_mb: float = 20.0) -> dict:
    """Архив YOLO-датасета, где картинка и `.txt` лежат в одном каталоге (`train/a.jpg`, `train/a.txt`, как у CigDet), → раскладка Ultralytics/Roboflow
    `<split>/images`, `<split>/labels` и `data.yaml` (`val` = `test`, если отдельного val нет). Берутся только картинки и `.txt`; пути из архива не используются
    (только имя файла и имя сплита), файлы крупнее `max_file_mb` пропускаются."""
    import yaml

    counts: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or info.file_size > max_file_mb * 1e6:
                continue
            p = Path(info.filename)
            ext = p.suffix.lower()
            split = next((YOLO_SPLITS[x.lower()] for x in reversed(p.parts[:-1]) if x.lower() in YOLO_SPLITS), None)
            if split is None or ext not in YOLO_IMG_EXT + (".txt",):
                continue
            kind = "labels" if ext == ".txt" else "images"
            out = dest / split / kind / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
            counts.setdefault(split, dict(images=0, labels=0))[kind] += 1
    cfg: dict = {k: f"{k}/images" for k in ("train", "test") if k in counts}
    if "val" in counts:
        cfg["val"] = "val/images"
    elif "test" in counts:
        cfg["val"] = "test/images"
    cfg["names"] = {i: n for i, n in enumerate(names)}
    (dest / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dict(splits=counts, images=sum(c["images"] for c in counts.values()), boxes_files=sum(c["labels"] for c in counts.values()))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bsdtar() -> str:
    """Системный bsdtar (Windows 10/11: C:\\Windows\\System32\\tar.exe, libarchive читает RAR5/7z); GNU tar из Git RAR не умеет."""
    import shutil

    for c in (r"C:\Windows\System32\tar.exe", shutil.which("bsdtar") or "", shutil.which("7z") or ""):
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("нужен bsdtar (Windows: C:\\Windows\\System32\\tar.exe) или 7z для распаковки RAR")


def extract_images_from_archive(archive: Path, dest: Path, tool: str | None = None, exts=(".jpg", ".jpeg", ".png")) -> list[Path]:
    """Распаковывает архив во временный каталог, переносит в `dest` только картинки (всё остальное — в мусор), возвращает пути."""
    import shutil
    import subprocess
    import tempfile

    tool = tool or bsdtar()
    tmp = Path(tempfile.mkdtemp(prefix="sd_extract_", dir=dest))
    try:
        cmd = [tool, "-xf", str(archive), "-C", str(tmp)] if Path(tool).name.lower().startswith(("tar", "bsdtar")) else [tool, "x", str(archive), f"-o{tmp}", "-y"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"распаковка не удалась: {r.stderr[:300]}")
        moved = []
        for p in sorted(tmp.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                rel = p.relative_to(tmp)
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(out))
                moved.append(out)
        return moved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _download(url: str, dst: Path, progress=None) -> None:
    import httpx

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120, headers={"User-Agent": "sd-photos/0.1"}) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    tmp.replace(dst)


def fetch(name: str, root: Path = EXTERNAL, keep_zip: bool = False, progress=None, **kw) -> dict:
    """Скачать источник `name` (только публичные файлы, без входа в аккаунт) и разложить в data_external."""
    from huggingface_hub import hf_hub_download, snapshot_download

    info = SOURCES[name]
    dest = dest_dir(name, root)
    dest.mkdir(parents=True, exist_ok=True)
    if info.get("type") == "yolo_zip":
        arc = dest / "_download" / "archive.zip"
        if not arc.exists():
            _download(info["url_file"], arc, progress)
        if info.get("sha256") and _sha256(arc) != info["sha256"]:
            arc.unlink(missing_ok=True)
            raise RuntimeError("sha256 архива не совпал с указанным на странице источника: скачивание повредилось или файл на странице заменён — набор не использован")
        res = extract_yolo_zip(arc, dest, info["names"])
        if not keep_zip:
            arc.unlink(missing_ok=True)
            arc.parent.rmdir()
        (dest / "LICENSE.txt").write_text(f"{info['license']}\nИсточник: {info['url']}\n", encoding="utf-8")
        return dict(name=name, dest=str(dest), data_yaml=str(dest / "data.yaml"), **res)
    if info.get("type") == "rar":
        from .photo_data import label_from_path

        arc = dest / "_download" / "archive.rar"
        if not arc.exists():
            _download(info["url_file"], arc, progress)
        files = extract_images_from_archive(arc, dest)
        if not keep_zip:
            arc.unlink(missing_ok=True)
            arc.parent.rmdir()
        rows = []
        for p in files:
            rel = p.relative_to(dest).as_posix()
            lab = label_from_path(p, dest)
            rows.append(dict(file=rel, label="" if lab is None else lab, cls=p.parent.name, split="all"))
        rows = [r for r in rows if r["label"] != ""]    # без определимой метки в обучение не берём (их число видно по разнице с числом файлов)
        skipped = len(files) - len(rows)
        _write_meta(dest, name, rows)
        n_pos = sum(r["label"] for r in rows)
        return dict(name=name, dest=str(dest), images=len(rows), smoking=n_pos, not_smoking=len(rows) - n_pos, unlabeled_skipped=skipped)
    if name == "stanford40":
        tmp = dest / "_download"
        zp = Path(hf_hub_download(info["repo"], info["file"], repo_type="dataset", local_dir=tmp))
        rows = extract_stanford40(zp, dest, **kw)
        if not keep_zip:
            zp.unlink(missing_ok=True)
            for p in sorted(tmp.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            tmp.rmdir()
    else:
        snapshot_download(info["repo"], repo_type="dataset", allow_patterns=info.get("patterns"), local_dir=dest)
        rows = []
        for p in sorted(dest.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rel = p.relative_to(dest).as_posix()
                cls = p.parent.name
                rows.append(dict(file=rel, label=int(not re.search(r"not[_\- ]?smok", cls, re.I)), cls=cls,
                                 split="test" if "/test/" in f"/{rel}" else "all"))
    _write_meta(dest, name, rows)
    n_pos = sum(r["label"] for r in rows)
    return dict(name=name, dest=str(dest), images=len(rows), smoking=n_pos, not_smoking=len(rows) - n_pos)


def labels_from_yolo(root: Path, positive: set[str] | None = None) -> list[dict]:
    """YOLO-датасет (Roboflow «YOLOv8», Ultralytics Platform: `data.yaml`, `<split>/images`, `<split>/labels`) → `labels.csv` для фото-конвейера.

    Метка снимка: 1, если в нём есть хотя бы одна рамка нужного класса (`positive` — имена классов; по умолчанию любые), иначе 0. `cls` — классы рамок через «+»
    (`cigarette_in_hand+pen`) либо `background`. Внимание: снимок без рамок в детекционном наборе — это ФОН, а не «трудный негатив» (человек, пьющий воду, там
    чаще всего просто не размечен) — для проверки переноса на трудные случаи берите классы вида `pen`/`negative`, если авторы их выделили.
    """
    import yaml

    d = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8")) or {}
    names = d.get("names")
    names = list(names.values()) if isinstance(names, dict) else list(names or [])
    pos_ids = {i for i, n in enumerate(names) if positive is None or n in positive}
    rows: list[dict] = []
    for split, canon in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        im_dir, lb_dir = root / split / "images", root / split / "labels"
        if not im_dir.exists():
            continue
        for p in sorted(im_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            ids = []
            f = lb_dir / (p.stem + ".txt")
            if f.exists():
                for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = ln.split()
                    if len(parts) >= 5:
                        ids.append(int(float(parts[0])))
            cls = "+".join(sorted({names[i] if i < len(names) else str(i) for i in ids})) or "background"
            rows.append(dict(file=p.relative_to(root).as_posix(), label=int(any(i in pos_ids for i in ids)), cls=cls, split=canon))
    with open(root / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "cls", "split"])
        w.writeheader()
        w.writerows(rows)
    return rows
