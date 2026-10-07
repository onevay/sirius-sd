"""Подключение и проверка НОВЫХ моделей (в том числе принесённых с другой машины): безопасность, реестр пользователя, перенос пакетов, быстрые оценки.

Правила безопасности те же, что у остального проекта: чужой код не исполняется.
  * `.pt/.pth/.bin` (pickle) — только после статического сканирования (`safety`); `danger/error` — отказ, `unknown` — только с явным разрешением;
  * `.onnx`, `.safetensors` — форматы без исполнения кода, принимаются;
  * пакеты моделей проекта (photo/cycle) — только JSON/TXT/CSV/safetensors; в zip проверяются пути (нет выхода за каталог), размеры и sha256 из манифеста.
Пользовательские записи реестра лежат в `models/user_models.yaml` (не в `configs/`, поэтому не затираются обновлением проекта).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .paths import MODELS
from .safety import scan_torch_archive

USER_REGISTRY = MODELS / "user_models.yaml"
SAFE_BUNDLE_EXT = {".json", ".txt", ".csv", ".safetensors", ".md"}
NO_CODE_EXT = {".onnx", ".safetensors"}
PICKLE_EXT = {".pt", ".pth", ".bin", ".ckpt"}
KINDS = ("detector", "pose", "photo", "vlm", "tube")
MAX_MEMBER_MB = 400
BUNDLE_KINDS = {"photo": MODELS / "photo", "cycle": MODELS / "cycle"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def scan_model_path(path: str | Path) -> dict:
    """Вердикт безопасности файла/каталога: {verdict: ok|unknown|danger|error, format, notes}."""
    p = Path(path)
    if p.is_dir():
        reps = [scan_model_path(f) for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in PICKLE_EXT | NO_CODE_EXT | SAFE_BUNDLE_EXT]
        worst = max((r["verdict"] for r in reps), key=["ok", "unknown", "error", "danger"].index, default="ok")
        return dict(verdict=worst, format="dir", notes=[f"{r['file']}: {r['verdict']}" for r in reps if r["verdict"] != "ok"])
    ext = p.suffix.lower()
    if ext in PICKLE_EXT:
        rep = scan_torch_archive(p)
        return dict(verdict=rep.verdict, format="pickle", file=p.name, notes=[rep.summary()] + rep.notes)
    if ext in NO_CODE_EXT | SAFE_BUNDLE_EXT:
        return dict(verdict="ok", format=ext[1:], file=p.name, notes=["формат без исполнения кода"])
    return dict(verdict="unknown", format=ext[1:] or "?", file=p.name, notes=["неизвестный формат: не загружается"])


# ------------------------------------------------------------------------------------------ реестр пользователя
def load_user_models() -> list[dict]:
    if not USER_REGISTRY.exists():
        return []
    return (yaml.safe_load(USER_REGISTRY.read_text(encoding="utf-8")) or {}).get("models", [])


def register_user_model(spec: dict, copy_from: str | Path | None = None, allow_unknown: bool = False) -> dict:
    """Добавляет модель в реестр пользователя. `copy_from` — файл весов, который копируется в models/user/<id><ext> после проверки безопасности.

    Возвращает {spec, scan}. Бросает ValueError/RuntimeError при неверной схеме или небезопасных весах.
    """
    mid = str(spec.get("id", "")).strip()
    if not mid or not all(c.isalnum() or c in "-_." for c in mid):
        raise ValueError("id: только буквы, цифры, - _ .")
    if spec.get("kind") not in KINDS:
        raise ValueError(f"kind ∈ {KINDS}")
    scan = None
    spec = dict(spec)
    spec.setdefault("license", "?")
    spec["trust"] = "community"            # чужое — всегда community: сканирование и отказ при danger
    spec.setdefault("note", "пользовательская запись (models/user_models.yaml)")
    if copy_from:
        src = Path(copy_from)
        scan = scan_model_path(src)
        if scan["verdict"] in ("danger", "error") or (scan["verdict"] == "unknown" and not allow_unknown):
            raise RuntimeError(f"файл не прошёл проверку безопасности: {'; '.join(scan['notes'])}")
        dst = MODELS / "user" / f"{mid}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        spec["file"] = f"user/{dst.name}"
        spec["note"] += f"; sha256 {sha256_file(dst)[:16]}"
    models = [m for m in load_user_models() if m.get("id") != mid]
    models.append({k: v for k, v in spec.items() if v not in (None, "", [])})
    USER_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    USER_REGISTRY.write_text(yaml.safe_dump({"models": models}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dict(spec=spec, scan=scan)


def remove_user_model(mid: str, delete_file: bool = True) -> bool:
    models = load_user_models()
    keep = [m for m in models if m.get("id") != mid]
    if len(keep) == len(models):
        return False
    for m in models:
        if m.get("id") == mid and delete_file and m.get("file", "").startswith("user/"):
            (MODELS / m["file"]).unlink(missing_ok=True)
    USER_REGISTRY.write_text(yaml.safe_dump({"models": keep}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return True


# ------------------------------------------------------------------------------------------ перенос пакетов между машинами
def list_bundles() -> list[dict]:
    """Пакеты моделей проекта: models/photo/<имя>, models/cycle/<имя> (по manifest.json)."""
    out = []
    for kind, root in BUNDLE_KINDS.items():
        for d in sorted(root.glob("*/manifest.json")) if root.exists() else []:
            try:
                m = json.loads(d.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append(dict(kind=kind, name=d.parent.name, path=str(d.parent), created=m.get("created", ""), n=m.get("n"), features=len(m.get("features", [])) or None,
                            note=str(m.get("note", ""))[:80]))
    return out


def export_bundles(items: list[tuple[str, str]], out_zip: Path) -> dict:
    """items = [(kind, name)] → zip с MANIFEST.sha256.json; внутри только допустимые форматы."""
    files, manifest = [], {}
    for kind, name in items:
        root = BUNDLE_KINDS[kind] / name
        for f in sorted(root.rglob("*")):
            if f.is_file():
                if f.suffix.lower() not in SAFE_BUNDLE_EXT:
                    raise ValueError(f"{f}: формат {f.suffix} не допускается в переносимом пакете")
                rel = f"{kind}/{name}/{f.relative_to(root).as_posix()}"
                files.append((f, rel))
                manifest[rel] = sha256_file(f)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f, rel in files:
            z.write(f, rel)
        z.writestr("MANIFEST.sha256.json", json.dumps(manifest, indent=1))
    return manifest


def import_bundles(zip_path: Path, overwrite: bool = False) -> list[str]:
    """Проверенная распаковка пакетов: допустимые каталоги/расширения/размеры, sha256 совпадает с манифестом. Возвращает список импортированных «kind/name»."""
    # фаза 1: проверяем ВСЕ члены архива и ничего не пишем — при любой ошибке на диске не остаётся полуимпортированных пакетов
    payload: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if "MANIFEST.sha256.json" not in names:
            raise ValueError("в архиве нет MANIFEST.sha256.json — это не экспорт `export_bundles`")
        man = json.loads(z.read("MANIFEST.sha256.json").decode("utf-8"))
        total = 0
        for n in names:
            if n == "MANIFEST.sha256.json":
                continue
            parts = Path(n).parts
            if n.startswith(("/", "\\")) or ".." in parts or len(parts) < 3 or parts[0] not in BUNDLE_KINDS:
                raise ValueError(f"недопустимый путь в архиве: {n}")
            if Path(n).suffix.lower() not in SAFE_BUNDLE_EXT:
                raise ValueError(f"недопустимый формат файла в пакете: {n}")
            size = z.getinfo(n).file_size
            total += size
            if size > MAX_MEMBER_MB * 1e6 or total > 4 * MAX_MEMBER_MB * 1e6:
                raise ValueError(f"{n}: слишком большой файл/архив (лимит {MAX_MEMBER_MB} МБ на файл)")
            data = z.read(n)
            if hashlib.sha256(data).hexdigest() != man.get(n):
                raise ValueError(f"{n}: sha256 не совпадает с манифестом (файл изменён или повреждён)")
            payload[n] = data
    keys = sorted({"/".join(Path(n).parts[:2]) for n in payload})
    for key in keys:
        k, name = key.split("/")
        if (BUNDLE_KINDS[k] / name).exists() and not overwrite:
            raise FileExistsError(f"{BUNDLE_KINDS[k] / name} уже существует (overwrite=False)")
    # фаза 2: запись
    for key in keys:
        k, name = key.split("/")
        if (BUNDLE_KINDS[k] / name).exists():
            shutil.rmtree(BUNDLE_KINDS[k] / name)
    for n, data in payload.items():
        parts = Path(n).parts
        dst = BUNDLE_KINDS[parts[0]].joinpath(*parts[1:])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return keys


# ------------------------------------------------------------------------------------------ быстрые оценки
def auc_with_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray | None = None, n_boot: int = 300, seed: int = 0) -> tuple[float, float, float]:
    """AUC и 95% интервал бутстрэпа (по группам, если переданы — циклы одного видео зависимы)."""
    from sklearn.metrics import roc_auc_score

    y, s = np.asarray(y), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    g = np.asarray(groups)[ok] if groups is not None else np.arange(len(y))
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    pt = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ug}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        if len(np.unique(y[pick])) == 2:
            vals.append(roc_auc_score(y[pick], s[pick]))
    return (pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 20 else (pt, float("nan"), float("nan"))


def eval_cycle_bundle(bundle_dir: Path, tab: pd.DataFrame, n_boot: int = 300) -> dict:
    """Пакет классификатора цикла на размеченной таблице (`y`, `video`): AUC с интервалом по видео, какие признаки не найдены, оценки по циклам."""
    from .bundle import Bundle

    b = Bundle(bundle_dir)
    miss = b.missing(tab)
    s = b.score(tab)
    pt, lo, hi = auc_with_ci(tab.y.to_numpy(), s, tab.video.to_numpy(), n_boot)
    return dict(n=int(len(tab)), n_pos=int(tab.y.sum()), auc=pt, auc_lo=lo, auc_hi=hi, missing_features=miss, scores=s,
                note="если признаки модели обучали на тех же циклах, AUC завышен: честная оценка — только на циклах, которых не было в обучении")


def eval_photo_bundle(bundle_dir: Path, df: pd.DataFrame, backend: str = "auto", progress=None) -> dict:
    """Фото-пакет на таблице снимков (path, label[, cls, source]): AUC, recall/FPR при 0.5, AUC против каждого негативного класса."""
    from . import photo_clf as PC
    from . import photo_feats as PF

    bundle = PC.load_bundle(Path(bundle_dir))
    X = {bb: PF.embed_files(PF.make_embedder(bb, backend), df.path.tolist(), progress=(lambda i, n, dt, bb=bb: progress(bb, i, n)) if progress else None)
         for bb in bundle["manifest"]["backbones"]}
    ok = ~np.any([np.isnan(m).any(1) for m in X.values()], axis=0)
    s = np.full(len(df), np.nan)
    s[ok] = PC.score_bundle(bundle, {k: v[ok] for k, v in X.items()})
    d = df.assign(score=s)[ok]
    out = PC.summarize(d.label.to_numpy(int), d.score.to_numpy())
    if "cls" in d:
        out["per_negative"] = PC.per_negative_auc(d.assign(label=d.label.astype(int)), d.score.to_numpy()).to_dict("records")
    out["scores"] = s
    return out


def detect_images(model_path: str | Path, images: list[np.ndarray], conf: float = 0.15, imgsz: int = 320) -> list[list[dict]]:
    """YOLO-детектор (после проверки безопасности `.pt`) на списке BGR-картинок: [[{cls, conf, box}, …], …]."""
    from ultralytics import YOLO, settings

    p = Path(model_path)
    sc = scan_model_path(p)
    if sc["verdict"] in ("danger", "error", "unknown") and p.suffix.lower() in PICKLE_EXT:
        raise RuntimeError("веса не прошли проверку безопасности: " + "; ".join(sc["notes"]))
    settings.update({"sync": False})
    m = YOLO(str(p))
    out = []
    for im in images:
        r = m.predict(im, imgsz=imgsz, conf=conf, verbose=False, device="cpu")[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                dets.append(dict(cls=str(m.names[int(k)]), conf=float(c), box=b.tolist()))
        out.append(dets)
    return out
