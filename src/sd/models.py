"""Реестр и загрузка весов: статус, скачивание, sha256, проверка безопасности community-`.pt`."""
from __future__ import annotations

from . import _env  # noqa: F401

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .paths import CONFIGS, MODELS
from .safety import ScanReport, scan_torch_archive

LOCK = MODELS / "registry.lock.json"


@dataclass
class Spec:
    id: str
    kind: str
    license: str = "?"
    trust: str = "community"
    note: str = ""
    file: str | None = None
    url: str | None = None
    hf_repo: str | None = None
    hf_files: list[str] = field(default_factory=list)
    hf_patterns: list[str] = field(default_factory=list)
    rtmlib: dict | None = None

    @property
    def local_path(self) -> Path | None:
        """Куда кладём одиночный файл (для HF-репозиториев целиком — None, см. `hf_dir`)."""
        return MODELS / self.file if self.file else None


def load_registry(include_user: bool = True) -> dict[str, Spec]:
    """Реестр из configs/models.yaml + пользовательские записи models/user_models.yaml (их добавляет UI «Модели»; одноимённая запись пользователя не заменяет встроенную)."""
    raw = yaml.safe_load((CONFIGS / "models.yaml").read_text(encoding="utf-8"))
    reg = {m["id"]: Spec(**m) for m in raw["models"]}
    user = MODELS / "user_models.yaml"
    if include_user and user.exists():
        for m in (yaml.safe_load(user.read_text(encoding="utf-8")) or {}).get("models", []):
            if m["id"] not in reg:
                reg[m["id"]] = Spec(**m)
    return reg


def _read_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}


def _write_lock(d: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _download_url(url: str, dst: Path, progress=None) -> None:
    import httpx

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60, headers={"User-Agent": "sd-models/0.1"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    tmp.replace(dst)


def hf_dir(spec: Spec) -> Path | None:
    """Каталог снимка HF-репозитория в локальном кэше (если уже скачан)."""
    if not spec.hf_repo or spec.hf_files:
        return None
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(spec.hf_repo, allow_patterns=spec.hf_patterns or None, local_files_only=True))
    except Exception:
        return None


def status(spec: Spec) -> dict:
    lock = _read_lock().get(spec.id, {})
    p = spec.local_path
    present = bool(p and p.exists()) or bool(hf_dir(spec))
    size = p.stat().st_size / 1e6 if p and p.exists() else (
        sum(f.stat().st_size for f in hf_dir(spec).rglob("*") if f.is_file()) / 1e6 if hf_dir(spec) else 0)
    return dict(id=spec.id, kind=spec.kind, present=present, size_mb=round(size, 1), license=spec.license, trust=spec.trust,
                sha256=(lock.get("sha256") or "")[:12], scan=lock.get("scan", {}).get("verdict", "-"))


def _retry(fn, attempts: int = 6, base_sleep: float = 4.0):
    """Нестабильное соединение/прокси: повторяем с нарастающей паузой; huggingface_hub докачивает .incomplete."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # сетевые ошибки requests/httpx/hf_hub
            last = e
            time.sleep(base_sleep * (i + 1))
    raise last


def fetch(spec: Spec, progress=None) -> Path:
    """Скачивает веса. Возвращает путь к файлу/каталогу. Для community-.pt сразу сканирует."""
    lock = _read_lock()
    if spec.rtmlib:  # rtmlib качает сам при первом запуске
        return MODELS
    if spec.url:
        dst = spec.local_path
        if not dst.exists():
            _retry(lambda: _download_url(spec.url, dst, progress))
        path = dst
    elif spec.hf_repo and spec.hf_files:
        from huggingface_hub import hf_hub_download

        got = _retry(lambda: hf_hub_download(spec.hf_repo, spec.hf_files[0]))
        dst = spec.local_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes(Path(got).read_bytes())
        path = dst
    elif spec.hf_repo:
        from huggingface_hub import snapshot_download

        path = Path(_retry(lambda: snapshot_download(spec.hf_repo, allow_patterns=spec.hf_patterns or None)))
    else:
        raise ValueError(f"{spec.id}: нет url/hf_repo — положите файл вручную в models/{spec.file}")

    entry = lock.get(spec.id, {})
    entry.update(downloaded_at=time.strftime("%Y-%m-%d %H:%M:%S"), license=spec.license, trust=spec.trust)
    if path.is_file():
        entry["sha256"] = sha256_file(path)
        entry["size"] = path.stat().st_size
        if path.suffix in (".pt", ".pth"):
            rep = scan_torch_archive(path)
            entry["scan"] = dict(verdict=rep.verdict, globals=rep.globals_found, danger=rep.danger, unknown=rep.unknown)
    lock[spec.id] = entry
    _write_lock(lock)
    return path


def scan_status(spec_id: str) -> ScanReport | None:
    spec = load_registry()[spec_id]
    p = spec.local_path
    return scan_torch_archive(p) if p and p.exists() and p.suffix in (".pt", ".pth") else None


def local_weights(spec_id: str, allow_unknown: bool = False) -> Path:
    """Путь к весам для использования. Community-.pt отдаём только после успешного сканирования."""
    spec = load_registry()[spec_id]
    p = spec.local_path
    if p is None or not p.exists():
        raise FileNotFoundError(f"Веса «{spec_id}» не скачаны: sd models get {spec_id}")
    if spec.trust != "official" and p.suffix in (".pt", ".pth"):
        rep = scan_torch_archive(p)
        if rep.verdict not in ("ok",) and not (allow_unknown and rep.verdict == "unknown"):
            raise RuntimeError(f"Весам «{spec_id}» отказано в загрузке: {rep.summary()}")
    return p
