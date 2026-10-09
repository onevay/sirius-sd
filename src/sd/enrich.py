"""Дорогие признаки цикла (предмет, VLM) по готовым запускам `sd recognize`, с кэшем и перезапуском.

Для каждого клипа в `outputs/enrich/<клип>/`: `evidence_cycles.parquet` (предмет на кропах кисть–рот, колонки `obj_<детектор>_*`) и `vlm.parquet` (ответ VLM по кадрам рта, `score_yesno`).
Уже посчитанные циклы не пересчитываются (ключ video, tid, start): прогон можно прервать и продолжить, добавить новые циклы или детектор. Формат совпадает с каталогом `analysis/`
запуска, поэтому `feature_auc._attach_features` присоединяет признаки к таблице циклов без отдельного кода.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from . import analysis as A
from . import stages
from .calibrate import cfg_current_for_run
from .paths import OUTPUTS
from .tracks import Tracks

ROOT = OUTPUTS / "enrich"
KEY = ["video", "tid", "start"]


def clip_dir(clip: str, root: Path = ROOT) -> Path:
    return root / clip


def _keys(df: pd.DataFrame) -> set:
    return {(r.video, int(r.tid), round(float(r.start), 3)) for r in df.itertuples()} if len(df) else set()


def missing(cycles: pd.DataFrame, done: pd.DataFrame) -> pd.DataFrame:
    """Циклы, для которых признак ещё не посчитан."""
    have = _keys(done)
    keep = [(r.video, int(r.tid), round(float(r.start), 3)) not in have for r in cycles.itertuples()]
    return cycles[keep].reset_index(drop=True)


def evidence_signature(cfg: dict, detectors: Sequence[str]) -> str:
    """Отпечаток всего, от чего зависит результат детекторов предмета: параметры `evidence` (размер входа, порог, масштаб кропа, …), список детекторов и sha256 их весов."""
    from .config import stable_hash
    from .models import local_weights

    ws = {}
    for d in sorted(detectors):
        try:
            w = local_weights(d, allow_unknown=False)
            ws[d] = _sha(w)
        except Exception:   # noqa: BLE001 — нет весов: отпечаток без них (расчёт всё равно упадёт ниже)
            ws[d] = None
    ev = {k: v for k, v in dict(cfg["evidence"]).items() if k not in ("detectors", "device", "runtime")}      # где считается (CPU/iGPU/CUDA) на результат не влияет
    return stable_hash(dict(ev=ev, dets=ws), 12)


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


def evidence_slot(dest: Path, sig: str) -> Path:
    """Каталог кэша признаков предмета для отпечатка `sig`: основной (`dest`), если он свободен или уже принадлежит этому отпечатку, иначе `dest/alt_<отпечаток>`.
    Раньше кэш принадлежал тому, кто записал первым, и всё, посчитанное другой конфигурацией (другие детекторы, частота кадров), пересчитывалось при каждом запуске."""
    m = dest / "evidence_meta.json"
    if not m.exists() or not (dest / "evidence_cycles.parquet").exists():
        return dest
    try:
        if json.loads(m.read_text(encoding="utf-8")).get("signature") == sig:
            return dest
    except Exception:   # noqa: BLE001 — битый отпечаток: считаем каталог свободным для нового
        return dest
    return dest / f"alt_{sig}"


def cached_evidence(cycles: pd.DataFrame, cfg: dict, detectors: Sequence[str], dest: Path) -> pd.DataFrame | None:
    """Строки признаков предмета для ВСЕХ `cycles` из кэша `dest`, если параметры и веса совпали (`evidence_meta.json`) и покрыты все циклы и детекторы; иначе None."""
    dest = evidence_slot(dest, evidence_signature(cfg, detectors))
    f, m = dest / "evidence_cycles.parquet", dest / "evidence_meta.json"
    if not (f.exists() and m.exists()):
        return None
    try:
        if json.loads(m.read_text(encoding="utf-8")).get("signature") != evidence_signature(cfg, detectors):
            return None
        old = pd.read_parquet(f)
    except Exception:   # noqa: BLE001 — битый кэш = нет кэша
        return None
    if not all(f"obj_{d}_max_conf" in old.columns for d in detectors) or len(missing(cycles, old)):
        return None
    o = old.assign(start=old.start.round(3))
    c = cycles[["video", "tid", "start"]].assign(start=cycles.start.round(3))
    return c.merge(o, on=["video", "tid", "start"], how="left")


def save_evidence(rows: pd.DataFrame, cfg: dict, detectors: Sequence[str], dest: Path) -> None:
    """Кладёт строки в кэш; чужой кэш с другим отпечатком не затирается — для другой конфигурации заводится соседний слот `alt_<отпечаток>`."""
    sig = evidence_signature(cfg, detectors)
    dest = evidence_slot(dest, sig)
    m = dest / "evidence_meta.json"
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / "evidence_cycles.parquet"
    rows = rows.assign(start=rows.start.round(3))
    if f.exists() and m.exists():
        old = pd.read_parquet(f)
        keep = old.merge(rows[["video", "tid", "start"]], on=["video", "tid", "start"], how="left", indicator=True)
        rows = pd.concat([old[(keep._merge == "left_only").to_numpy()], rows], ignore_index=True)
    rows.to_parquet(f, index=False)
    m.write_text(json.dumps(dict(signature=sig, detectors=sorted(detectors), created=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False), encoding="utf-8")


def enrich_objects(cycles: pd.DataFrame, rd: Path, detectors: Sequence[str], dest: Path, cfg: dict | None = None) -> int:
    """Предмет по циклам одного запуска; возвращает число новых циклов. Если параметры/веса изменились (другой отпечаток), кэш пересчитывается целиком."""
    cfg = cfg or cfg_current_for_run(rd)
    sig = evidence_signature(cfg, detectors)
    legacy_f = dest / "evidence_cycles.parquet"
    if legacy_f.exists() and not (dest / "evidence_meta.json").exists():       # файл до появления отпечатков: принадлежит основному слоту
        pass
    else:
        dest = evidence_slot(dest, sig)
    f, m = dest / "evidence_cycles.parquet", dest / "evidence_meta.json"
    legacy = f.exists() and not m.exists()      # файл посчитан до появления отпечатка (тот же код и конфиг по умолчанию): принимаем и помечаем
    if legacy:
        dest.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps(dict(signature=sig, detectors=sorted(detectors), created="legacy", note="отпечаток проставлен задним числом"), ensure_ascii=False), encoding="utf-8")
    same = m.exists() and json.loads(m.read_text(encoding="utf-8")).get("signature") == sig
    old = pd.read_parquet(f) if (f.exists() and same) else pd.DataFrame()
    cols_have = {d for d in detectors if f"obj_{d}_max_conf" in old.columns}
    todo = missing(cycles, old) if cols_have == set(detectors) else cycles
    if todo.empty:
        return 0
    tr = Tracks.load(rd / "pose")
    ser = stages.stage_features(tr, cfg, rd)
    new = pd.DataFrame(A.evidence_rows(todo, tr, ser, cfg, list(detectors)))
    new["start"] = new.start.round(3)
    if cols_have == set(detectors) and len(old):
        new = pd.concat([old, new], ignore_index=True)
    dest.mkdir(parents=True, exist_ok=True)
    new.to_parquet(f, index=False)
    m.write_text(json.dumps(dict(signature=sig, detectors=sorted(detectors), created=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False), encoding="utf-8")
    return len(todo)


def enrich_vlm(cycles: pd.DataFrame, rd: Path, model: str, dest: Path, progress: Callable[[int, int], None] | None = None) -> int:
    """VLM («курят ли?» по 6 кадрам рта и кисти) по циклам одного запуска; `run_vlm` дописывает результат после каждого цикла."""
    from .vlm import OllamaVLM
    from .vlm_eval import run_vlm

    f = dest / "vlm.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    todo = missing(cycles, old[old.ok.astype(bool)] if len(old) and "ok" in old else old)
    if todo.empty:
        return 0
    tr = Tracks.load(rd / "pose")
    cfg = cfg_current_for_run(rd)
    ser = stages.stage_features(tr, cfg, rd)
    dest.mkdir(parents=True, exist_ok=True)
    run_vlm(todo, {"yesno": OllamaVLM(model, size=224, mode="yesno")}, f, 6, 2.5, progress, "mouth", 2.5, 224, ctx={str(rd): (tr, cfg, ser)})
    return len(todo)


def photo_signature(bundle: Path) -> str:
    """Отпечаток фото-признаков: содержимое пакета фото-модели (манифест и веса) и параметры кропов (6 кадров, окно 2.5 с, масштаб 2.5)."""
    import hashlib

    h = hashlib.sha256()
    for fn in sorted(Path(bundle).glob("*.json")):
        h.update(fn.name.encode())
        h.update(fn.read_bytes())
    h.update(b"frames6;window2.5;scale2.5;zs=smoke,none,drink,phone")
    return h.hexdigest()[:12]


def cached_photo(cycles: pd.DataFrame, bundle: Path, dest: Path) -> pd.DataFrame | None:
    """Фото-признаки ВСЕХ `cycles` из кэша клипа, если отпечаток пакета совпал и покрыты все циклы; иначе None."""
    f, m = dest / "photo_cycles.parquet", dest / "photo_meta.json"
    if not (f.exists() and m.exists()):
        return None
    try:
        if json.loads(m.read_text(encoding="utf-8")).get("signature") != photo_signature(bundle):
            return None
        old = pd.read_parquet(f)
    except Exception:   # noqa: BLE001
        return None
    if "photo_zsd_mean" not in old.columns or len(missing(cycles, old)):
        return None
    c = cycles[["video", "tid", "start"]].assign(start=cycles.start.round(3))
    return c.merge(old.assign(start=old.start.round(3)), on=["video", "tid", "start"], how="left")


def _stamp_photo(dest: Path, bundle: Path, note: str = "") -> None:
    (dest / "photo_meta.json").write_text(json.dumps(dict(signature=photo_signature(bundle), bundle=str(bundle), created=time.strftime("%Y-%m-%d %H:%M:%S"), note=note), ensure_ascii=False), encoding="utf-8")


def enrich_photo(cycles: pd.DataFrame, rd: Path, bundle: Path, dest: Path, backend: str = "auto", progress=None) -> int:
    """Фото-модель и zero-shot CLIP по кропам рта циклов клипа (`photo_cycles.parquet`: photo_p_*, photo_zs_*, photo_zsd_*). Считается заново только если появились новые циклы."""
    f, m = dest / "photo_cycles.parquet", dest / "photo_meta.json"
    if f.exists() and not m.exists():          # посчитано до появления отпечатка тем же пакетом и кодом: принимаем и помечаем
        _stamp_photo(dest, Path(bundle), "отпечаток проставлен задним числом")
    same = m.exists() and json.loads(m.read_text(encoding="utf-8")).get("signature") == photo_signature(bundle)
    old = pd.read_parquet(f) if (f.exists() and same) else pd.DataFrame()
    have_cols = len(old) and "photo_zsd_mean" in old.columns
    todo = missing(cycles, old) if have_cols else cycles
    if todo.empty:
        return 0
    tr = Tracks.load(rd / "pose")
    cfg = cfg_current_for_run(rd)
    ser = stages.stage_features(tr, cfg, rd)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "photo_new.parquet"
    new = A.photo_all(todo, Path(bundle), backend, out=tmp, progress=progress, ctx={str(rd): (tr, cfg, ser)})
    tmp.unlink(missing_ok=True)
    pd.concat([old, new], ignore_index=True).to_parquet(f, index=False) if have_cols else new.to_parquet(f, index=False)
    _stamp_photo(dest, Path(bundle))
    return len(todo)


def enrich_clips(clips: dict, objects: Sequence[str] = (), vlm_model: str | None = None, root: Path = ROOT,
                 progress: Callable[[str, int, int, str], None] | None = None, photo_bundle: Path | None = None) -> pd.DataFrame:
    """Прогон по клипам (`oof_eval.collect()`): таблица clip, циклов, новых по предмету/VLM, секунды. Ошибка клипа не останавливает остальных."""
    rows = []
    for i, (name, c) in enumerate(clips.items()):
        t0 = time.perf_counter()
        row = dict(clip=name, cycles=0, new_objects=0, new_vlm=0, new_photo=0, error="")
        try:
            cyc = pd.read_parquet(c.out_dir / "analysis" / "cycles_all.parquet")
            cyc["start"] = cyc.start.round(3)
            row["cycles"] = len(cyc)
            dest = clip_dir(name, root)
            if objects:
                row["new_objects"] = enrich_objects(cyc, c.run_dir, objects, dest)
            if photo_bundle:
                row["new_photo"] = enrich_photo(cyc, c.run_dir, photo_bundle, dest)
            if vlm_model:
                row["new_vlm"] = enrich_vlm(cyc, c.run_dir, vlm_model, dest, (lambda a, b, n=name: progress("VLM", a, b, n)) if progress else None)
        except Exception as e:   # noqa: BLE001 — один клип не должен останавливать прогон
            row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        row["sec"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        if progress:
            progress("клип", i + 1, len(clips), name)
    return pd.DataFrame(rows)


def attach(tab: pd.DataFrame, clips: Sequence[str], root: Path = ROOT, vlm: bool = True) -> pd.DataFrame:
    """Присоединяет предмет (`obj_any_*`) и VLM (`vlm_yesno`) к таблице циклов по ключу (video, tid, start); клипы без данных остаются без этих колонок.
    Признаки самого запуска (`analysis/` каталога запуска — `oof_eval.collect`) главнее кэша: кэш `enrich` принадлежит тому, кто записал его первым, и может быть посчитан другой
    конфигурацией (другие детекторы, частота кадров) — подмешивать его поверх признаков запуска значит учить классификатор на чужих циклах. Кэш подставляется только клипам, у которых
    признаков предмета в запуске нет вовсе."""
    from . import feature_auc as FA

    stale = [c for c in tab.columns if c.startswith(("obj_", "photo_", "vlm_", "xclip_", "videomae_")) or c == "n_det_frames"]
    parts = []
    for name in clips:
        t = tab[tab.video == name]
        d = clip_dir(name, root)
        if t.empty:
            continue
        own = "obj_any_max_conf" in t.columns and bool(t["obj_any_max_conf"].notna().any())      # признаки предмета этого запуска уже в таблице
        if not own:
            t = t.drop(columns=[c for c in stale if c in t.columns])
            v = d / "vlm.parquet" if (vlm and (d / "vlm.parquet").exists()) else None
            if (d / "evidence_cycles.parquet").exists() or v is not None:
                t = FA._attach_features(t.assign(start=t.start.round(3)), d, v, only_vlm=False)
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else tab
