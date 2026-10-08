"""Решатель — полный пайплайн как переносимый пакет: профиль + классификатор цикла + (по желанию) веса и набор разметки, с метаданными, по которым архитектура определяется автоматически.

    sd solver pack  -p final --with-weights --with-dataset --run <id эксперимента> -o final.sdsolver.zip
    sd solver inspect final.sdsolver.zip          # что внутри: стадии, модели, требования, метрики, целостность
    sd solver install final.sdsolver.zip          # на другой машине: пакеты в models/, профиль в configs/experiments/
    sd run --solver final.sdsolver.zip --input … --out preds.csv

Состав архива (`*.sdsolver.zip`):
    solver.json            метаданные (схема 1): архитектура по стадиям, компоненты с sha256, требования, оценка качества, список файлов с sha256
    profile.yaml           профиль эксперимента (читается человеком и `sd eval -p`)
    bundles/<kind>/<имя>/  пакеты классификатора цикла и фото-модели (только JSON/TXT/CSV/safetensors, без pickle)
    weights/               веса (по флагу --with-weights): onnx / safetensors / .pt (последние сканируются при установке)
    prompts/               промпт VLM
    dataset/               (по флагу --with-dataset) events_gt.csv, cycle_labels.csv, clips_index.csv — набор, на котором получена оценка
    evaluation/            (по флагу --run) report.json эксперимента

Безопасность установки как у остального проекта: чужой код не исполняется — проверка путей (нет `..`/абсолютных), белый список расширений по разделам, размеры, sha256 каждого файла против solver.json,
`.pt` — статическим сканером `safety` (danger/error/unknown → отказ без явного разрешения), установка двухфазная (в каталог моделей ничего не пишется, пока всё не проверено).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import hashlib
import json
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import profiles as PR
from .paths import LABELS, MODELS, repo_path

SCHEMA = 1
EXT = ".sdsolver.zip"
BUNDLE_EXT = {".json", ".txt", ".csv", ".safetensors", ".md"}
WEIGHT_EXT = {".onnx", ".safetensors", ".pt", ".pth"}
PICKLE_EXT = {".pt", ".pth"}
MAX_FILE_MB = 2000

# ------------------------------------------------------------------------------------------ группы признаков классификатора
GROUP_RULES = [("object", re.compile(r"^obj_")), ("photo", re.compile(r"^(photo_|zs_)|_zs")), ("vlm", re.compile(r"^vlm_")), ("tube", re.compile(r"^(xclip_|videomae_)")),
               ("pose", re.compile(r"^p_"))]
OBJ_DET = re.compile(r"^obj_(?P<det>.+)_(max_conf|hit_frames|hit)$")


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    """Признаки классификатора по источникам: kinematics (поза, пауза, ритм — всегда есть) / pose (p_*) / object / photo / vlm / tube."""
    out: dict[str, list[str]] = {}
    for f in features:
        g = next((name for name, rx in GROUP_RULES if rx.search(f)), "kinematics")
        out.setdefault(g, []).append(f)
    return out


def required_detectors(features: list[str]) -> list[str]:
    """Детекторы предмета, нужные признакам `obj_<детектор>_*` (агрегат `obj_any_*` детектор не называет)."""
    return sorted({m.group("det") for f in features if (m := OBJ_DET.match(f)) and m.group("det") != "any"})


def describe_bundle(path: str | Path) -> dict:
    """Краткая сводка пакета классификатора по manifest.json."""
    p = repo_path(path)
    m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    feats = m.get("features", [])
    groups = feature_groups(feats)
    cv = m.get("cv") or {}
    return dict(name=p.name, path=str(path), format=m.get("format"), created=m.get("created"), n_features=len(feats), groups={k: len(v) for k, v in groups.items()},
                members={n: v.get("kind") for n, v in (m.get("members") or {}).items()}, calibrated=bool(m.get("calibrated")), n=m.get("n"), n_pos=m.get("n_pos"),
                n_groups=m.get("n_groups"), auc_oof=cv.get("auc_ensemble_calibrated") or cv.get("auc_ensemble"), requires=dict(
                    objects=required_detectors(feats), photo="photo" in groups, vlm="vlm" in groups, tube="tube" in groups))


# ------------------------------------------------------------------------------------------ проверка совместимости
@dataclass
class Issue:
    level: str          # error | warn
    code: str
    text: str

    def to_dict(self) -> dict:
        return dict(level=self.level, code=self.code, text=self.text)


MODES = ("offline", "replay", "live")      # offline — прогон файла целиком; replay — имитация потока по файлу (тяжёлые признаки читают исходный файл); live — настоящий поток


def check(profile: PR.Profile, mode: str = "offline") -> list[Issue]:
    """Проверка, что профиль собирает рабочий пайплайн с классификатором. Ошибки (`error`) мешают запуску; предупреждения — нет.

    Главное правило: классификатор цикла обязателен; его признаки должны быть обеспечены экстракторами профиля (иначе недостающие признаки молча станут пропусками
    и качество упадёт незаметно); VLM, предмет и фото-модель — надстройка над классификатором через `fusion` или каскад `cycle_bundle_full`, вектор признаков классификатора они не меняют."""
    if mode not in MODES:
        raise ValueError(f"mode ∈ {MODES}")
    issues: list[Issue] = []
    bundle_vlm = False          # вектор классификатора уже включает ответы VLM (тогда VLM нужен ему самому)
    o = profile.opts()
    problem = profile.classifier_problem()
    if problem:
        return [Issue("error", "classifier_required", problem)]
    if not o["cycle_bundle"]:
        issues.append(Issue("warn", "heuristic", "классификатор не выбран, оценка цикла — эвристика (allow_heuristic): только для отладки"))
    else:
        bp = repo_path(o["cycle_bundle"])
        if not (bp / "manifest.json").exists():
            return [Issue("error", "classifier_missing", f"пакет классификатора не найден: {o['cycle_bundle']} (положите его в models/cycle/ или установите решатель: sd solver install)")]
        try:
            d = describe_bundle(o["cycle_bundle"])
        except Exception as e:
            return [Issue("error", "classifier_broken", f"пакет {o['cycle_bundle']} не читается: {e}")]
        req = d["requires"]
        bundle_vlm = bool(req["vlm"])
        have = set(o["objects"] or [])
        lack = [x for x in req["objects"] if x not in have]
        if lack:
            issues.append(Issue("error", "objects_missing", f"классификатор использует признаки предмета детекторов {lack}, а в профиле их нет: признаки станут пропусками, качество упадёт. Добавьте детекторы в options.objects"))
        if req["photo"] and not o["photo_bundle"]:
            issues.append(Issue("error", "photo_missing", "классификатор использует признаки фото-модели, а photo_bundle в профиле не задан"))
        if req["vlm"] and not (o["vlm_model"] and o["vlm_mode"] == "all"):
            issues.append(Issue("error", "vlm_required", "признаки VLM входят в вектор классификатора: нужен vlm_model и vlm_mode=all (VLM считается для каждого цикла). Либо возьмите базовый классификатор без VLM и подключите VLM через fusion"))
        if req["tube"]:
            issues.append(Issue("error", "tube_unsupported", "классификатор использует признаки видео-моделей (X-CLIP/VideoMAE): в `recognize` и потоке они не считаются"))
        if mode == "live" and (req["objects"] or req["photo"] or req["vlm"]):
            issues.append(Issue("error", "live_features", "в настоящем потоке доступны только признаки позы и паузы: пакет требует " + ", ".join(
                x for x, on in (("предмет", bool(req["objects"])), ("фото-модель", req["photo"]), ("VLM", req["vlm"])) if on) + " — используйте replay по файлу или классификатор на быстрых признаках"))
        elif mode == "replay" and req["vlm"]:
            issues.append(Issue("error", "replay_vlm", "признаки VLM в имитации потока не считаются (нужна очередь VLM); используйте классификатор без VLM"))
    try:
        from .fusion import FusionSpec

        spec = FusionSpec.from_dict(o["fusion"])
    except Exception as e:
        return issues + [Issue("error", "fusion_invalid", f"fusion: {e}")]
    w = spec.weights
    if w.get("vlm", 0) > 0 and not (o["vlm_model"] and o["vlm_mode"] != "off"):
        issues.append(Issue("warn", "fusion_vlm_inactive", "вес VLM в fusion задан, но VLM выключен: поправки не будет"))
    if w.get("object", 0) > 0 and not o["objects"]:
        issues.append(Issue("warn", "fusion_object_inactive", "вес предмета в fusion задан, но детекторы не выбраны"))
    if w.get("photo", 0) > 0 and not o["photo_bundle"]:
        issues.append(Issue("warn", "fusion_photo_inactive", "вес фото-модели в fusion задан, но photo_bundle не выбран"))
    if spec.active and o["cycle_bundle_full"]:
        issues.append(Issue("warn", "fusion_and_full", "заданы и fusion, и cycle_bundle_full: VLM будет учтён дважды (каскад и поправка) — оставьте один способ"))
    if o["vlm_model"] and o["vlm_mode"] != "off" and not (spec.active or o["cycle_bundle_full"] or bundle_vlm):
        issues.append(Issue("warn", "vlm_unused", "VLM включён, но его ответы нигде не используются (нет fusion и cycle_bundle_full): считается впустую"))
    if spec.active and mode in ("replay", "live"):
        issues.append(Issue("error", "fusion_stream", "fusion пока не применяется в потоке и replay (только offline/eval/sd.run): результат отличался бы от оценки. Отключите fusion или используйте offline"))
    if mode == "live" and o["vlm_model"] and o["vlm_mode"] != "off":
        issues.append(Issue("warn", "live_vlm", "VLM в настоящем потоке не подключён (нужна асинхронная очередь): будет проигнорирован"))
    return issues


def errors(profile: PR.Profile, mode: str = "offline") -> list[Issue]:
    return [i for i in check(profile, mode) if i.level == "error"]


# ------------------------------------------------------------------------------------------ архитектура
def _spec(model_id: str) -> dict:
    from . import models as M

    reg = M.load_registry()
    s = reg.get(model_id)
    if s is None:
        return dict(id=model_id, registered=False)
    lock = M._read_lock().get(model_id, {})
    return dict(id=model_id, registered=True, kind=s.kind, file=s.file, hf_repo=s.hf_repo, rtmlib=s.rtmlib, license=s.license, trust=s.trust, sha256=lock.get("sha256"),
                present=bool((s.local_path and s.local_path.exists()) or s.rtmlib))


# оценка потребности в памяти по компонентам (ГБ): порядок величин из docs/ANALYSIS.md и docs/MVP.md, НЕ измерение на конкретной машине
RAM_VRAM_GB = dict(pose_n=0.5, pose_s=0.8, pose_m=1.5, pose_x=3.0, rtmpose_s=0.3, rtmpose_m=0.6, detector=0.35, clip_b32=0.5, convnext_b=1.2, vlm_2b_q4=2.4, vlm_4b_q4=3.8, vlm_8b_q4=6.0)


def estimate_resources(arch: dict) -> dict:
    """Грубая оценка памяти (ГБ) по стадиям: сколько нужно ОЗУ/VRAM «в пике», если всё на одном устройстве. Для выбора железа и предупреждений, не гарантия."""
    st = {s["id"]: s for s in arch["stages"]}
    pose_id = st["pose_track"]["weights"]["id"]
    size = re.search(r"(\d+)?([nsmx])-pose", pose_id)
    pose = RAM_VRAM_GB.get(f"pose_{size.group(2)}", 0.8) if size else 0.8
    pose *= 1.0 + max(st["pose_track"]["imgsz"] - 960, 0) / 1920          # вход крупнее — больше активаций
    parts = {"pose": pose}
    r = st["pose_track"]["refine"]
    if r.get("enabled") and str(r.get("method", "")).startswith("rtmpose"):
        parts["refine"] = RAM_VRAM_GB["rtmpose_m" if str(r["method"]).endswith("m") else "rtmpose_s"]
    if "evidence" in st:
        parts["detectors"] = RAM_VRAM_GB["detector"] * len(st["evidence"]["detectors"])
    if "photo" in st:
        parts["photo(CLIP)"] = RAM_VRAM_GB["clip_b32"]
    if "vlm" in st:
        m = st["vlm"]["model"].lower()
        parts["vlm"] = RAM_VRAM_GB["vlm_8b_q4" if "8b" in m or "9b" in m else "vlm_4b_q4" if "4b" in m else "vlm_2b_q4"]
    total = sum(parts.values())
    return dict(parts={k: round(v, 2) for k, v in parts.items()}, peak_gb=round(total, 1), sequential_gb=round(max(parts.values()), 1) if parts else 0,
                note="оценка порядка величин; этапы идут последовательно, поэтому пик ближе к «sequential», если модели выгружаются, и к «peak», если держатся вместе")


def detect_architecture(profile: PR.Profile) -> dict:
    """Описание пайплайна по стадиям — выводится из профиля и пакетов автоматически (ничего не заполняется вручную)."""
    cfg, o = profile.cfg(), profile.opts()
    c = cfg
    stages: list[dict] = [dict(id="pose_track", backend=c["pose"]["backend"], weights=_spec(c["pose"]["weights"]), runtime=c["pose"]["runtime"], device=c["pose"]["device"],
                               imgsz=c["pose"]["imgsz"], refine=dict(enabled=bool(c["pose"]["refine"]["enabled"]), method=c["pose"]["refine"].get("method")),
                               tracker=c["tracking"]["tracker"], process_fps=c["video"]["process_fps"], min_person_height_px=c["video"]["min_person_height_px"]),
                          dict(id="cycle_fsm", th_in=c["cycles"]["th_in"], th_out=c["cycles"]["th_out"], hold_sec=[c["cycles"]["hold_min_sec"], c["cycles"]["hold_max_sec"]])]
    cl: dict[str, Any] = dict(id="cycle_classifier", bundle=o["cycle_bundle"], heuristic=not o["cycle_bundle"])
    if o["cycle_bundle"] and (repo_path(o["cycle_bundle"]) / "manifest.json").exists():
        cl["summary"] = describe_bundle(o["cycle_bundle"])
    stages.append(cl)
    if o["objects"]:
        stages.append(dict(id="evidence", detectors=[_spec(d) for d in o["objects"]], imgsz=c["evidence"]["imgsz"], conf=c["evidence"]["conf"]))
    if o["photo_bundle"]:
        stages.append(dict(id="photo", bundle=o["photo_bundle"], backend=o["backend"]))
    if o["vlm_model"] and o["vlm_mode"] != "off":
        stages.append(dict(id="vlm", model=o["vlm_model"], mode=o["vlm_mode"], grey=o["grey"], prompt_file=c["vlm"]["prompt_file"], backend=c["vlm"]["backend"]))
    if o["cycle_bundle_full"]:
        stages.append(dict(id="cascade", bundle=o["cycle_bundle_full"]))
    from .fusion import FusionSpec

    fz = FusionSpec.from_dict(o["fusion"])
    if fz.active:
        stages.append(dict(id="fusion", **fz.to_dict()))
    stages.append(dict(id="events", cycle_th=c["events"]["cycle_th"], merge_gap_sec=c["events"]["merge_gap_sec"], window_sec=c["events"]["window_sec"], window_mode=c["events"]["window_mode"],
                       confidence_threshold=c["events"]["confidence_threshold"]))
    ids = [s["id"] for s in stages]
    mode = "classifier" + ("+fusion" if "fusion" in ids else "") + ("+cascade" if "cascade" in ids else "") + ("+evidence" if "evidence" in ids else "") + ("+photo" if "photo" in ids else "") \
        + ("+vlm" if "vlm" in ids else "")
    arch = dict(schema=SCHEMA, task="smoking events per person track", mode=mode, stages=stages)
    arch["resources_estimate"] = estimate_resources(arch)
    arch["streaming"] = dict(offline=not errors(profile, "offline"), replay=not errors(profile, "replay"), live=not errors(profile, "live"))
    return arch


# ------------------------------------------------------------------------------------------ упаковка
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _clean_name(name: str) -> str:
    n = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("_")
    if not n or n.startswith(".") or n.endswith("."):
        raise ValueError(f"недопустимое имя решателя: {name!r}")
    return n


def _portable_bundle(spec: str | Path) -> str:
    p = repo_path(spec)
    return f"models/{p.parent.name}/{p.name}"


def _normalized(profile: PR.Profile) -> PR.Profile:
    """Копия профиля со ссылками на пакеты в виде `models/<kind>/<имя>`: так они находятся после установки на любой машине."""
    opts = dict(profile.options)
    for k in ("cycle_bundle", "cycle_bundle_full", "photo_bundle"):
        if opts.get(k):
            opts[k] = _portable_bundle(opts[k])
    return PR.Profile(profile.name, profile.description, profile.base, dict(profile.config), opts)


def _bundle_files(spec: str) -> list[tuple[Path, str]]:
    root = repo_path(spec)
    kind = root.parent.name
    if kind not in ("cycle", "photo"):
        raise ValueError(f"пакет {spec}: ожидается models/cycle/<имя> или models/photo/<имя>")
    out = []
    for f in sorted(root.rglob("*")):
        if f.is_file():
            if f.suffix.lower() not in BUNDLE_EXT:
                raise ValueError(f"{f}: формат {f.suffix} не допускается в переносимом пакете")
            out.append((f, f"bundles/{kind}/{root.name}/{f.relative_to(root).as_posix()}"))
    if not out:
        raise FileNotFoundError(f"пакет не найден: {spec}")
    return out


def pack(profile: PR.Profile, out: str | Path | None = None, *, name: str | None = None, with_weights: bool = False, with_dataset: bool = False, run_id: str | None = None,
         exp_root: Path | None = None, labels_dir: Path | None = None) -> dict:
    """Собирает `*.sdsolver.zip`. Возвращает описание (solver.json + путь и размер). Архив пишется во временный файл и переименовывается — недособранного не остаётся."""
    name = _clean_name(name or profile.name)
    arch = detect_architecture(profile)
    errs = [i.text for i in errors(profile)]
    if errs:
        raise ValueError("профиль не собирает рабочий пайплайн:\n  - " + "\n  - ".join(errs))
    o = profile.opts()
    files: list[tuple[Path, str]] = []
    for spec in (o["cycle_bundle"], o["cycle_bundle_full"], o["photo_bundle"]):
        if spec:
            files += _bundle_files(spec)
    notes: list[str] = []
    weights = []
    comps = [s["weights"] for s in arch["stages"] if s["id"] == "pose_track"] + [d for s in arch["stages"] if s["id"] == "evidence" for d in s["detectors"]]
    for comp in comps:
        row = dict(id=comp["id"], file=comp.get("file"), sha256=comp.get("sha256"), license=comp.get("license"), trust=comp.get("trust"), hf_repo=comp.get("hf_repo"),
                   registered=comp.get("registered"), included=False)
        if with_weights and comp.get("file"):
            wp = MODELS / comp["file"]
            if wp.exists() and wp.suffix.lower() in WEIGHT_EXT:
                row.update(included=True, sha256=_sha(wp), size_mb=round(wp.stat().st_size / 1e6, 1), archive_path=f"weights/{Path(comp['file']).name}")
                files.append((wp, row["archive_path"]))
            else:
                notes.append(f"веса {comp['id']} не найдены на диске ({wp}) — в архив не вошли")
        weights.append(row)
    if with_weights is False and any(w["file"] for w in weights):
        notes.append("веса не включены (--with-weights): на новой машине их скачивает `sd models get <id>` либо они лежат в registry")
    if o["vlm_model"] and o["vlm_mode"] != "off":
        pf = repo_path(profile.cfg()["vlm"]["prompt_file"])
        if pf.exists():
            files.append((pf, f"prompts/{pf.name}"))
    dataset = {}
    if with_dataset:
        ld = labels_dir or LABELS
        for fn in ("events_gt.csv", "cycle_labels.csv"):
            if (ld / fn).exists():
                files.append((ld / fn, f"dataset/{fn}"))
                dataset[fn] = True
        if (ld / "events_gt.csv").exists():
            import pandas as pd

            from . import gt as GT
            from .paths import from_portable

            gtd = GT.load(ld / "events_gt.csv")
            tmp_idx = Path(tempfile.mkdtemp()) / "clips_index.csv"
            rows = []
            for cid in sorted(gtd.clip_id.unique()):
                rows.append(dict(clip_id=cid, gt_rows=int((gtd.clip_id == cid).sum())))
            pd.DataFrame(rows).to_csv(tmp_idx, index=False)
            files.append((tmp_idx, "dataset/clips_index.csv"))
            dataset["clips"] = len(rows)
            dataset["gt_fingerprint"] = GT.fingerprint(gtd)
    evaluation = None
    if run_id:
        from . import experiments as XP

        d = (exp_root or XP.EXP_DIR) / run_id
        if (d / "report.json").exists():
            files.append((d / "report.json", "evaluation/report.json"))
            rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
            evaluation = dict(run_id=run_id, mode=rep.get("mode"), metrics={k: rep["metrics"].get(k) for k in ("f1", "precision", "recall", "tp", "fp", "fn", "fp_per_hour", "threshold", "clips")},
                              ci=rep.get("ci"), budget=rep.get("budget"), gt_fingerprint=meta.get("gt_fingerprint"), role=meta.get("role"), created=meta.get("created"))
        else:
            notes.append(f"эксперимент {run_id} не найден: оценка в архив не вошла")
    norm = _normalized(profile)
    for s in arch["stages"]:                       # в описании архитектуры тоже переносимые ссылки
        if s["id"] in ("cycle_classifier", "photo", "cascade") and s.get("bundle"):
            s["bundle"] = _portable_bundle(s["bundle"])
        if s.get("summary"):
            s["summary"]["path"] = _portable_bundle(s["summary"]["path"])
    prof_yaml = yaml.safe_dump(norm.to_dict(), allow_unicode=True, sort_keys=False)
    base_overlay = None
    if profile.base and repo_path(profile.base).exists():
        base_overlay = yaml.safe_load(repo_path(profile.base).read_text(encoding="utf-8"))
    from . import experiments as XP

    meta = dict(schema=SCHEMA, id=name, kind="sdsolver", created=time.strftime("%Y-%m-%dT%H:%M:%S"), git=XP.git_state(), fingerprint=profile.fingerprint(), description=profile.description,
                architecture=arch, profile=norm.to_dict(), base_overlay=base_overlay, weights=weights, dataset=dataset or None, evaluation=evaluation, notes=notes,
                packages=_observed_packages())
    out_p = Path(out) if out else Path(f"{name}{EXT}")
    if not str(out_p).endswith(EXT):
        out_p = out_p.with_name(out_p.name + EXT)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sums = {arc: _sha(src) for src, arc in files}
    sums["profile.yaml"] = hashlib.sha256(prof_yaml.encode("utf-8")).hexdigest()
    meta["files"] = sums
    tmp = out_p.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("solver.json", json.dumps(meta, ensure_ascii=False, indent=1, default=str))
        z.writestr("profile.yaml", prof_yaml)
        for src, arc in files:
            z.write(src, arc)
    tmp.replace(out_p)
    return dict(path=str(out_p), size_mb=round(out_p.stat().st_size / 1e6, 2), files=len(files) + 2, meta=meta)


def _observed_packages() -> dict:
    from .doctor import installed_version

    return {n: installed_version(n) for n in ("numpy", "pandas", "scikit-learn", "lightgbm", "ultralytics", "torch", "openvino", "onnxruntime", "opencv-python", "rtmlib", "streamlit")}


# ------------------------------------------------------------------------------------------ чтение, проверка, установка
def _open(path: str | Path):
    p = Path(path)
    if p.is_dir():
        return None, p
    return zipfile.ZipFile(p), None


def inspect(path: str | Path, verify: bool = True) -> dict:
    """Метаданные решателя + проверка целостности (sha256 каждого файла против solver.json). `ok=False` и список `problems` — архив повреждён или подменён."""
    z, d = _open(path)
    problems: list[str] = []
    try:
        meta = json.loads((z.read("solver.json") if z else (d / "solver.json").read_bytes()).decode("utf-8"))
        if meta.get("schema") != SCHEMA or meta.get("kind") != "sdsolver":
            raise ValueError(f"схема {meta.get('schema')}/{meta.get('kind')} не поддерживается (ожидается {SCHEMA}/sdsolver)")
        if verify:
            for arc, want in meta.get("files", {}).items():
                try:
                    data = z.read(arc) if z else (d / arc).read_bytes()
                except KeyError:
                    problems.append(f"нет файла {arc}")
                    continue
                if hashlib.sha256(data).hexdigest() != want:
                    problems.append(f"sha256 не совпадает: {arc}")
    finally:
        if z:
            z.close()
    return dict(meta=meta, ok=not problems, problems=problems)


def _safe_member(arc: str) -> None:
    parts = Path(arc).parts
    if arc.startswith(("/", "\\")) or ".." in parts or re.match(r"^[A-Za-z]:", arc):
        raise ValueError(f"недопустимый путь в архиве: {arc}")
    top = parts[0] if parts else ""
    ext = Path(arc).suffix.lower()
    rules = {"bundles": BUNDLE_EXT, "weights": WEIGHT_EXT, "prompts": {".txt", ".md"}, "dataset": {".csv"}, "evaluation": {".json"}}
    if arc in ("solver.json", "profile.yaml"):
        return
    if top not in rules or ext not in rules[top]:
        raise ValueError(f"недопустимый файл в архиве: {arc}")


def install(path: str | Path, *, overwrite: bool = False, models_dir: Path | None = None, profiles_dir: Path | None = None, labels_dir: Path | None = None,
            install_dataset: bool = False, allow_unknown_weights: bool = False) -> dict:
    """Устанавливает решатель: пакеты → `models/<kind>/<имя>`, веса → `models/`, профиль → `configs/experiments/<имя>.yaml`, решатель → `models/solvers/<имя>/`.
    Двухфазно: сначала всё проверяется и распаковывается во временный каталог рядом, затем переносится. Возвращает {name, profile, installed, skipped, issues}."""
    from .safety import scan_torch_archive

    md = models_dir or MODELS
    info = inspect(path)
    if not info["ok"]:
        raise ValueError("архив не прошёл проверку целостности: " + "; ".join(info["problems"]))
    meta = info["meta"]
    name = _clean_name(meta["id"])
    for arc in meta.get("files", {}):
        _safe_member(arc)
        if Path(arc).parts[0] == "bundles" and (len(Path(arc).parts) < 4 or Path(arc).parts[1] not in ("cycle", "photo")):
            raise ValueError(f"недопустимое расположение пакета в архиве: {arc} (ожидается bundles/<cycle|photo>/<имя>/<файл>)")
    for w in meta.get("weights", []):
        if w.get("included"):
            _safe_member(str(w.get("archive_path", "")))
            if w["archive_path"] not in meta["files"]:
                raise ValueError(f"вес {w['archive_path']} не описан в files (нет sha256)")
            if w.get("file"):
                _safe_member("weights/" + str(w["file"]).replace("\\", "/"))
    opts = (meta.get("profile") or {}).get("options") or {}
    if opts.get("allow_heuristic") or not opts.get("cycle_bundle"):
        raise ValueError("в решателе не выбран классификатор цикла (или разрешена эвристика): такой решатель не устанавливается")
    if (md / "solvers" / name / "solver.json").exists() and not overwrite:
        raise FileExistsError(f"решатель {name} уже установлен; overwrite=True заменит")
    z, d = _open(path)
    md.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".solver_", dir=str(md)))      # рядом с целью: перенос внутри одного тома
    try:
        stage = tmp / "x"
        for arc in meta["files"]:
            dst = stage / arc
            dst.parent.mkdir(parents=True, exist_ok=True)
            if z:
                size = z.getinfo(arc).file_size
                if size > MAX_FILE_MB * 1e6:
                    raise ValueError(f"{arc}: файл больше {MAX_FILE_MB} МБ")
                with z.open(arc) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
            else:
                shutil.copyfile(d / arc, dst)
        for arc in meta["files"]:
            if Path(arc).suffix.lower() in PICKLE_EXT:
                rep = scan_torch_archive(stage / arc)
                if rep.verdict in ("danger", "error") or (rep.verdict == "unknown" and not allow_unknown_weights):
                    raise RuntimeError(f"веса {arc} не прошли проверку безопасности: {rep.summary()}")
        targets: list[tuple[Path, Path]] = []
        for arc in meta["files"]:
            parts = Path(arc).parts
            if parts[0] == "bundles":
                targets.append((stage / "bundles" / parts[1] / parts[2], md / parts[1] / parts[2]))
        targets = list({(s, t) for s, t in targets})
        for _, t in targets:
            if t.exists() and not overwrite:
                raise FileExistsError(f"{t} уже существует (overwrite=False)")
        installed: list[str] = []
        for s, t in targets:
            if t.exists():
                shutil.rmtree(t)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(s, t)
            installed.append(f"{t.parent.name}/{t.name}")
        skipped: list[str] = []
        for w in meta.get("weights", []):
            if w.get("included"):
                src = stage / w["archive_path"]
                if _sha(src) != w["sha256"]:
                    raise ValueError(f"вес {w['archive_path']}: sha256 содержимого не совпадает с метаданными")
                dst = md / (w["file"] or Path(w["archive_path"]).name)
                if dst.exists() and not overwrite and _sha(dst) != w["sha256"]:
                    skipped.append(f"{dst.name}: уже есть другой файл (sha256 отличается), не перезаписан")
                    continue
                if not dst.exists() or overwrite:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    installed.append(f"weights/{dst.name}")
        sd = md / "solvers" / name
        if sd.exists() and not overwrite:
            raise FileExistsError(f"решатель {name} уже установлен ({sd}); overwrite=True заменит")
        if sd.exists():
            shutil.rmtree(sd)
        sd.mkdir(parents=True)
        (sd / "solver.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        for sub in ("evaluation", "dataset", "prompts"):
            if (stage / sub).exists():
                shutil.copytree(stage / sub, sd / sub)
        prof = PR.Profile(name=name, description=meta.get("description", ""), base=None, config=_flat_config(meta), options=dict(meta["profile"].get("options") or {}))
        prof.cfg()
        prof.opts()
        pf = PR.save(prof, profiles_dir)
        if install_dataset and (stage / "dataset").exists():
            ld = labels_dir or LABELS
            ld.mkdir(parents=True, exist_ok=True)
            for f in (stage / "dataset").glob("*.csv"):
                if f.name != "clips_index.csv" and not (ld / f.name).exists():
                    shutil.copyfile(f, ld / f.name)
                    installed.append(f"labels/{f.name}")
    finally:
        if z:
            z.close()
        shutil.rmtree(tmp, ignore_errors=True)
    issues = [i.to_dict() for i in check(prof)]
    return dict(name=name, profile=str(pf), installed=installed, skipped=skipped, issues=issues, architecture=meta["architecture"]["mode"])


def _flat_config(meta: dict) -> dict:
    """Конфиг профиля с учётом накладки `base`: файл накладки на новой машине может отсутствовать, поэтому её ключи переносятся в точечные переопределения."""
    cfg = dict(meta["profile"].get("config") or {})
    ov = meta.get("base_overlay")
    if ov:
        def walk(d: dict, pre: str = ""):
            for k, v in d.items():
                if isinstance(v, dict):
                    yield from walk(v, f"{pre}{k}.")
                else:
                    yield f"{pre}{k}", v

        for k, v in walk(ov):
            cfg.setdefault(k, v)
    return cfg


def installed_solvers(models_dir: Path | None = None) -> list[dict]:
    root = (models_dir or MODELS) / "solvers"
    out = []
    for f in sorted(root.glob("*/solver.json")) if root.exists() else []:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
            out.append(dict(name=m["id"], created=m.get("created"), mode=m["architecture"]["mode"], fingerprint=m.get("fingerprint"), f1=((m.get("evaluation") or {}).get("metrics") or {}).get("f1")))
        except Exception:
            continue
    return out


def resolve_profile(spec: str | Path, *, overwrite: bool = False) -> PR.Profile:
    """Что угодно → профиль: имя профиля, путь к YAML, установленный решатель, каталог или архив `*.sdsolver.zip` (устанавливается автоматически, если ещё не установлен)."""
    s = str(spec)
    p = Path(s)
    if s.endswith(EXT) or s.endswith(".zip") or (p.is_dir() and (p / "solver.json").exists()):
        meta = inspect(p, verify=False)["meta"]
        name = _clean_name(meta["id"])
        cur = MODELS / "solvers" / name / "solver.json"
        if not cur.exists():
            install(p)
        elif overwrite:
            install(p, overwrite=True)
        else:
            old = json.loads(cur.read_text(encoding="utf-8"))
            if old.get("fingerprint") != meta.get("fingerprint") or old.get("created") != meta.get("created"):
                raise ValueError(f"решатель {name} уже установлен, но это другая версия архива: `sd solver install {p} --overwrite` заменит, либо переименуйте решатель")
        return PR.load(name)
    if (MODELS / "solvers" / s / "solver.json").exists():
        return PR.load(s)
    return PR.load(s)
