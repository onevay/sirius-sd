"""Профили моделей, реестр, перенос (решатель, конфигурации, пакеты) и задачи доработки — логика веб-приложения без HTTP. Всё опирается на существующие модули проекта."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import yaml

from .. import catalog as CAT
from .. import model_tools as MT
from .. import profiles as PR
from .. import runner as RN
from .. import solver as SV
from .. import tasks as TK
from ..config import load_config
from ..paths import OUTPUTS

EXPORTS = OUTPUTS / "web" / "exports"
MAIN_KEYS = ("pose.weights", "pose.runtime", "pose.device", "pose.imgsz", "pose.refine.enabled", "pose.refine.method", "tracking.tracker", "video.process_fps",
             "cycles.th_in", "cycles.th_out", "events.cycle_th", "events.confidence_threshold")


def _flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        kk = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, kk + "."))
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[kk] = v
    return out


def catalog() -> dict:
    ch = lambda xs: [dict(id=c.id, label=c.label, ready=c.ready, note=c.note) for c in xs]
    return dict(pose=ch(CAT.pose_models(False)), detectors=ch(CAT.detector_models(False)), cycle=ch(CAT.cycle_bundles()), photo=ch(CAT.photo_bundles()), vlm=ch(CAT.vlm_models()),
                runtimes={k: list(v) for k, v in CAT.RUNTIMES.items()}, trackers=list(CAT.TRACKERS), refine=list(CAT.REFINE_METHODS), vlm_modes=list(CAT.VLM_MODES),
                imgsz=[480, 640, 800, 960, 1088, 1280, 1600, 1920])


def profile_get(name: str) -> dict:
    p = PR.load(name)
    eff = _flat(p.cfg().to_dict())
    default = _flat(load_config(None).to_dict())
    iss = [dict(level=i.level, code=i.code, text=i.text) for i in SV.check(p, "replay", devices=True)]
    return dict(name=p.name, description=p.description, base=p.base, config=dict(p.config), options={**PR.DEFAULT_OPTIONS, **p.options}, effective=eff, default=default,
                main={k: eff.get(k) for k in MAIN_KEYS}, describe=p.describe(), issues=iss, fingerprint=p.fingerprint())


def profile_save(name: str, body: dict) -> dict:
    """Сохраняет профиль: проверка ключей конфигурации и options до записи (опечатка = ошибка). Пустые options убираются."""
    cfgp = {str(k): v for k, v in (body.get("config") or {}).items()}
    opts = {k: v for k, v in (body.get("options") or {}).items() if PR.DEFAULT_OPTIONS.get(k) != v or k in ("cycle_bundle",)}
    p = PR.Profile(name=name, description=str(body.get("description", "")), base=body.get("base") or None, config=cfgp, options=opts)
    PR.save(p)
    return profile_get(p.name)


def profile_clone(src: str, dst: str) -> dict:
    p = PR.load(src)
    PR.save(PR.Profile(name=dst, description=p.description, base=p.base, config=dict(p.config), options=dict(p.options)))
    return profile_get(dst)


def profile_export_yaml(name: str) -> str:
    return yaml.safe_dump(PR.load(name).to_dict(), allow_unicode=True, sort_keys=False)


def profile_import_yaml(data: bytes, name: str | None = None) -> dict:
    raw = yaml.safe_load(data.decode("utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("ожидается YAML-словарь профиля")
    nm = name or raw.get("name") or "imported"
    return profile_save(nm, dict(description=raw.get("description", ""), base=raw.get("base"), config=raw.get("config") or {}, options=raw.get("options") or {}))


def registry() -> dict:
    from .. import models as M

    rows = []
    for s in M.load_registry().values():
        st = M.status(s)
        rows.append(dict(id=s.id, kind=s.kind, trust=s.trust, license=s.license, present=bool(st.get("present")), size_mb=st.get("size_mb"), note=s.note[:100]))
    return dict(models=rows, user=[m["id"] for m in MT.load_user_models()], bundles=MT.list_bundles(), solvers=SV.installed_solvers())


def model_add(mid: str, kind: str, lic: str, filename: str | None, data_path: Path | None, hf_repo: str | None = None, allow_unknown: bool = False) -> dict:
    if data_path is not None:
        r = MT.register_user_model(dict(id=mid, kind=kind, license=lic), copy_from=data_path, allow_unknown=allow_unknown)
        return dict(file=r["spec"]["file"], scan=r["scan"])
    if hf_repo:
        MT.register_user_model(dict(id=mid, kind=kind, license=lic, hf_repo=hf_repo, hf_patterns=["*.json", "model.safetensors"]))
        return dict(file=None, scan=None, note="запись добавлена; скачайте веса задачей «Скачать веса моделей»")
    raise ValueError("нужен файл весов или репозиторий Hugging Face")


def model_remove(mid: str) -> bool:
    return MT.remove_user_model(mid)


def solver_pack(profile: str, name: str | None, with_weights: bool, with_dataset: bool, classifier: str | None = None, run_id: str | None = None) -> dict:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    p = RN.with_classifier(PR.load(profile), classifier or None)
    nm = (name or profile).strip()
    out = SV.pack(p, EXPORTS / nm, name=nm, with_weights=with_weights, with_dataset=with_dataset, run_id=run_id or None)
    f = Path(out["path"])
    return dict(file=f.name, size_mb=out["size_mb"], meta=SV.inspect(f)["meta"])


def solver_inspect(path: Path) -> dict:
    r = SV.inspect(path)
    a = r["meta"]["architecture"]
    return dict(ok=r["ok"], problems=r["problems"], id=r["meta"]["id"], created=r["meta"]["created"], fingerprint=r["meta"].get("fingerprint"), mode=a["mode"], stages=[s["id"] for s in a["stages"]],
                streaming=a["streaming"], resources=a["resources_estimate"], evaluation=(r["meta"].get("evaluation") or {}).get("metrics"), notes=r["meta"].get("notes", []))


def solver_install(path: Path, overwrite: bool) -> dict:
    return SV.install(path, overwrite=overwrite)


def bundles_export(items: list[str]) -> dict:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    out = EXPORTS / f"models_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    man = MT.export_bundles([tuple(i.split("/", 1)) for i in items], out)
    return dict(file=out.name, files=len(man))


def bundles_import(path: Path, overwrite: bool) -> list[str]:
    return MT.import_bundles(path, overwrite=overwrite)


def export_file(name: str) -> Path:
    f = (EXPORTS / Path(name).name).resolve()
    f.relative_to(EXPORTS.resolve())
    if not f.is_file():
        raise FileNotFoundError(name)
    return f


# ------------------------------------------------------------------------------------------ задачи доработки (белый список из tasks.CATALOG)
def tasks_catalog() -> list[dict]:
    return [dict(id=t.id, title=t.title, description=t.description, needs_net=t.needs_net,
                 fields=[dict(key=f.key, label=f.label, kind=f.kind, default=f.default, choices=list(f.choices), help=f.help) for f in t.fields]) for t in TK.CATALOG.values()]


def task_start(task_id: str, values: dict) -> dict:
    return dict(id=TK.start(task_id, values or {}))


def tasks_list() -> list[dict]:
    return TK.list_tasks()


def task_log(tid: str, lines: int = 120) -> dict:
    st = TK.status(tid)
    return dict(state=st["state"], title=st["title"], seconds=round(st["seconds"], 1), log=TK.tail(tid, lines))


def task_stop(tid: str) -> bool:
    return TK.stop(tid)


def tmp_save(stream, length: int, suffix: str, limit_gb: float = 8.0) -> Path:
    """Тело запроса → временный файл (потоком). Каталог удаляется вызывающим через shutil.rmtree(path.parent)."""
    if length <= 0 or length > limit_gb * 1e9:
        raise ValueError("пустой файл или больше допустимого размера")
    d = Path(tempfile.mkdtemp(prefix="sdweb_"))
    f = d / ("upload" + suffix)
    left = length
    with open(f, "wb") as out:
        while left > 0:
            chunk = stream.read(min(1 << 20, left))
            if not chunk:
                raise ValueError("загрузка оборвалась")
            out.write(chunk)
            left -= len(chunk)
    return f
