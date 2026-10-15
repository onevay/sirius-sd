"""Справочники и профили для интерфейса разработчика: каталог моделей, реестр весов, профили (чтение, сохранение с проверкой ключей, копия, YAML) и журнал задач.
Без HTTP: маршруты — `routers/dev.py`. Всё опирается на существующие модули проекта."""
from __future__ import annotations

import yaml

from ... import catalog as CAT
from ... import model_tools as MT
from ... import profiles as PR
from ... import solver as SV
from ... import tasks as TK
from ...config import load_config

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
    ch = lambda xs: [dict(id=c.id, label=c.label, ready=c.ready, note=c.note) for c in xs]      # noqa: E731
    return dict(pose=ch(CAT.pose_models(False)), detectors=ch(CAT.detector_models(False)), cycle=ch(CAT.cycle_bundles()), photo=ch(CAT.photo_bundles()), vlm=ch(CAT.vlm_models()),
                runtimes={k: list(v) for k, v in CAT.RUNTIMES.items()}, trackers=list(CAT.TRACKERS), refine=list(CAT.REFINE_METHODS), vlm_modes=list(CAT.VLM_MODES),
                imgsz=[480, 640, 800, 960, 1088, 1280, 1600, 1920])


def profile_get(name: str) -> dict:
    p = PR.load_named(name)
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
    p = PR.load_named(src)
    PR.save(PR.Profile(name=dst, description=p.description, base=p.base, config=dict(p.config), options=dict(p.options)))
    return profile_get(dst)


def profile_export_yaml(name: str) -> str:
    return yaml.safe_dump(PR.load_named(name).to_dict(), allow_unicode=True, sort_keys=False)


def registry() -> dict:
    from ... import models as M

    rows = []
    for s in M.load_registry().values():
        st = M.status(s)
        rows.append(dict(id=s.id, kind=s.kind, trust=s.trust, license=s.license, present=bool(st.get("present")), size_mb=st.get("size_mb"), note=s.note[:100]))
    return dict(models=rows, user=[m["id"] for m in MT.load_user_models()], bundles=MT.list_bundles(), solvers=SV.installed_solvers())


# ------------------------------------------------------------------------------------------ задачи доработки (белый список из tasks.CATALOG)
def tasks_catalog() -> list[dict]:
    return [dict(id=t.id, title=t.title, description=t.description, needs_net=t.needs_net,
                 fields=[dict(key=f.key, label=f.label, kind=f.kind, default=f.default, choices=list(f.choices), help=f.help) for f in t.fields]) for t in TK.CATALOG.values()]


def task_log(tid: str, lines: int = 120) -> dict:
    st = TK.status(tid)
    return dict(state=st["state"], title=st["title"], seconds=round(st["seconds"], 1), log=TK.tail(tid, lines))
