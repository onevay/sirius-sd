"""Каталог ДОСТУПНЫХ моделей по типам — единый источник для выпадающих списков интерфейса и CLI.

Любая модель, попавшая в реестр (`configs/models.yaml`, `models/user_models.yaml`, пакеты в `models/cycle|photo`, теги локального Ollama), появляется в списке сама:
для нового эксперимента не нужно править код интерфейса. Модель считается доступной, если её веса есть на диске (или её скачивает сама библиотека, как rtmlib).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import models as M
from . import model_tools as MT

REFINE_METHODS = ("rtmpose-s", "rtmpose-m", "yolo")
RUNTIMES = {"torch": ["cpu", "cuda:0"], "openvino": ["intel:gpu", "intel:cpu"]}
TRACKERS = ("botsort", "bytetrack")
VLM_MODES = ("off", "grey", "all")


@dataclass(frozen=True)
class Choice:
    id: str
    label: str
    ready: bool = True
    note: str = ""


def _reg_choices(kind: str, only_ready: bool) -> list[Choice]:
    out = []
    for s in M.load_registry().values():
        if s.kind != kind:
            continue
        ready = bool(s.rtmlib) or bool(s.local_path and s.local_path.exists())
        if only_ready and not ready:
            continue
        tag = "" if s.trust == "official" else " · community"
        out.append(Choice(s.id, f"{s.id}{tag}", ready, f"{s.license}; {s.note}"[:120]))
    return out


def pose_models(only_ready: bool = True) -> list[Choice]:
    return _reg_choices("pose", only_ready)


def detector_models(only_ready: bool = True) -> list[Choice]:
    return _reg_choices("detector", only_ready)


def cycle_bundles() -> list[Choice]:
    return [Choice(f"models/cycle/{b['name']}", b["name"], True, b.get("note", "")) for b in MT.list_bundles() if b["kind"] == "cycle"]


def photo_bundles() -> list[Choice]:
    return [Choice(f"models/photo/{b['name']}", b["name"], True, b.get("note", "")) for b in MT.list_bundles() if b["kind"] == "photo"]


def vlm_models() -> list[Choice]:
    """Теги, доступные в локальном Ollama (пусто, если сервер не запущен)."""
    try:
        from . import ollama_ctl as O

        if not O.is_up():
            return []
        return [Choice(m["name"], m["name"]) for m in O.status().get("models", [])]
    except Exception:
        return []


def summary() -> dict[str, int]:
    return dict(pose=len(pose_models()), detector=len(detector_models()), cycle=len(cycle_bundles()), photo=len(photo_bundles()), vlm=len(vlm_models()))
