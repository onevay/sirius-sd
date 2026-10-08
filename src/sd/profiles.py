"""Профиль эксперимента: ЧТО именно запускаем — какие модели и параметры. Один файл = один воспроизводимый вариант системы.

    configs/experiments/<имя>.yaml
        name: baseline
        description: поза YOLO26n + классификатор cycle_fast
        base: configs/profiles/fast.yaml          # необязательно: готовая накладка поверх default.yaml
        config:                                   # точечные переопределения default.yaml (точечные ключи; опечатка в ключе = ошибка, а не молчание)
          pose.weights: yolo11m-pose
          events.confidence_threshold: 0.6
        options:                                  # то, что не лежит в конфиге: пакеты и модели для оценки цикла
          cycle_bundle: models/cycle/cycle_fast
          photo_bundle: null
          objects: [smoking_yolo11m_beehzod]
          vlm_model: qwen3.5:2b-q4_K_M
          vlm_mode: grey

Любая зарегистрированная модель (реестр `configs/models.yaml` + свои в `models/user_models.yaml`) выбирается именем, поэтому новые эксперименты —
это новый YAML-файл, а не правка кода. Отпечаток профиля (`fingerprint`) включает sha256 весов и пакетов: кэш прогонов инвалидируется при замене весов.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import Cfg, load_config, stable_hash
from .paths import CONFIGS, ROOT, repo_path

EXPERIMENTS_DIR = CONFIGS / "experiments"
OPTION_KEYS = ("cycle_bundle", "cycle_bundle_full", "photo_bundle", "objects", "vlm_model", "vlm_mode", "grey", "backend", "fusion", "allow_heuristic")
DEFAULT_OPTIONS: dict[str, Any] = dict(cycle_bundle=None, cycle_bundle_full=None, photo_bundle=None, objects=[], vlm_model=None, vlm_mode="off", grey=[0.3, 0.8], backend="auto",
                                       fusion=None, allow_heuristic=False)
_FP_OMIT_IF_DEFAULT = ("fusion", "allow_heuristic")      # новые опции не меняют отпечаток профилей, которые их не используют (кэш прогонов остаётся годным)


@dataclass
class Profile:
    name: str
    description: str = ""
    base: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- сборка
    def opts(self) -> dict[str, Any]:
        unknown = set(self.options) - set(OPTION_KEYS)
        if unknown:
            raise KeyError(f"профиль «{self.name}»: неизвестные options {sorted(unknown)} (допустимо {OPTION_KEYS})")
        return {**DEFAULT_OPTIONS, **self.options}

    def cfg(self) -> Cfg:
        """default.yaml (+ накладка `base`) + точечные переопределения. Неизвестный ключ — `KeyError`."""
        over = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in self.config.items()]
        return load_config(_abs(self.base) if self.base else None, over)

    def options_obj(self, render: bool = False, camera_id: str = "cam_local"):
        """`pipeline.Options` (импорт отложен: pipeline тянет тяжёлые зависимости)."""
        from .pipeline import Options

        o = self.opts()
        return Options(cycle_bundle=o["cycle_bundle"], cycle_bundle_full=o["cycle_bundle_full"], photo_bundle=o["photo_bundle"], objects=tuple(o["objects"] or ()),
                       vlm_model=o["vlm_model"], vlm_mode=o["vlm_mode"] if o["vlm_model"] else "off", grey=tuple(o["grey"]), render=render, camera_id=camera_id, backend=o["backend"],
                       fusion=o["fusion"], allow_heuristic=bool(o["allow_heuristic"]))

    def classifier_problem(self) -> str | None:
        """Текст проблемы, если у профиля нет классификатора цикла (а он обязателен), иначе None. Эвристика по длительности паузы допускается только явным `allow_heuristic: true`."""
        o = self.opts()
        if o["cycle_bundle"] or o["allow_heuristic"]:
            return None
        return (f"профиль «{self.name}»: не выбран классификатор цикла (options.cycle_bundle). Классификатор обязателен: без него оценка цикла — эвристика по длительности паузы, "
                "она не отличает питьё и телефон от курения. Выберите пакет из models/cycle/ (или явно разрешите эвристику для отладки: options.allow_heuristic: true)")

    @property
    def threshold(self) -> float:
        return float(self.cfg()["events"]["confidence_threshold"])

    # ---------------------------------------------------------------- идентичность
    def fingerprint(self) -> str:
        """Отпечаток всего, что влияет на предсказания: конфиг, опции, sha256 весов и пакетов. Порог confidence НЕ входит: он применяется после прогона."""
        cfg = {k: v for k, v in self.cfg().to_dict().items()}
        cfg.get("events", {}).pop("confidence_threshold", None)
        o = self.opts()
        o_fp = {k: v for k, v in o.items() if not (k in _FP_OMIT_IF_DEFAULT and v == DEFAULT_OPTIONS[k])}
        for k in ("cycle_bundle", "cycle_bundle_full", "photo_bundle"):      # путь к пакету зависит от машины; идентичность — имя + sha256 содержимого (в weights)
            if o_fp.get(k):
                o_fp[k] = Path(o_fp[k]).name
        ident = dict(cfg=cfg, options=o_fp, weights=_weights_identity(cfg, o))
        return stable_hash(ident, 10)

    def describe(self) -> dict[str, Any]:
        """Краткая сводка «какие модели выбраны» для таблиц и журнала."""
        c, o = self.cfg(), self.opts()
        return dict(pose=c["pose"]["weights"], refine=(c["pose"]["refine"]["method"] if c["pose"]["refine"]["enabled"] else "—"), runtime=f"{c['pose']['runtime']}/{c['pose']['device']}",
                    imgsz=c["pose"]["imgsz"], tracker=c["tracking"]["tracker"], fps=c["video"]["process_fps"], cycle_model=Path(o["cycle_bundle"]).name if o["cycle_bundle"] else "эвристика",
                    photo=Path(o["photo_bundle"]).name if o["photo_bundle"] else "—", objects=",".join(o["objects"]) or "—",
                    vlm=(f"{o['vlm_model']} ({o['vlm_mode']})" if o["vlm_model"] and o["vlm_mode"] != "off" else "—"), threshold=c["events"]["confidence_threshold"],
                    fusion=(", ".join(f"{k}×{v:g}" for k, v in (o["fusion"] or {}).get("weights", {}).items() if v) or "—"))

    # ---------------------------------------------------------------- файл
    def to_dict(self) -> dict:
        return dict(name=self.name, description=self.description, **({"base": self.base} if self.base else {}), config=dict(self.config), options=dict(self.options))


def _abs(p: str | Path) -> Path:
    return repo_path(p)


def _sha_path(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _weights_identity(cfg: dict, o: dict) -> dict:
    from . import models as M

    lock = M._read_lock()
    ids = [cfg["pose"]["weights"], *(o["objects"] or ())]
    ident = {i: (lock.get(i, {}).get("sha256") or "")[:16] for i in ids}
    for k in ("cycle_bundle", "cycle_bundle_full", "photo_bundle"):
        if o.get(k):
            ident[k] = _sha_path(_abs(o[k]))
    ident["vlm_prompt"] = _sha_path(_abs(cfg["vlm"]["prompt_file"])) if o.get("vlm_model") else None
    return ident


def _check_name(name: str) -> str:
    n = str(name).strip()
    if not n or not all(c.isalnum() or c in "-_." for c in n):
        raise ValueError("имя профиля: буквы, цифры, - _ .")
    return n


def save(p: Profile, directory: Path | None = None) -> Path:
    p.cfg()          # проверка ключей до записи: битый профиль на диск не попадает
    p.opts()
    d = directory or EXPERIMENTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{_check_name(p.name)}.yaml"
    f.write_text(yaml.safe_dump(p.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return f


def load(name_or_path: str | Path, directory: Path | None = None) -> Profile:
    f = Path(name_or_path)
    if not f.suffix:
        f = (directory or EXPERIMENTS_DIR) / f"{name_or_path}.yaml"
    raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    return Profile(name=raw.get("name") or f.stem, description=raw.get("description", ""), base=raw.get("base"), config=raw.get("config") or {}, options=raw.get("options") or {})


def list_profiles(directory: Path | None = None) -> list[str]:
    d = directory or EXPERIMENTS_DIR
    return sorted(f.stem for f in d.glob("*.yaml")) if d.exists() else []


def delete(name: str, directory: Path | None = None) -> bool:
    f = (directory or EXPERIMENTS_DIR) / f"{_check_name(name)}.yaml"
    if f.exists():
        f.unlink()
        return True
    return False


def heuristic_profile() -> Profile:
    """Отладочная база БЕЗ классификатора (оценка цикла — эвристика по длительности паузы). Явная, поэтому не может появиться «случайно»; для оценки качества не годится."""
    return Profile(name="heuristic", description="только для отладки: эвристика по длительности паузы вместо классификатора", options=dict(allow_heuristic=True))


def default_profile() -> Profile:
    """Профиль «как в configs/default.yaml», без классификатора цикла (эвристика) — исходная точка сравнения."""
    return Profile(name="default", description="configs/default.yaml без классификатора цикла (эвристика по длительности паузы)")
