"""Загрузка YAML-конфига с переопределениями из CLI (`-s section.key=value`)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import CONFIGS


class Cfg(dict):
    """dict с доступом через точку: `cfg.cycles.th_in`."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def set_by_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            raise KeyError(f"Нет секции «{k}» в конфиге (путь {dotted})")
        cur = cur[k]
    if keys[-1] not in cur:
        raise KeyError(f"Нет параметра «{dotted}» в конфиге (опечатка?)")
    cur[keys[-1]] = value


def deep_merge(base: dict, over: dict) -> dict:
    """Рекурсивно накладывает `over` на `base` (на месте). Ключ, которого нет в base, — ошибка: так ловятся опечатки в профилях."""
    for k, v in over.items():
        if k not in base:
            raise KeyError(f"Профиль задаёт неизвестный параметр «{k}»")
        if isinstance(v, dict) and isinstance(base[k], dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Cfg:
    """default.yaml + (необязательно) профиль-накладка `path` + переопределения `-s секция.ключ=значение`.

    `path` может быть как полным конфигом, так и маленьким профилем из configs/profiles/ (достаточно указать изменяемые ключи).
    """
    default = CONFIGS / "default.yaml"
    raw = copy.deepcopy(yaml.safe_load(default.read_text(encoding="utf-8")))
    if path and Path(path).resolve() != default.resolve():
        deep_merge(raw, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"Ожидается section.key=value, получено: {item}")
        key, val = item.split("=", 1)
        set_by_path(raw, key.strip(), yaml.safe_load(val))
    return _wrap(raw)


def stable_hash(obj: Any, n: int = 8) -> str:
    """Короткий стабильный хэш любого JSON-совместимого объекта — для ключей кэша артефактов."""
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def section_hash(cfg: dict, *keys: str, n: int = 8) -> str:
    """Хэш выбранных верхнеуровневых секций конфига."""
    return stable_hash({k: cfg[k] for k in keys}, n)
