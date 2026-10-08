"""Позднее слияние: дополнительные сигналы (VLM, предмет, фото-модель) подмешиваются к оценке КЛАССИФИКАТОРА ЦИКЛА, а не к его входу.

Классификатор цикла — основа решения и всегда обязателен. Его вектор признаков (список `features` в manifest пакета) не меняется ни при каких настройках слияния: добавить сигнал
значит добавить поправку к логиту его оценки, а не новую колонку в обученную модель (которую пришлось бы переобучать и заново проверять):

    logit(итог) = logit(оценка классификатора) + Σ вес_k · (logit(p_k) − logit(опорное_k))     по сигналам k, которые есть у цикла

* сигнала нет (NaN: VLM вызывается только для «серой зоны», предмет не считался) — поправка 0, оценка цикла не меняется;
* веса по умолчанию 0 — слияние выключено, итог равен оценке классификатора побитово;
* опорное значение — «нейтральная» точка сигнала, где поправка нулевая: VLM 0.5 (порог `vlm.threshold`), фото-модель 0.5, предмет 0.15 (порог `evidence.conf`);
* поправка ограничена (`clip_logit`), поэтому один шумный сигнал не может «перевесить» классификатор.

Веса подбирают на валидации (страница «Эксперименты» → «Варианты профиля», ключ `fusion`), а не вручную «на глаз»: слияние — гиперпараметр, и честно оценивается тем же `sd eval`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SIGNALS = {
    "vlm": dict(column="vlm_yesno", ref=0.5),
    "photo": dict(column="photo_p_mean", ref=0.5),
    "object": dict(column="obj_any_max_conf", ref=0.15),
}
EPS = 0.02


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class FusionSpec:
    weights: dict[str, float] = field(default_factory=dict)      # {'vlm': 0.8, 'photo': 0.0, 'object': 0.5}
    clip_logit: float = 2.0                                       # максимум |поправки| в логитах
    refs: dict[str, float] = field(default_factory=dict)         # переопределение опорных значений

    @classmethod
    def from_dict(cls, d: dict | None) -> "FusionSpec":
        d = dict(d or {})
        unknown = set(d) - {"weights", "clip_logit", "refs"}
        if unknown:
            raise KeyError(f"fusion: неизвестные ключи {sorted(unknown)} (допустимо weights, clip_logit, refs)")
        w = {str(k): float(v) for k, v in (d.get("weights") or {}).items()}
        bad = set(w) - set(SIGNALS)
        if bad:
            raise KeyError(f"fusion.weights: неизвестные сигналы {sorted(bad)} (допустимо {sorted(SIGNALS)})")
        if any(v < 0 or v > 5 for v in w.values()):
            raise ValueError("fusion.weights: вес сигнала ∈ [0, 5] (отрицательный вес означал бы «чем увереннее VLM, тем меньше курение»)")
        return cls(w, float(d.get("clip_logit", 2.0)), {str(k): float(v) for k, v in (d.get("refs") or {}).items()})

    @property
    def active(self) -> bool:
        return any(v > 0 for v in self.weights.values())

    def to_dict(self) -> dict:
        return dict(weights=dict(self.weights), clip_logit=self.clip_logit, **({"refs": dict(self.refs)} if self.refs else {}))


def signal_column(tab: pd.DataFrame, name: str) -> pd.Series | None:
    col = SIGNALS[name]["column"]
    if col in tab.columns:
        return tab[col].astype(float)
    if name == "object":      # `obj_any_max_conf` может не быть посчитан: берём максимум по детекторам
        cols = [c for c in tab.columns if c.startswith("obj_") and c.endswith("_max_conf")]
        return tab[cols].astype(float).max(axis=1) if cols else None
    return None


def fuse(base: np.ndarray, tab: pd.DataFrame, spec: FusionSpec | dict | None) -> tuple[np.ndarray, pd.DataFrame]:
    """(итоговая оценка, вклад каждого сигнала в логитах). `base` — оценка классификатора; NaN в ней остаётся NaN."""
    spec = spec if isinstance(spec, FusionSpec) else FusionSpec.from_dict(spec)
    base = np.asarray(base, float)
    contrib = pd.DataFrame(0.0, index=tab.index, columns=[f"fusion_{k}" for k in SIGNALS])
    if not spec.active:
        return base.copy(), contrib
    shift = np.zeros(len(base))
    for name, w in spec.weights.items():
        if w <= 0:
            continue
        s = signal_column(tab, name)
        if s is None:
            continue
        ref = spec.refs.get(name, SIGNALS[name]["ref"])
        c = np.where(np.isfinite(s.to_numpy()), w * (_logit(s.to_numpy()) - _logit(ref)), 0.0)
        c = np.clip(c, -spec.clip_logit, spec.clip_logit)
        contrib[f"fusion_{name}"] = c
        shift += c
    shift = np.clip(shift, -spec.clip_logit, spec.clip_logit)
    out = 1.0 / (1.0 + np.exp(-(_logit(base) + shift)))
    return np.where(np.isfinite(base) & (shift != 0), out, base), contrib       # у циклов без поправки оценка остаётся ровно прежней
