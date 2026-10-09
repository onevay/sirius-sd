"""Эксперименты «что если»: варианты профиля (одна строка таблицы = одно изменение) и парное сравнение сохранённых прогонов.

    base = Profile('base')
    vs = variants_from_rows(base, [dict(name='imgsz640', param='pose.imgsz', value='640'),
                                   dict(name='без_уточнения', param='pose.refine.enabled', value='false'),
                                   dict(name='vlm', param='vlm_model', value='qwen3.5:2b-q4_K_M'), dict(name='vlm', param='vlm_mode', value='grey')])   # одинаковое имя = одно изменение из нескольких параметров
    outs = run_variants(dirs, vs, ...)           # те же папки, кэш позы общий — пересчитывается только изменившееся

Параметр: точечный ключ конфига (`events.cycle_th`, `pose.weights`, …) или ключ опций профиля (`cycle_bundle`, `photo_bundle`, `objects`, `vlm_model`, `vlm_mode`, `grey`, `backend`).
Значение разбирается как YAML (`0.6`, `true`, `[a, b]`, `models/cycle/x`). Неверный ключ — `KeyError` до запуска.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import yaml

from . import evaluation as EV
from . import experiments as XP
from . import gt as GT
from .profiles import OPTION_KEYS, Profile


def variants_from_rows(base: Profile, rows: Sequence[dict], include_base: bool = True) -> list[Profile]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        name, param = str(r.get("name") or "").strip(), str(r.get("param") or "").strip()
        if not name or not param:
            continue
        groups.setdefault(name, []).append(r)
    out = [Profile(base.name, base.description, base.base, dict(base.config), dict(base.options))] if include_base else []
    for name, rs in groups.items():
        cfg, opt = dict(base.config), dict(base.options)
        for r in rs:
            key = str(r["param"]).strip()
            val = yaml.safe_load(str(r.get("value", "")))
            (opt if key in OPTION_KEYS else cfg)[key] = val
        p = Profile(f"{base.name}+{name}", base.description, base.base, cfg, opt)
        p.cfg()
        p.opts()           # неверный ключ — ошибка сейчас, а не посреди долгого прогона
        out.append(p)
    return out


def run_variants(dirs: Sequence, variants: Sequence[Profile], *, progress: Callable | None = None, **kw) -> list:
    """`runner.evaluate_dirs` для каждого варианта на одних и тех же папках; возвращает список `Outcome` (журнал пишется как обычно)."""
    from . import runner as RN

    return [RN.evaluate_dirs(dirs, p, progress=progress, **kw) for p in variants]


def compare_saved(rid_a: str, rid_b: str, n: int = 500, root=None) -> dict:
    """Парное сравнение двух сохранённых прогонов на общих клипах (пороги берутся каждый свой). Прогоны должны быть в режиме `events`."""
    ra, ma, ea, ga = XP.load_run(rid_a, root)
    rb, mb, eb, gb = XP.load_run(rid_b, root)
    if ra.mode != "events" or rb.mode != "events":
        raise ValueError("парное сравнение — только для оценки по эталону событий")
    da = {c["clip_id"]: c["duration"] for c in ma["clips"] if c["duration"] > 0}
    db = {c["clip_id"]: c["duration"] for c in mb["clips"] if c["duration"] > 0}
    common = {c: da[c] for c in da if c in db}
    gt = gb if len(gb) else ga
    gts = [g for g in GT.to_eval_gts(gt, set(common))]
    pa = [p for c in common for p in EV.to_preds(ea[ea.clip_id == c], c)] if len(ea) else []
    pb = [p for c in common for p in EV.to_preds(eb[eb.clip_id == c], c)] if len(eb) else []
    sa = EV.Settings(threshold=ra.settings["threshold"], target_f1=ra.settings.get("target_f1", 0.8))
    sb = replace(sa, threshold=rb.settings["threshold"])
    r = EV.paired_bootstrap(pa, pb, gts, common, sa, sb, n=n)
    r["same_gt"] = ma.get("gt_fingerprint") == mb.get("gt_fingerprint")
    return r
