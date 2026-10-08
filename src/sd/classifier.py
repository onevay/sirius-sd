"""Управление классификатором цикла: спецификация (признаки, члены ансамбля, их гиперпараметры, схема кросс-валидации), диагностика переобучения и обучение пакета.

Классификатор оценивает цикл «рука ко рту» (затяжка или нет). Обучающих циклов всего сотни, поэтому главный риск — переобучение на текущем датасете. Диагностика даёт четыре независимых сигнала:
  разрыв train − oof      — качество на обучающих данных против качества на видео, которых модель не видела (большой разрыв = запоминание);
  контроль перестановкой  — AUC при перемешанных метках: должен быть ≈ 0.5; если выше, схема кросс-валидации «подсматривает» (утечка);
  кривая обучения         — oof-AUC при 40–100 % обучающих видео: растёт к концу — данных не хватает, плоская — модель упёрлась в признаки;
  важность признаков      — падение oof-AUC при перемешивании признака среди тестовых циклов; отрицательная/нулевая — признак бесполезен или вреден.
Всё считается по таблице циклов с метками жестов (`oof_eval.attach_labels` + `enrich.attach`), без обращения к скрытому набору.
Спецификации хранятся в `configs/classifiers/<имя>.yaml` и воспроизводимы: имя набора или явный список признаков, члены, параметры, CV.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from . import bundle as B
from . import cycle_models as CM
from .feature_auc import _auc
from .paths import ROOT

DIR = ROOT / "configs" / "classifiers"
KINDS = {"logreg": "логистическая регрессия (C — обратная сила регуляризации)", "lgbm": "LightGBM (деревья малой глубины)", "mlp": "MLP (один скрытый слой)"}
PARAMS = {   # допустимые гиперпараметры и значения по умолчанию (для интерфейса; неизвестные ключи fit_member молча передаёт LightGBM — проверяем здесь)
    "logreg": {"C": 0.3},
    "lgbm": {"num_leaves": 4, "max_depth": 2, "n_estimators": 150, "learning_rate": 0.05, "min_child_samples": 8, "lambda_l2": 5.0, "feature_fraction": 0.7},
    "mlp": {"hidden": 8, "alpha": 1.0},
}
DEFAULT_CV = dict(n_splits=5, repeats=3, by_scene=False, seed=0)


def default_spec(set_name: str = "obj_hold_zsd", kinds: Sequence[str] = ("lr", "gb", "nn"), tab: pd.DataFrame | None = None) -> dict:
    """Стартовая спецификация из набора признаков (как у `sd train-bundle --set`); если дана таблица — только доступные в ней колонки."""
    feats = CM.available(list(CM.SETS[set_name]), tab) if tab is not None else list(dict.fromkeys(CM.SETS[set_name]))
    return dict(name=set_name, description="", set=set_name, features=feats, members=CM.members_for(feats, kinds), cv=dict(DEFAULT_CV), calibrate=False)


def validate(spec: dict, tab: pd.DataFrame | None = None) -> list[str]:
    """Список проблем спецификации (пусто — можно обучать)."""
    bad: list[str] = []
    mem = spec.get("members") or {}
    if not mem:
        bad.append("нет ни одного члена ансамбля")
    for n, m in mem.items():
        if m.get("kind") not in KINDS:
            bad.append(f"член «{n}»: неизвестный тип {m.get('kind')!r} (допустимо {', '.join(KINDS)})")
            continue
        if not m.get("features"):
            bad.append(f"член «{n}»: пустой список признаков")
        extra = set(m.get("params") or {}) - set(PARAMS[m["kind"]])
        if extra:
            bad.append(f"член «{n}»: неизвестные параметры {sorted(extra)} (допустимо {sorted(PARAMS[m['kind']])})")
        if tab is not None:
            miss = [c for c in m.get("features", []) if c not in tab.columns]
            if miss:
                bad.append(f"член «{n}»: нет колонок в таблице {miss} (нужен `sd enrich`?)")
    cv = spec.get("cv") or {}
    if int(cv.get("n_splits", 5)) < 2:
        bad.append("cv.n_splits ≥ 2")
    if int(cv.get("repeats", 1)) < 1:
        bad.append("cv.repeats ≥ 1")
    return bad


def spec_features(spec: dict) -> list[str]:
    return sorted({c for m in spec["members"].values() for c in m["features"]})


def _groups(tab: pd.DataFrame, by_scene: bool) -> np.ndarray:
    from .oof_eval import scene_group

    return (tab.video.map(scene_group) if by_scene else tab.video).to_numpy()


def _prepare(tab: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    t = tab[tab.y.notna()].reset_index(drop=True)
    return t, t.y.to_numpy(float).astype(int)


def cv_predict(tab: pd.DataFrame, spec: dict, train_frac: float = 1.0, permute: Sequence[str] = (), shuffle_labels: bool = False, seed: int | None = None) -> dict[str, np.ndarray]:
    """Out-of-fold оценки каждого члена и ансамбля (`__ensemble__`). Фолды — по видео (или по сцене). `train_frac` < 1 берёт долю обучающих групп в каждом фолде (кривая обучения);
    `permute` — колонки, перемешиваемые среди тестовых строк (важность); `shuffle_labels` — контроль перестановкой (метки обучения перемешаны между циклами)."""
    t, y = _prepare(tab)
    cv = {**DEFAULT_CV, **(spec.get("cv") or {})}
    sd = int(cv["seed"] if seed is None else seed)
    groups = _groups(t, bool(cv["by_scene"]))
    mem = spec["members"]
    out = {n: np.zeros(len(y)) for n in mem}
    cnt = np.zeros(len(y))
    rng = np.random.default_rng(sd + 12345)
    for r in range(int(cv["repeats"])):
        folds = StratifiedGroupKFold(n_splits=min(int(cv["n_splits"]), len(set(groups))), shuffle=True, random_state=sd + r)
        for tr, te in folds.split(t, y, groups):
            if train_frac < 1.0:
                ug = np.array(sorted(set(groups[tr])))
                keep = set(rng.choice(ug, size=max(2, int(round(len(ug) * train_frac))), replace=False))
                tr = np.array([i for i in tr if groups[i] in keep])
            ytr = y[tr]
            if len(np.unique(ytr)) < 2:
                continue
            if shuffle_labels:
                ytr = rng.permutation(ytr)
            Xte = t.iloc[te].copy()
            for c in permute:
                Xte[c] = rng.permutation(Xte[c].to_numpy())
            for n, s in mem.items():
                st = B.fit_member(s["kind"], t.iloc[tr], ytr, s["features"], seed=sd + r, **(s.get("params") or {}))
                out[n][te] += B.predict_member(st, Xte)
            cnt[te] += 1
    out = {n: v / np.maximum(cnt, 1) for n, v in out.items()}
    out["__ensemble__"] = np.mean([out[n] for n in mem], axis=0)
    return out


def fit_all_scores(tab: pd.DataFrame, spec: dict) -> dict[str, np.ndarray]:
    """Оценки на ТЕХ ЖЕ данных, на которых модель обучена (resubstitution): верхняя граница, с которой сравнивается oof."""
    t, y = _prepare(tab)
    out = {}
    for n, s in spec["members"].items():
        st = B.fit_member(s["kind"], t, y, s["features"], seed=int(spec.get("cv", {}).get("seed", 0)), **(s.get("params") or {}))
        out[n] = B.predict_member(st, t)
    out["__ensemble__"] = np.mean([out[n] for n in spec["members"]], axis=0)
    return out


def _boot_auc(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ug = np.array(sorted(set(groups)))
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug))])
        if len(np.unique(y[pick])) == 2:
            vals.append(_auc(y[pick], s[pick]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def diagnose(tab: pd.DataFrame, spec: dict, n_perm: int = 5, curve: Sequence[float] = (0.4, 0.6, 0.8, 1.0), importance: bool = True, progress=None) -> dict[str, Any]:
    """Полная диагностика переобучения (см. docstring модуля). Возвращает словарь с таблицами `members`, `curve`, `importance` и числами."""
    t, y = _prepare(tab)
    if len(np.unique(y)) < 2:
        raise ValueError("в таблице нет обоих классов (затяжка / не затяжка): разметьте жесты и выполните `sd enrich`")
    groups = _groups(t, bool({**DEFAULT_CV, **spec.get("cv", {})}["by_scene"]))
    steps = 2 + n_perm + len(curve) + (len(spec_features(spec)) if importance else 0)
    done = [0]

    def tick(msg: str) -> None:
        done[0] += 1
        if progress:
            progress(done[0], steps, msg)

    oof = cv_predict(t, spec)
    tick("кросс-валидация")
    allfit = fit_all_scores(t, spec)
    tick("обучение на всех")
    rows = []
    for n in [*spec["members"], "__ensemble__"]:
        lo, hi = _boot_auc(y, oof[n], groups) if n == "__ensemble__" else (np.nan, np.nan)
        a_oof, a_tr = _auc(y, oof[n]), _auc(y, allfit[n])
        rows.append(dict(член="ансамбль" if n == "__ensemble__" else n, AUC_oof=round(a_oof, 3), AUC_train=round(a_tr, 3), разрыв=round(a_tr - a_oof, 3),
                         oof_lo=round(lo, 3) if np.isfinite(lo) else None, oof_hi=round(hi, 3) if np.isfinite(hi) else None))
    members = pd.DataFrame(rows)
    perm = []
    for i in range(n_perm):
        perm.append(_auc(y, cv_predict(t, spec, shuffle_labels=True, seed=100 + i)["__ensemble__"]))
        tick("контроль перестановкой")
    cur = []
    for f in curve:
        cur.append(dict(доля_видео=f, AUC_oof=round(_auc(y, cv_predict(t, spec, train_frac=f)["__ensemble__"]), 3)))
        tick("кривая обучения")
    imp = pd.DataFrame(columns=["признак", "падение_AUC"])
    if importance:
        base = _auc(y, oof["__ensemble__"])
        rows = []
        for c in spec_features(spec):
            drops = [base - _auc(y, cv_predict(t, spec, permute=[c], seed=200 + k)["__ensemble__"]) for k in range(2)]
            rows.append(dict(признак=c, падение_AUC=round(float(np.mean(drops)), 4)))
            tick(f"важность {c}")
        imp = pd.DataFrame(rows).sort_values("падение_AUC", ascending=False, ignore_index=True)
    ens = members[members.член == "ансамбль"].iloc[0]
    perm_mean = float(np.mean(perm)) if perm else float("nan")
    verdict = []
    if ens.разрыв > 0.15:
        verdict.append(f"разрыв train−oof {ens.разрыв:.2f} велик: модель запоминает; уменьшите число признаков, усильте регуляризацию (C↓, alpha↑, num_leaves↓, min_child_samples↑)")
    if perm and perm_mean > 0.58:
        verdict.append(f"AUC при перемешанных метках {perm_mean:.2f} выше 0.5: возможна утечка между фолдами — включите схему «по сценам»")
    if cur and cur[-1]["AUC_oof"] - cur[0]["AUC_oof"] > 0.05:
        verdict.append("кривая обучения ещё растёт: больше размеченных видео улучшит качество")
    if importance and len(imp) and (imp.падение_AUC <= 0).sum() >= max(2, len(imp) // 2):
        verdict.append(f"у {(imp.падение_AUC <= 0).sum()} из {len(imp)} признаков перемешивание не снижает AUC: они либо бесполезны, либо дублируют друг друга — проверьте, убрав по одному")
    if not verdict:
        verdict.append("явных признаков переобучения нет (но выборка мала — интервал AUC широк)")
    return dict(members=members, perm_auc=perm, perm_mean=perm_mean, curve=pd.DataFrame(cur), importance=imp, verdict=verdict, n=int(len(y)), n_pos=int(y.sum()), n_groups=int(len(set(groups))),
                oof=oof["__ensemble__"], y=y, table=t)


def train(tab: pd.DataFrame, spec: dict, out_dir: Path, meta: dict | None = None) -> dict:
    """Обучает и сохраняет пакет по спецификации (без pickle). Возвращает manifest."""
    t, y = _prepare(tab)
    problems = validate(spec, t)
    if problems:
        raise ValueError("; ".join(problems))
    cv = {**DEFAULT_CV, **(spec.get("cv") or {})}
    groups = _groups(t, bool(cv["by_scene"]))
    mem = {n: dict(kind=m["kind"], features=list(m["features"]), params=dict(m.get("params") or {})) for n, m in spec["members"].items()}
    info = dict(spec={k: spec.get(k) for k in ("name", "description", "set")}, features_requested=spec_features(spec), by_scene=bool(cv["by_scene"]), **(meta or {}))
    return B.train_bundle(t, y, groups, mem, out_dir, meta=info, n_splits=int(cv["n_splits"]), repeats=int(cv["repeats"]), seed=int(cv["seed"]), calibrate=bool(spec.get("calibrate", False)))


# ---------------------------------------------------------------- спецификации на диске
def _check_name(name: str) -> str:
    n = str(name).strip()
    if not n or not all(c.isalnum() or c in "-_." for c in n):
        raise ValueError("имя: буквы, цифры, - _ .")
    return n


def save_spec(spec: dict, directory: Path | None = None) -> Path:
    d = directory or DIR
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{_check_name(spec['name'])}.yaml"
    f.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return f


def load_spec(name: str, directory: Path | None = None) -> dict:
    return yaml.safe_load(((directory or DIR) / f"{_check_name(name)}.yaml").read_text(encoding="utf-8"))


def list_specs(directory: Path | None = None) -> list[str]:
    d = directory or DIR
    return sorted(f.stem for f in d.glob("*.yaml")) if d.exists() else []


def load_table(enriched: bool = True) -> pd.DataFrame:
    """Таблица всех циклов полных запусков с метками жестов и (если `enriched`) признаками `sd enrich` — то же, что получает `sd train-bundle --enriched`."""
    from . import enrich as EN
    from . import oof_eval as O
    from . import start_eval as SE

    clips = O.collect()
    tab = O.attach_labels(clips, pd.read_csv(SE.GT_CSV, encoding="utf-8-sig"))
    return EN.attach(tab, list(clips)) if enriched else tab
