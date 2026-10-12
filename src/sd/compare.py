"""Аналитика по подходам: сравнение многих прогонов оценки сразу — какие параметры различаются, где по клипам подходы расходятся, какие ошибки у кого,
кривые порога, «что даёт параметр» по всему журналу и попарные сравнения с бутстрэпом.

Подход = прогон оценки (`outputs/experiments/<id>`) или группа прогонов с одним тегом (`experiments.set_notes(tag=…)`). Сравнивать корректно только прогоны
с одним эталоном: `catalog()` несёт `gt_fingerprint`, и интерфейс по умолчанию оставляет только прогоны с тем же эталоном, что у выбранного опорного.
Всё читается из сохранённых файлов прогонов — модели не запускаются.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import experiments as XP

PARAM_PREFIX = ("cfg.", "opt.")


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flat(v, kk + "."))
        else:
            out[kk] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
    return out


def catalog(root: Path | None = None) -> pd.DataFrame:
    """Все прогоны: метрики + описание + действующие параметры профиля (cfg.<ключ>, opt.<опция>) одной строкой на прогон."""
    base = root or XP.EXP_DIR
    runs = XP.list_runs(base)
    if runs.empty:
        return runs
    rows = []
    for rid in runs.id:
        try:
            m = json.loads((base / rid / "meta.json").read_text(encoding="utf-8"))
        except Exception:      # noqa: BLE001
            m = {}
        prof = m.get("profile") or {}
        r = {"id": rid}
        try:            # действующие значения (профиль поверх configs/default.yaml): «не задано в профиле» иначе смешивало бы 5 и 10 к/с
            from .profiles import Profile

            p = Profile(prof.get("name", rid), "", prof.get("base"), prof.get("config") or {}, prof.get("options") or {})
            r.update({f"cfg.{k}": v for k, v in _flat(p.cfg().to_dict()).items()})
            r.update({f"opt.{k}": v for k, v in _flat(p.opts()).items()})
        except Exception:      # noqa: BLE001 — профиль из старой версии с исчезнувшими ключами: как записано
            r.update({f"cfg.{k}": v for k, v in _flat(prof.get("config") or {}).items()})
            r.update({f"opt.{k}": v for k, v in _flat(prof.get("options") or {}).items()})
        r["n_clips_meta"] = len(m.get("clips", []))
        r["leak_clips"] = len((m.get("leak") or {}).get("clips") or []) if m.get("leak") is not None else np.nan
        r["seconds_total"] = m.get("total_seconds")
        rows.append(r)
    params = pd.DataFrame(rows)
    return runs.merge(params, on="id", how="left")


def param_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(PARAM_PREFIX)]


def varying(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    """Параметры, значение которых различается между выбранными прогонами (пропуск = значение по умолчанию профиля — тоже отличие)."""
    cols = cols if cols is not None else param_columns(df)
    out = []
    for c in cols:
        vals = df[c].map(lambda v: "∅" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        if vals.nunique() > 1:
            out.append(c)
    return out


def label(df: pd.DataFrame) -> pd.Series:
    """Короткая подпись прогона для графиков: тег (если есть) или имя, плюс дата."""
    tag = df.get("tag", pd.Series([""] * len(df), index=df.index)).fillna("")
    base = np.where(tag != "", tag, df["name"].astype(str))
    return pd.Series([f"{b} · {str(c)[5:16]}" for b, c in zip(base, df["created"])], index=df.index)


def per_clip(run_ids: list[str], root: Path | None = None, value: str = "f1") -> pd.DataFrame:
    """Клип × прогон: значение колонки per_clip (f1, tp, fp, fn, status)."""
    cols = {}
    for rid in run_ids:
        rep, *_ = XP.load_run(rid, root)
        pc = rep.per_clip
        if len(pc) and value in pc:
            cols[rid] = pc.set_index("clip_id")[value]
    return pd.DataFrame(cols)


def disagreement(f1: pd.DataFrame) -> pd.Series:
    """Насколько подходы расходятся на клипе: размах F1 (клипы без позитивов, где F1 не определён, считаются по наличию FP отдельно)."""
    return (f1.max(axis=1) - f1.min(axis=1)).fillna(0.0)


def outcome_matrix(run_ids: list[str], root: Path | None = None) -> pd.DataFrame:
    """Клип × прогон: компактный исход «TP/FP/FN» строкой (например «2/0/1») — видно, что именно поменялось."""
    parts = {}
    for rid in run_ids:
        rep, *_ = XP.load_run(rid, root)
        pc = rep.per_clip
        if len(pc):
            parts[rid] = pc.set_index("clip_id").apply(lambda r: f"{int(r.tp)}/{int(r.fp)}/{int(r.fn)}", axis=1)
    return pd.DataFrame(parts)


def error_causes(run_ids: list[str], root: Path | None = None) -> pd.DataFrame:
    """Прогон × причина ошибки: число."""
    rows = []
    for rid in run_ids:
        rep, *_ = XP.load_run(rid, root)
        er = rep.errors
        if len(er):
            for (k, c), n in er.groupby(["kind", "cause_ru"]).size().items():
                rows.append(dict(id=rid, тип=k, причина=c, n=int(n)))
    return pd.DataFrame(rows, columns=["id", "тип", "причина", "n"])


def curves(run_ids: list[str], root: Path | None = None) -> pd.DataFrame:
    """Кривые порога всех прогонов в длинном формате: id, threshold, f1, precision, recall."""
    parts = []
    for rid in run_ids:
        rep, *_ = XP.load_run(rid, root)
        if len(rep.curve):
            parts.append(rep.curve[["threshold", "f1", "precision", "recall"]].assign(id=rid, chosen=rep.settings["threshold"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["threshold", "f1", "precision", "recall", "id", "chosen"])


def effect(df: pd.DataFrame, param: str, metric: str = "f1") -> pd.DataFrame:
    """«Что даёт параметр»: по значениям параметра — сколько прогонов, средняя/лучшая/худшая метрика. Внимание: прогоны различаются и другими параметрами —
    это обзор журнала, а не контролируемый эксперимент (для вывода — пара прогонов, отличающихся только этим параметром, и парный бутстрэп)."""
    v = df[param].map(lambda x: "по умолчанию" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))
    g = df.assign(_v=v).groupby("_v")[metric].agg(["count", "mean", "max", "min"]).reset_index()
    return g.rename(columns={"_v": "значение", "count": "прогонов", "mean": "среднее", "max": "лучшее", "min": "худшее"}).sort_values("среднее", ascending=False)


def controlled_pairs(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """Пары прогонов, различающиеся ТОЛЬКО этим параметром (и с одним эталоном) — на них влияние параметра видно честно."""
    cols = [c for c in param_columns(df) if c != param]
    key = df[cols].astype(str).agg("|".join, axis=1) + "|" + df["gt_fingerprint"].astype(str)
    out = []
    for _, g in df.assign(_k=key).groupby("_k"):
        if len(g) < 2:
            continue
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            if str(a[param]) != str(b[param]):
                out.append(dict(первый=a["id"], значение_1=a[param], F1_1=a["f1"], второй=b["id"], значение_2=b[param], F1_2=b["f1"], разница=b["f1"] - a["f1"]))
    return pd.DataFrame(out)


def pairwise(run_ids: list[str], n: int = 300, root: Path | None = None) -> pd.DataFrame:
    """Матрица парных сравнений (строка − столбец, п.п. F1) с 95 % интервалом; '≈' — различие в пределах шума. Не больше 6 прогонов (n² бутстрэпов)."""
    from . import sweep as SW

    ids = list(run_ids)[:6]
    cells = pd.DataFrame("", index=ids, columns=ids)
    for a, b in itertools.permutations(ids, 2):
        try:
            r = SW.compare_saved(b, a, n=n, root=root)          # diff = a − b
        except Exception as e:      # noqa: BLE001
            cells.loc[a, b] = f"— ({type(e).__name__})"
            continue
        if r.get("diff") is None:
            cells.loc[a, b] = "мало общих клипов"
            continue
        sig = "↑" if r["lo"] > 0 else ("↓" if r["hi"] < 0 else "≈")
        cells.loc[a, b] = f"{sig} {r['diff'] * 100:+.1f} [{r['lo'] * 100:+.1f}…{r['hi'] * 100:+.1f}]" + ("" if r["same_gt"] else " ⚠эталон")
    for a in ids:
        cells.loc[a, a] = "—"
    return cells


def classifier_studies(out_root: Path | None = None) -> pd.DataFrame:
    """Результаты диагностик классификатора (outputs/classifier/*/diag.json + members.csv) одной таблицей: спецификация, источник, AUC oof/train, разрыв, перестановка."""
    from .paths import OUTPUTS

    base = out_root or (OUTPUTS / "classifier")
    rows = []
    for f in sorted(base.glob("*/diag.json"), reverse=True) if base.exists() else []:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            m = pd.read_csv(f.parent / "members.csv", encoding="utf-8-sig")
            ens = m[m["член"] == "ансамбль"].iloc[0]
        except Exception:      # noqa: BLE001
            continue
        rows.append(dict(диагностика=f.parent.name, спецификация=j.get("spec"), источник=(j.get("source") or {}).get("run") or "последние запуски", циклов=j.get("n"),
                         затяжек=j.get("n_pos"), AUC_oof=ens.get("AUC_oof"), AUC_train=ens.get("AUC_train"), разрыв=ens.get("разрыв"), перестановка=j.get("perm_mean"),
                         признаков=len({c for mm in (j.get("spec_full") or {}).get("members", {}).values() for c in mm.get("features", [])}), создана=j.get("created")))
    return pd.DataFrame(rows)


def oof_studies(analysis_dir: Path | None = None) -> pd.DataFrame:
    """Все таблицы `sd oof-eval` (outputs/analysis/oof_study*.csv) вместе: набор признаков × файл — честный F1 (вложенный) и AUC."""
    from .paths import OUTPUTS

    base = analysis_dir or (OUTPUTS / "analysis")
    parts = []
    for f in sorted(base.glob("oof_study*.csv")) if base.exists() else []:
        try:
            parts.append(pd.read_csv(f, encoding="utf-8-sig").assign(файл=f.stem))
        except Exception:      # noqa: BLE001
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
