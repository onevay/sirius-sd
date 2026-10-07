"""Журнал экспериментов: каждая оценка сохраняется с тем, что нужно для воспроизведения и сравнения (правило руководства §11.2: любое изменение — с прогоном и записью).

    outputs/experiments/<время>_<имя>/
        meta.json        профиль (модели и параметры), отпечатки профиля и эталона, папки и клипы, режим, порог, время, git
        report.json      метрики, интервал, бюджет ошибок, плато (evaluation.Report)
        curve.csv  per_clip.csv  errors.csv
        events.csv       все события всех клипов ДО фильтра по порогу (кривую порога можно пересчитать без повторного прогона)
        gt.csv           срез эталона, по которому считали
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import Report
from .paths import OUTPUTS, ROOT, slug

EXP_DIR = OUTPUTS / "experiments"
INDEX_COLUMNS = ["id", "created", "name", "mode", "role", "clips", "f1", "ci_lo", "ci_hi", "precision", "recall", "tp", "fp", "fn", "fp_per_hour", "threshold", "target_f1", "reached",
                 "dirs", "pose", "cycle_model", "vlm"]


def git_state() -> dict:
    def run(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip()

    try:
        return dict(commit=run("rev-parse", "--short", "HEAD") or None, dirty=bool(run("status", "--porcelain", "--untracked-files=no")))
    except Exception:
        return dict(commit=None, dirty=None)


def new_id(name: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{slug(name) or 'run'}"


def save_run(rep: Report, meta: dict, events: pd.DataFrame, gt: pd.DataFrame, run_id: str | None = None, root: Path | None = None) -> Path:
    rid = run_id or new_id(meta.get("name", "run"))
    d = (root or EXP_DIR) / rid
    d.mkdir(parents=True, exist_ok=True)
    rep.save(d)
    events.to_csv(d / "events.csv", index=False, encoding="utf-8-sig")
    gt.to_csv(d / "gt.csv", index=False, encoding="utf-8-sig")
    (d / "meta.json").write_text(json.dumps({**meta, "id": rid, "created": meta.get("created") or time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state()}, ensure_ascii=False, indent=1,
                                            default=str), encoding="utf-8")
    return d


def load_run(rid: str, root: Path | None = None) -> tuple[Report, dict, pd.DataFrame, pd.DataFrame]:
    d = (root or EXP_DIR) / rid
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(d / "events.csv", encoding="utf-8-sig") if (d / "events.csv").exists() else pd.DataFrame()
    gt = pd.read_csv(d / "gt.csv", encoding="utf-8-sig", dtype={"person_gt_id": str}) if (d / "gt.csv").exists() else pd.DataFrame()
    return Report.load(d), meta, ev, gt


def list_runs(root: Path | None = None) -> pd.DataFrame:
    """Сводка всех экспериментов (новые сверху)."""
    rows = []
    base = root or EXP_DIR
    for f in sorted(base.glob("*/meta.json")) if base.exists() else []:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
            r = json.loads((f.parent / "report.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        mt, ci, bd = r["metrics"], r.get("ci", {}), r.get("budget", {})
        d = m.get("describe", {})
        f1ci = ci.get("f1") or [None, None]
        rows.append(dict(id=m["id"], created=m.get("created"), name=m.get("name"), mode=r.get("mode"), role=m.get("role"), clips=mt.get("clips"), f1=mt.get("f1"), ci_lo=f1ci[0], ci_hi=f1ci[1],
                         precision=mt.get("precision"), recall=mt.get("recall"), tp=mt.get("tp"), fp=mt.get("fp"), fn=mt.get("fn"), fp_per_hour=mt.get("fp_per_hour"),
                         threshold=mt.get("threshold"), target_f1=bd.get("target"), reached=bd.get("reached"), dirs=", ".join(Path(x).name for x in m.get("dirs", [])),
                         pose=d.get("pose"), cycle_model=d.get("cycle_model"), vlm=d.get("vlm")))
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    return df.sort_values("created", ascending=False).reset_index(drop=True) if len(df) else df


def delete_run(rid: str, root: Path | None = None) -> bool:
    d = (root or EXP_DIR) / rid
    if d.is_dir() and (d / "meta.json").exists():
        shutil.rmtree(d)
        return True
    return False
