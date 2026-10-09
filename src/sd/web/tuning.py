"""Проверка и подгонка моделей из веб-интерфейса: оценка профиля на папках (Event F1), разбор прошлых прогонов, анализ порога цикла по OOF-оценкам пакета,
спецификация и диагностика классификатора, обучение пакета. Тяжёлое идёт фоновыми задачами (`Ops`), интерфейс опрашивает прогресс."""
from __future__ import annotations

import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .. import classifier as C
from .. import cycle_models as CM
from .. import experiments as XP
from .. import profiles as PR
from .. import runner as RN
from ..paths import MODELS, from_portable, repo_path


def clean(o: Any) -> Any:
    """Любые numpy/pandas значения → JSON (NaN → None)."""
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict("records"))
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


class Op:
    def __init__(self, kind: str, title: str):
        self.id, self.kind, self.title = uuid.uuid4().hex[:10], kind, title
        self.state, self.progress, self.msg, self.error, self.result = "running", 0.0, "старт", None, None
        self.created = time.time()

    def report(self, i: float, n: float, msg: str = "") -> None:
        self.progress, self.msg = min(0.99, i / n if n else 0.0), msg or self.msg

    def to_dict(self) -> dict:
        return dict(id=self.id, kind=self.kind, title=self.title, state=self.state, progress=round(self.progress, 3), msg=self.msg, error=self.error, result=clean(self.result), seconds=round(time.time() - self.created, 1))


class Tuning:
    def __init__(self, app):
        self.app = app
        self.ops: dict[str, Op] = {}
        self._table_cache: dict[tuple, pd.DataFrame] = {}
        self._lock = threading.Lock()          # тяжёлые операции (обучение, оценка) — по одной: память и процессор ограничены

    # ------------------------------------------------------------------ фоновые операции
    def start(self, kind: str, title: str, fn: Callable[[Op], Any]) -> Op:
        op = Op(kind, title)
        self.ops[op.id] = op

        def run() -> None:
            with self._lock:
                try:
                    op.result = fn(op)
                    op.state, op.progress = "done", 1.0
                except Exception as e:      # noqa: BLE001 — любая ошибка показывается пользователю текстом
                    op.state, op.error = "error", f"{type(e).__name__}: {e}"[:600]

        threading.Thread(target=run, daemon=True).start()
        return op

    def op(self, oid: str) -> Op:
        from .api import NotFound

        if oid not in self.ops:
            raise NotFound(oid)
        return self.ops[oid]

    # ------------------------------------------------------------------ данные классификатора
    def table(self, enriched: bool = True, refresh: bool = False) -> pd.DataFrame:
        key = (bool(enriched),)
        if refresh or key not in self._table_cache:
            self._table_cache[key] = C.load_table(enriched)
        return self._table_cache[key]

    def table_info(self, enriched: bool = True, refresh: bool = False) -> dict:
        try:
            tab = self.table(enriched, refresh)
        except Exception as e:      # noqa: BLE001 — нет запусков/меток: подсказываем, что сделать
            return dict(ok=False, error=f"{type(e).__name__}: {e}", sets=list(CM.SETS), kinds=C.KINDS, params=C.PARAMS, cv=C.DEFAULT_CV)
        lab = tab[tab.y.notna()]
        cols = [c for c in tab.columns if c not in CM.CONFOUNDS and pd.api.types.is_numeric_dtype(tab[c]) and c not in ("y", "peak_t", "tid", "start", "end")]
        return dict(ok=True, n=int(len(tab)), n_labeled=int(len(lab)), n_pos=int((lab.y == 1).sum()), n_neg=int((lab.y == 0).sum()), n_videos=int(lab.video.nunique()),
                    columns=[dict(name=c, filled=round(float(tab[c].notna().mean()), 3)) for c in cols], sets={k: list(dict.fromkeys(v)) for k, v in CM.SETS.items()},
                    kinds=C.KINDS, params=C.PARAMS, cv=C.DEFAULT_CV, saved=C.list_specs())

    def default_spec(self, set_name: str, kinds: list[str], enriched: bool = True) -> dict:
        if set_name not in CM.SETS:
            raise ValueError(f"неизвестный набор признаков {set_name!r}")
        try:
            tab = self.table(enriched)
        except Exception:      # noqa: BLE001
            tab = None
        spec = C.default_spec(set_name, kinds or ("lr", "gb", "nn"), tab)
        return dict(spec, problems=C.validate(spec, tab))

    def validate_spec(self, spec: dict, enriched: bool = True) -> list[str]:
        try:
            tab = self.table(enriched)
        except Exception:      # noqa: BLE001
            tab = None
        return C.validate(spec, tab)

    def diagnose(self, spec: dict, n_perm: int, importance: bool, enriched: bool = True) -> Op:
        tab = self.table(enriched)
        tab = tab[tab.y.notna()].reset_index(drop=True)
        return self.start("diagnose", "Диагностика переобучения", lambda op: C.diagnose(tab, spec, n_perm=n_perm, importance=importance, progress=lambda i, n, m: op.report(i, n, m)))

    def train(self, spec: dict, name: str, attach: str | None, enriched: bool = True) -> Op:
        if not name or not all(ch.isalnum() or ch in "-_." for ch in name) or name.startswith("."):
            raise ValueError("имя пакета: буквы, цифры, - _ .")
        tab = self.table(enriched)
        tab = tab[tab.y.notna()].reset_index(drop=True)

        def run(op: Op) -> dict:
            op.report(0, 1, "обучение")
            man = C.train(tab, spec, MODELS / "cycle" / name, meta=dict(source="web"))
            attached = None
            if attach:
                p = PR.load(attach)
                PR.save(PR.Profile(name=p.name, description=p.description, base=p.base, config=p.config, options={**p.options, "cycle_bundle": f"models/cycle/{name}"}))
                attached = attach
            return dict(bundle=f"models/cycle/{name}", auc=man["cv"].get("auc_ensemble"), n=man["n"], n_pos=man["n_pos"], attached=attached)

        return self.start("train", f"Обучение пакета {name}", run)

    # ------------------------------------------------------------------ анализ пакета по OOF
    def bundle_analysis(self, name: str, member: str = "oof_ensemble") -> dict:
        """Качество оценки цикла по порогам (OOF: оценки получены моделью, не видевшей этот клип) и рекомендация порога по F1 цикла."""
        from ..solver import describe_bundle

        if not all(ch.isalnum() or ch in "-_." for ch in name) or name.startswith("."):
            raise ValueError("имя пакета")
        d = repo_path(f"models/cycle/{name}")
        info = describe_bundle(d)
        f = d / "oof.csv"
        out = dict(name=name, info=info, thresholds=[], members=[], hist=None, best=None)
        if not f.exists():
            out["note"] = "в пакете нет oof.csv: пороги по нему не посчитать (пакет обучен без сохранения OOF)"
            return clean(out)
        df = pd.read_csv(f)
        cols = [c for c in df.columns if c.startswith("oof_")]
        member = member if member in df.columns else (cols[-1] if cols else None)
        if member is None or "y" not in df:
            out["note"] = "oof.csv неполный"
            return clean(out)
        y, s = df["y"].to_numpy(int), df[member].to_numpy(float)
        ok = np.isfinite(s)
        y, s = y[ok], s[ok]
        from sklearn.metrics import roc_auc_score

        rows = []
        for th in np.round(np.arange(0.05, 0.96, 0.05), 2):
            pred = s >= th
            tp, fp, fn, tn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
            p, r = (tp / (tp + fp) if tp + fp else None), (tp / (tp + fn) if tp + fn else None)
            rows.append(dict(threshold=float(th), tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r, f1=(2 * p * r / (p + r) if p and r else 0.0)))
        out["thresholds"] = rows
        best = max(rows, key=lambda r: r["f1"])
        near = [r for r in rows if r["f1"] >= best["f1"] - 0.02]        # плато: пороги, не хуже лучшего на 0.02, — берём середину, чтобы не стоять на пике
        out["best"] = dict(threshold=best["threshold"], f1=best["f1"], plateau=[near[0]["threshold"], near[-1]["threshold"]], suggested=near[len(near) // 2]["threshold"])
        out["members"] = [dict(name=c, auc=float(roc_auc_score(df["y"][df[c].notna()], df[c].dropna()))) for c in cols if df[c].notna().sum() > 5 and df["y"][df[c].notna()].nunique() == 2]
        bins = np.linspace(0, 1, 11)
        out["hist"] = dict(bins=bins.tolist(), pos=np.histogram(s[y == 1], bins)[0].tolist(), neg=np.histogram(s[y == 0], bins)[0].tolist())
        out["member"], out["n"], out["n_pos"] = member, int(len(y)), int((y == 1).sum())
        return clean(out)

    # ------------------------------------------------------------------ оценка профиля на папках и прошлые прогоны
    def evaluate(self, profile: str, folders: list[str], classifier: str | None, mode: str, policy: str, only_labeled: bool, max_sec: float | None, n_boot: int) -> Op:
        dirs = [self.app.resolve(f) for f in folders]
        if not dirs:
            raise ValueError("выберите хотя бы одну папку")
        prof = self.app.load_profile(profile)

        def run(op: Op) -> dict:
            out = RN.evaluate_dirs(dirs, prof, classifier=classifier or None, mode=mode, policy=policy, only_labeled=only_labeled, max_sec=max_sec or None, n_boot=n_boot,
                                   name=f"web_{prof.name}", progress=lambda i, n, m: op.report(i, n, m))
            return self.experiment_detail(out.run_id)

        return self.start("eval", f"Оценка: {prof.name}", run)

    def experiments(self, limit: int = 30) -> list[dict]:
        df = XP.list_runs()
        return clean(df.head(limit)) if len(df) else []

    def experiment_detail(self, rid: str) -> dict:
        from .api import NotFound

        if not (XP.EXP_DIR / rid / "meta.json").exists():
            raise NotFound(f"прогон {rid}")
        rep, meta, ev, gt = XP.load_run(rid)
        clips = []
        for c in meta.get("clips", []):
            try:
                fid = self.app.fid(from_portable(c["path"]))
            except Exception:      # noqa: BLE001 — видео вне подключённых папок: открыть в разборе нельзя
                fid = None
            clips.append(dict(clip_id=c["clip_id"], duration=c.get("duration"), error=c.get("error"), cycles=c.get("cycles"), people=c.get("people"), video=fid))
        by = {c["clip_id"]: c for c in clips}
        errs = clean(rep.errors) if len(rep.errors) else []
        for e in errs:
            e["video"] = (by.get(e.get("clip_id")) or {}).get("video")
        return clean(dict(id=rid, name=meta.get("name"), created=meta.get("created"), mode=rep.mode, describe=meta.get("describe"), fingerprint=meta.get("fingerprint"), settings=rep.settings,
                          metrics=rep.metrics, ci=rep.ci, budget=rep.budget, plateau=rep.plateau, curve=rep.curve, per_clip=rep.per_clip, errors=errs, notes=rep.notes, clips=clips,
                          dirs=[Path(d).name for d in meta.get("dirs", [])], skipped_unlabeled=len(meta.get("skipped_unlabeled", []))))
