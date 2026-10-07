"""Калибровка автомата циклов по данным: перебор порогов th_in / th_out и продления кисти k.

Поза уже посчитана и лежит в кэше запусков (`outputs/runs`), поэтому перебор дешёвый: пересчитываются только признаки и циклы.
Две оценки:
  1) без разметки — число циклов в минуту по папкам (`курение` / `лжекурение`) и доля отброшенных жестов по причинам;
  2) против визуальных интервалов (`docs/review/visual_intervals.csv`): какие интервалы «рука у рта» автомат нашёл (recall), сколько
     лишних циклов (precision). Интервалы размечены просмотром кадров (не ground truth!) — это небольшая калибровочная выборка.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stages
from .config import Cfg, _wrap, set_by_path
from .paths import OUTPUTS, ROOT, weak_label
from .tracks import Tracks

INTERVALS_CSV = ROOT / "docs" / "review" / "visual_intervals.csv"


def run_cfg(rd: Path) -> Cfg:
    """Конфиг, с которым посчитан запуск (из run.json) — чтобы настройки позы/порог видимости совпадали."""
    return _wrap(json.loads((rd / "run.json").read_text(encoding="utf-8"))["cfg"])


def cfg_current_for_run(rd: Path, base: Cfg | None = None) -> Cfg:
    """Текущий конфиг (пороги циклов/признаки/события из default.yaml) + параметры позы и трекинга, с которыми посчитан запуск.

    Нужен, чтобы после калибровки порогов пересобирать циклы по старым (дорогим) результатам позы без её пересчёта.
    """
    import copy

    from .config import load_config

    cur = _wrap(copy.deepcopy(dict(base or load_config())))
    old = run_cfg(rd)
    for key in ("pose", "tracking"):
        cur[key] = old[key]
    cur["video"]["process_fps"] = old["video"]["process_fps"]
    return cur


def with_params(cfg: Cfg, th_in: float, th_out: float, k: float, extra: dict | None = None) -> Cfg:
    import copy

    c = _wrap(copy.deepcopy(dict(cfg)))
    # прямое присваивание (а не set_by_path): у старых запусков в сохранённом конфиге может не быть новых ключей
    c["cycles"]["th_in"], c["cycles"]["th_out"] = th_in, th_out
    c["features"]["hand_extend"] = k
    for key, v in (extra or {}).items():
        a, b = key.split(".")
        c[a][b] = v
    return c


def list_runs(root: Path = OUTPUTS / "runs", pose_sig: str | None = None) -> list[tuple[str, Path]]:
    out = []
    for v in sorted(root.iterdir()) if root.exists() else []:
        for rd in sorted(v.iterdir()):
            if (rd / "pose" / "tracks.parquet").exists() and (rd / "run.json").exists() and (pose_sig is None or pose_sig in rd.name):
                out.append((v.name, rd))
    return stages.drop_overlapping_runs(out)


def sweep(grid: list[tuple[float, float, float]], pose_sig: str | None = None, progress=None, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    runs = list_runs(pose_sig=pose_sig)
    for ri, (vname, rd) in enumerate(runs):
        tr = Tracks.load(rd / "pose")
        base = cfg_current_for_run(rd)   # текущие признаки/циклы/события + поза и трекинг, с которыми посчитан запуск
        info = tr.meta.get("video_info", {})
        summ = tr.summary(base["video"]["min_person_height_px"])
        ok_tids = set(summ[~summ.ignore_small].tid.tolist())
        minutes = float(sum(summ[summ.tid.isin(ok_tids)].dur)) / 60.0
        for (a, b, k) in grid:
            cfg = with_params(base, a, b, k, extra)
            ser = stages.stage_features(tr, cfg, rd)
            cyc, rej, _ = stages.stage_cycles(ser, cfg, rd)
            cyc, rej = cyc[cyc.tid.isin(ok_tids)], rej[rej.tid.isin(ok_tids)]
            rows.append(dict(video=vname, folder=weak_label(info.get("path", tr.meta.get("video", ""))) or "-", th_in=a, th_out=b, k=k,
                             person_min=round(minutes, 3), cycles=len(cyc), rejected=len(rej),
                             short=int((rej.reason == "short_touch").sum()), long=int((rej.reason == "long_hold").sum()),
                             lost=int((rej.reason == "lost").sum()), hold_med=float(cyc.hold.median()) if len(cyc) else np.nan))
        if progress:
            progress(ri + 1, len(runs))
    return pd.DataFrame(rows)


def summarize_sweep(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["th_in", "th_out", "k", "folder"]).agg(cycles=("cycles", "sum"), person_min=("person_min", "sum"), rejected=("rejected", "sum"),
                                                            short=("short", "sum"), long=("long", "sum")).reset_index()
    g["cycles_per_min"] = (g.cycles / g.person_min.clip(lower=1e-6)).round(2)
    return g.pivot_table(index=["th_in", "th_out", "k"], columns="folder", values=["cycles", "cycles_per_min"], aggfunc="sum").round(2)


def load_intervals() -> pd.DataFrame:
    if INTERVALS_CSV.exists():
        return pd.read_csv(INTERVALS_CSV)
    return pd.DataFrame(columns=["video", "t0", "t1", "kind", "note", "confidence"])


def eval_vs_intervals(grid: list[tuple[float, float, float]], kinds_pos: tuple[str, ...] = ("puff_like", "hand_at_mouth_long"),
                      pose_sig: str | None = None, extra: dict | None = None) -> pd.DataFrame:
    """Для каждой точки сетки: recall по визуальным интервалам и число циклов вне интервалов (по запускам, где интервалы есть)."""
    iv = load_intervals()
    if iv.empty:
        return pd.DataFrame()
    rows = []
    for vname, rd in list_runs(pose_sig=pose_sig):
        sub = iv[iv.video == vname]
        if sub.empty:
            continue
        tr = Tracks.load(rd / "pose")
        base = cfg_current_for_run(rd)
        for (a, b, k) in grid:
            cfg = with_params(base, a, b, k, extra)
            ser = stages.stage_features(tr, cfg, rd)
            cyc, rej, _ = stages.stage_cycles(ser, cfg, rd)
            for kind_set, name in ((kinds_pos, "positive"), (None, "any_contact")):
                s = sub if kind_set is None else sub[sub.kind.isin(kind_set)]
                if s.empty:
                    continue
                hit = 0
                for r in s.itertuples():
                    hit += bool(((cyc.start < r.t1) & (cyc.end > r.t0)).any())
                matched = [any((c.start < r.t1) and (c.end > r.t0) for r in s.itertuples()) for c in cyc.itertuples()]
                rows.append(dict(video=vname, th_in=a, th_out=b, k=k, target=name, intervals=len(s), hit=hit, cycles=len(cyc),
                                 cycles_in_target=int(np.sum(matched)), cycles_outside=int(len(cyc) - np.sum(matched))))
    return pd.DataFrame(rows)
