"""Признаки для ВСЕГО пула кандидатов (162 жеста от нескольких генераторов), чтобы оценивать классификатор в условиях «широкого» генератора:
среди кандидатов — шум позы, чужие жесты и затяжки, как будет и на скрытом наборе, если ослабить пороги ради recall.

Те же группы признаков и те же имена колонок, что у основной таблицы (`feature_auc`): кинематика/ритм/качество (`dataset.build_cycle_table`), положение точек (`pose_feats`),
предмет (`analysis.evidence_rows`), VLM (`vlm_eval.run_vlm`, возобновляемо). Границы кандидата берутся из цикла-эталона пула (рабочий генератор, если он нашёл жест),
ряд признаков — по рабочему конфигу (k = 0): для кандидатов, найденных только мягкими порогами, это означает «как увидел бы рабочий автомат без отсева».
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from . import stages
from .calibrate import cfg_current_for_run
from .dataset import build_cycle_table
from .paths import OUTPUTS
from .pose_feats import pose_rows
from .tracks import Tracks

POOL_AN = OUTPUTS / "analysis_pool"
CYC_COLS = ["tid", "start", "mouth_in", "mouth_out", "end", "hold", "hand", "d_min", "peak_t"]


def stage_format(pool_run: pd.DataFrame) -> pd.DataFrame:
    """Кандидаты одного запуска → формат `stage_cycles` (hand: −1 → 0, чтобы векторы кинематики считались; NaN-границы заполняются от старта)."""
    c = pool_run.copy()
    c["hand"] = c.hand.where(c.hand.isin([0, 1]), 0).astype(int)
    c["mouth_in"] = c.mouth_in.fillna(c.start + 0.5)
    c["mouth_out"] = c.mouth_out.fillna(c.end - 0.3)
    c["hold"] = c.hold.fillna(c.mouth_out - c.mouth_in)
    c["d_min"] = c.d_min.fillna(0.5)
    return c[CYC_COLS].reset_index(drop=True)


def build(pool: pd.DataFrame, out: Path = POOL_AN, detectors: tuple[str, ...] = ("smoking_yolo11m_beehzod", "smoking_yolo26s_basant18"), progress=None) -> Path:
    """Считает таблицы признаков по пулу в `out` (cycles_all, dataset_cycles, pose_cycles, evidence_cycles)."""
    out.mkdir(parents=True, exist_ok=True)
    cyc_rows, ds_rows, pose_out, ev_out = [], [], [], []
    runs = list(pool.groupby("run"))
    for i, (rd_s, g) in enumerate(runs):
        rd = Path(rd_s)
        vname = g.video.iloc[0]
        tr = Tracks.load(rd / "pose")
        cfg = cfg_current_for_run(rd)
        ser = stages.stage_features(tr, cfg, rd)
        cyc = stage_format(g.sort_values("start"))
        frame = pd.DataFrame([dict(video=vname, run=str(rd), tid=int(r.tid), start=round(float(r.start), 3), end=float(r.end), mouth_in=float(r.mouth_in), mouth_out=float(r.mouth_out),
                                   peak_t=float(r.peak_t), hold=float(r.hold), d_min=float(r.d_min), hand=int(r.hand), idx=k) for k, r in enumerate(cyc.itertuples())])
        cyc_rows.append(frame)
        ds_rows.append(build_cycle_table(Path(tr.meta["video"]), tr, ser, cyc))
        pose_out += pose_rows(vname, tr, cfg, ser, cyc)
        if detectors:
            ev_out += A.evidence_rows(frame, tr, ser, cfg, list(detectors))
        if progress:
            progress(i + 1, len(runs))
    pd.concat(cyc_rows, ignore_index=True).to_parquet(out / "cycles_all.parquet", index=False)
    pd.concat([d for d in ds_rows if len(d)], ignore_index=True).to_parquet(out / "dataset_cycles.parquet", index=False)
    pd.DataFrame(pose_out).to_parquet(out / "pose_cycles.parquet", index=False)
    if ev_out:
        pd.DataFrame(ev_out).to_parquet(out / "evidence_cycles.parquet", index=False)
    return out


def labelled_table(gt: pd.DataFrame, pool: pd.DataFrame, an: Path = POOL_AN, vlm: Path | None = None) -> pd.DataFrame:
    """Признаки кандидатов пула + метки: y = 1 для затяжки, 0 для остальных (без «неясно»); колонки `label`, `source`, `start_quality` из gt."""
    from . import feature_auc as FA
    from . import start_eval as SE

    tab = FA.all_cycles_features(an=an, vlm=vlm, dataset=an / "dataset_cycles.parquet")
    p = SE.attach_gt(pool, gt)[["video", "tid", "start", "label", "start_quality", "found_by"]]
    p["start"] = p.start.round(3)
    t = tab.assign(start=tab.start.round(3)).merge(p, on=["video", "tid", "start"], how="left")
    t = t[t.label.notna() & (t.label != "unsure")].copy()
    t["y"] = (t.label == "smoke").astype(int)
    return t.reset_index(drop=True)
