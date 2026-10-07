"""Массовый анализ циклов: предмет (детекторы сигареты) и видео-модели (X-CLIP / VideoMAE) по ВСЕМ найденным циклам.

Результаты пишутся по циклам (ключ: video, tid, start) в outputs/analysis/*.parquet и присоединяются к вектору признаков
(`dataset.attach_model_features`): группа D — предмет, группа E — внешний вид (X-CLIP / VideoMAE).
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd

from . import stages
from .calibrate import cfg_current_for_run, list_runs
from .evidence import cycle_object_evidence, run_evidence
from .paths import OUTPUTS
from .tracks import Tracks
from .tube import run_tube

KEY = ["video", "tid", "start"]


def cycles_frame(vname: str, rd: Path, tr: Tracks, cfg: dict, ser: pd.DataFrame, cyc: pd.DataFrame) -> pd.DataFrame:
    """Циклы одного запуска (кроме людей < min_person_height_px) в виде таблицы анализа: video, run, tid, start, end, mouth_in/out, peak_t, hold, d_min, hand, idx."""
    summ = tr.summary(cfg["video"]["min_person_height_px"])
    cyc = cyc[cyc.tid.isin(summ[~summ.ignore_small].tid)].sort_values("start").reset_index(drop=True)
    rows = [dict(video=vname, run=str(rd), tid=int(r.tid), start=round(float(r.start), 3), end=float(r.end), mouth_in=float(r.mouth_in),
                 mouth_out=float(r.mouth_out), peak_t=float(r.peak_t), hold=float(r.hold), d_min=float(r.d_min), hand=int(r.hand), idx=i)
            for i, r in enumerate(cyc.itertuples())]
    return pd.DataFrame(rows)


def collect_cycles(pose_sig: str | None = "rtmpose") -> pd.DataFrame:
    """Все циклы всех запусков (кроме людей < 80 px) с текущими порогами из default.yaml + путь запуска."""
    parts = []
    for vname, rd in list_runs(pose_sig=pose_sig):
        tr = Tracks.load(rd / "pose")
        cfg = cfg_current_for_run(rd)
        ser = stages.stage_features(tr, cfg, rd)
        cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
        parts.append(cycles_frame(vname, rd, tr, cfg, ser, cyc))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def evidence_rows(cyc: pd.DataFrame, tr: Tracks, ser: pd.DataFrame, cfg: dict, detector_ids: list[str]) -> list[dict]:
    """Признаки группы D (предмет на кропах кисть–рот) для циклов ОДНОГО запуска: строка на цикл, колонки obj_<детектор>_max_conf/_hit_frames/_hit."""
    cyc = cyc.reset_index(drop=True)
    det = run_evidence(tr.meta["video"], tr, ser, cyc, cfg, detector_ids)
    ev = cycle_object_evidence(det, cfg)
    rows = []
    for i, r in cyc.iterrows():
        sub = ev[ev.cycle == i]
        row = dict(video=r.video, tid=r.tid, start=r.start, n_det_frames=int(sub.frames.max()) if len(sub) else 0)
        for d in detector_ids:
            s = sub[sub.detector == d]
            row[f"obj_{d}_max_conf"] = float(s.max_conf.iloc[0]) if len(s) else 0.0
            row[f"obj_{d}_hit_frames"] = int(s.hit_frames.iloc[0]) if len(s) else 0
            row[f"obj_{d}_hit"] = bool(s.has_object.iloc[0]) if len(s) else False
        rows.append(row)
    return rows


def evidence_all(cycles: pd.DataFrame, detector_ids: list[str], out: Path = OUTPUTS / "analysis" / "evidence_cycles.parquet", progress=None) -> pd.DataFrame:
    res = []
    runs = list(cycles.groupby("run"))
    for ri, (rd_s, g) in enumerate(runs):
        rd = Path(rd_s)
        tr = Tracks.load(rd / "pose")
        cfg = cfg_current_for_run(rd)
        ser = stages.stage_features(tr, cfg, rd)
        res += evidence_rows(g, tr, ser, cfg, detector_ids)
        if progress:
            progress(ri + 1, len(runs))
    df = pd.DataFrame(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def pose_all(out: Path = OUTPUTS / "analysis" / "pose_cycles.parquet", progress=None, runs=None) -> pd.DataFrame:
    """Признаки группы H (положение точек, углы руки, время) по всем циклам: дёшево, ряд признаков и поза берутся из кэша запусков."""
    from .pose_feats import pose_table

    df = pose_table(runs=runs, progress=progress)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def photo_all(cycles: pd.DataFrame, bundle_dir: Path, backend: str = "auto", frames: int = 6, window: float = 2.5, scale: float = 2.5,
              out: Path = OUTPUTS / "analysis" / "photo_cycles.parquet", progress=None, zero_shot: bool = True, ctx: dict | None = None) -> pd.DataFrame:
    """Группа I: оценка «курит» фото-классификатором по кропам рта цикла (6 кадров): среднее, максимум и топ-2 по кадрам; + zero-shot CLIP (если есть эмбеддинг clip)."""
    from . import photo_clf as PC
    from . import photo_feats as PF
    from . import photo_video as PV

    bundle = PC.load_bundle(Path(bundle_dir))
    crops = PV.extract_crops(cycles, frames, window, 224, scale, progress=(lambda i, n: progress("crops", i, n)) if progress else None, ctx=ctx)
    ft = PV.frame_table(crops)
    emb = PV.embed_crops(ft, tuple(bundle["manifest"]["backbones"]) if not zero_shot else tuple(dict.fromkeys(list(bundle["manifest"]["backbones"]) + ["clip"])),
                         backend, progress=(lambda bb, i, n, dt: progress(f"embed {bb}", i, n)) if progress else None)
    per = {"p": PV.score_frames(bundle, emb)}
    if zero_shot and "clip" in emb:
        z = PF.zero_shot(emb["clip"], PF.clip_text_embeddings())
        per["zs"] = z["smoke"] - z["none"]
    df = PV.aggregate(ft, per)
    df["start"] = df.start.round(3)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def tube_all(cycles: pd.DataFrame, models: tuple[str, ...] = ("xclip",), window: float = 2.5, out: Path = OUTPUTS / "analysis" / "tube_cycles.parquet",
             progress=None) -> pd.DataFrame:
    res = []
    runs = list(cycles.groupby("run"))
    for ri, (rd_s, g) in enumerate(runs):
        rd = Path(rd_s)
        tr = Tracks.load(rd / "pose")
        cfg = cfg_current_for_run(rd)
        cyc = g.reset_index(drop=True)
        wins = [(i, int(r.tid), float(r.peak_t - window / 2), float(r.peak_t + window / 2)) for i, r in cyc.iterrows()]
        df = run_tube(tr.meta["video"], tr, wins, cfg, models)
        if len(df):
            df = df.rename(columns={"window": "idx"}).merge(cyc[["idx", "video", "start"]].rename(columns={"idx": "idx"}), on="idx")
            res.append(df)
        if progress:
            progress(ri + 1, len(runs))
    out_df = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    return out_df
