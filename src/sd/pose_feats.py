"""Признаки цикла по ПОЛОЖЕНИЮ ключевых точек и времени (группа H): где рука/локоть относительно рта и плеча, углы руки, вторая рука, поворот головы.

Дополняют кинематику `dataset.cycle_vector` (там — длительности, скорости, d_t): здесь геометрия позы в момент удержания у рта.
Все длины — в ширинах плеч `s`, углы — в градусах, лево/право зеркалятся (положительный dx = «к середине тела»), поэтому один признак
значит одно и то же для обеих рук. Берутся из сырых ключевых точек трека (`Tracks.kp`), ряд признаков (series) не меняется — кэш не ломается.
Признаки времени: положение пика в треке, номер цикла в треке, доля удержания; `track_len` как признак клипа не используется (смешение со сценой).
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd

from .features import LELB, LSHO, LWRI, NOSE, RELB, RSHO, RWRI, kp_threshold

POSE_COLS = ["p_wm_dx", "p_wm_dy", "p_wm_dist", "p_em_dx", "p_em_dy", "p_es_dx", "p_es_dy", "p_forearm_deg", "p_upperarm_deg",
             "p_elbow_peak", "p_elbow_min", "p_elbow_mean", "p_elbow_vel_max", "p_other_wrist_dy", "p_both_hands_frac", "p_head_yaw",
             "p_wrist_above_mouth", "t_peak_in_track", "cycle_index", "cycle_count_track", "hold_frac", "approach_speed"]
KEY = ["video", "tid", "start"]


def _angle_deg(v: np.ndarray, ref: np.ndarray) -> float:
    n = np.linalg.norm(v) * np.linalg.norm(ref)
    return float(np.degrees(np.arccos(np.clip(np.dot(v, ref) / n, -1, 1)))) if n > 1e-9 else float("nan")


def _med(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def pose_vector(rows: pd.DataFrame, kp: np.ndarray, ser: pd.DataFrame, c: pd.Series, track_cycles: pd.DataFrame, thr: float) -> dict:
    """Признаки одного цикла. `rows`/`kp` — детекции и точки трека, `ser` — его ряд признаков, `c` — цикл, `track_cycles` — все циклы трека."""
    out = {k: float("nan") for k in POSE_COLS}
    h = int(c.hand)
    s_t = ser[(ser.t >= c.start - 0.05) & (ser.t <= c.end + 0.05)]
    if h not in (0, 1) or s_t.empty:
        return out
    sc = _med(s_t.s)
    # ключевые точки трека на отрезке цикла (по кадрам ряда)
    pos = pd.Series(np.arange(len(rows)), index=rows.frame.values)
    idx = pos.reindex(s_t.frame.values).values
    ok = ~np.isnan(idx)
    if ok.sum() < 3 or not np.isfinite(sc) or sc <= 0:
        return out
    P = kp[idx[ok].astype(int)]                                   # (n, K, 3)
    t = s_t.t.values[ok]

    def pt(i: int) -> np.ndarray:
        a = P[:, i, :2].astype(float).copy()
        a[P[:, i, 2] < thr] = np.nan
        return a

    sh, el, wr = (LSHO, LELB, LWRI) if h == 0 else (RSHO, RELB, RWRI)
    osh, owr = (RSHO, RWRI) if h == 0 else (LSHO, LWRI)
    S, E, W, N, OS, OW = pt(sh), pt(el), pt(wr), pt(NOSE), pt(osh), pt(owr)
    mouth = s_t[["mouth_x", "mouth_y"]].values[ok]
    hold = (t >= c.mouth_in) & (t <= c.mouth_out)
    if hold.sum() < 1:
        hold = np.abs(t - c.peak_t) <= 0.3
    mid = (S + OS) / 2
    d_in = (mid - S)[:, 0]
    # куда от активного плеча лежит середина тела; плеч нет совсем → NaN (зеркалирование не определено, признаки по x остаются NaN), нулевой знак → 1
    inward = (np.sign(np.nanmedian(d_in)) if np.isfinite(d_in).any() else float("nan")) or 1.0

    def rel(a: np.ndarray, b: np.ndarray, flip: bool = True) -> np.ndarray:
        d = (a - b) / sc
        if flip:
            d[:, 0] *= inward
        return d

    wm, em, es = rel(W, mouth), rel(E, mouth), rel(E, S)
    out.update(p_wm_dx=_med(wm[hold, 0]), p_wm_dy=_med(wm[hold, 1]), p_wm_dist=_med(np.hypot(wm[hold, 0], wm[hold, 1])),
               p_em_dx=_med(em[hold, 0]), p_em_dy=_med(em[hold, 1]), p_es_dx=_med(es[hold, 0]), p_es_dy=_med(es[hold, 1]))
    fa = _med(np.degrees(np.arctan2(-(W - E)[hold, 1], (W - E)[hold, 0] * inward)))       # 90° = предплечье вертикально вверх
    ua = _med(np.degrees(np.arctan2((E - S)[hold, 1], (E - S)[hold, 0] * inward)))        # 90° = плечо вертикально вниз
    out.update(p_forearm_deg=fa, p_upperarm_deg=ua)
    ang = s_t["elbowL" if h == 0 else "elbowR"].values[ok]
    out.update(p_elbow_peak=float(ang[int(np.abs(t - c.peak_t).argmin())]), p_elbow_min=float(np.nanmin(ang)) if np.isfinite(ang).any() else float("nan"),
               p_elbow_mean=_med(ang))
    appr = (t >= c.start) & (t <= c.mouth_in)
    if appr.sum() >= 3 and np.isfinite(ang[appr]).sum() >= 3:
        v = np.abs(np.gradient(pd.Series(ang[appr]).interpolate().bfill().ffill().values, t[appr]))
        out["p_elbow_vel_max"] = float(np.nanmax(v))
        out["approach_speed"] = float((np.nanmax(s_t.d.values[ok][appr]) - c.d_min) / max(c.mouth_in - c.start, 1e-3)) if np.isfinite(s_t.d.values[ok][appr]).any() else float("nan")
    sho_y = np.nanmedian(np.where(np.isnan(OS[:, 1]), S[:, 1], (S[:, 1] + OS[:, 1]) / 2))
    out["p_other_wrist_dy"] = _med(((OW[:, 1] - sho_y) / sc)[hold])             # < 0: вторая рука выше плеч
    dO = np.linalg.norm(OW - mouth, axis=1) / sc
    dA = np.linalg.norm(W - mouth, axis=1) / sc
    out["p_both_hands_frac"] = float(np.mean((dO[hold] < 1.2) & (dA[hold] < 1.2))) if hold.any() else float("nan")
    out["p_head_yaw"] = _med(np.abs((N[:, 0] - mid[:, 0]) / sc)[hold])           # взгляд в сторону / профиль
    out["p_wrist_above_mouth"] = _med(((mouth[:, 1] - W[:, 1]) / sc)[hold])      # > 0: кисть выше рта
    tc = track_cycles.sort_values("peak_t").reset_index(drop=True)
    out.update(t_peak_in_track=float(c.peak_t - rows.t.min()), cycle_index=float(int((tc.peak_t < c.peak_t - 1e-6).sum())), cycle_count_track=float(len(tc)),
               hold_frac=float(c.hold / max(c.end - c.start, 1e-3)))
    return out


def pose_rows(vname: str, tr, cfg: dict, ser: pd.DataFrame, cyc: pd.DataFrame) -> list[dict]:
    """Признаки H всех циклов одного запуска (люди ниже min_person_height_px пропускаются). Ключ видео — имя каталога запуска, как в `analysis.cycles_frame`."""
    summ = tr.summary(cfg["video"]["min_person_height_px"])
    cyc = cyc[cyc.tid.isin(summ[~summ.ignore_small].tid)]
    thr = kp_threshold(cfg)
    out = []
    for tid, g in cyc.groupby("tid"):
        rows, kp = tr.of(int(tid))
        s = ser[ser.tid == tid].reset_index(drop=True)
        for c in g.itertuples():
            vec = pose_vector(rows, kp, s, pd.Series(c._asdict()), g, thr)
            out.append(dict(video=vname, tid=int(tid), start=round(float(c.start), 3), **vec))
    return out


def pose_table(runs: list[tuple[str, Path]] | None = None, progress=None) -> pd.DataFrame:
    """Признаки H по всем циклам всех запусков (текущие пороги default.yaml); ключ (video, tid, start) как у остальных таблиц."""
    from . import stages
    from .calibrate import cfg_current_for_run, list_runs
    from .tracks import Tracks

    rows_out: list[dict] = []
    runs = runs if runs is not None else list_runs(pose_sig="rtmpose")
    for i, (vname, rd) in enumerate(runs):
        tr = Tracks.load(rd / "pose")
        cfg = cfg_current_for_run(rd)
        ser = stages.stage_features(tr, cfg, rd)
        cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
        rows_out += pose_rows(vname, tr, cfg, ser, cyc)
        if progress:
            progress(i + 1, len(runs))
    return pd.DataFrame(rows_out)
