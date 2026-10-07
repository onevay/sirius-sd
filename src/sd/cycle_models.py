"""Наборы признаков и ансамбли для классификатора цикла: сравнение на размеченных циклах и обучение пакета (`bundle`).

Наборы заданы ЗАРАНЕЕ (не подбираются по AUC этих же циклов): «кинематика», «ритм», «поза/время», «предмет», «видео-модели», «VLM», «фото». Колонки-признаки КЛИПА
(`track_len`, `edge_dist`, `overlap_iou`, `brightness`, `t_peak_in_track`, `cycle_index`, `cycle_count_track`) в наборы не входят — они кодируют сцену, а не жест.
Честность оценки: фолды по видео, повторные перемешивания, интервал бутстрэпом по видео; на 68–120 циклах разница < 0.05 AUC — шум.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import numpy as np
import pandas as pd

from . import bundle as B
from .feature_auc import _auc

KIN = ["hold", "d_min", "dur", "mouth_dur", "t_approach", "t_retract", "v_approach_max", "v_retract_max", "amplitude", "elbow_hold", "elbow_range", "head_tilt_hold",
       "wrist_h_hold", "nose_disp", "wrist_disp", "hand"]
RHYTHM = ["n_cycles_pm20", "interval_prev", "interval_next", "same_hand_frac", "time_at_mouth_frac60"]
POSE = ["p_wm_dx", "p_wm_dy", "p_wm_dist", "p_em_dx", "p_em_dy", "p_es_dx", "p_es_dy", "p_forearm_deg", "p_upperarm_deg", "p_elbow_peak", "p_elbow_min", "p_elbow_mean",
        "p_elbow_vel_max", "p_other_wrist_dy", "p_both_hands_frac", "p_head_yaw", "p_wrist_above_mouth", "hold_frac", "approach_speed"]
OBJ = ["obj_any_max_conf", "obj_any_hit_frames"]
VIDEO = ["videomae_smoke_share", "videomae_smoke_minus_drink", "videomae_k_smoking", "xclip_smoke_or_vape", "xclip_smoke_logit"]
VLM = ["vlm_yesno"]
PHOTO = ["photo_zs_mean", "photo_zs_max", "photo_p_mean", "photo_p_max"]
SMALL = ["hold", "dur", "mouth_dur", "obj_any_max_conf", "p_wm_dist", "p_elbow_peak"]      # короткий список для линейной модели: по одному смыслу на сигнал
CONFOUNDS = {"track_len", "edge_dist", "overlap_iou", "brightness", "t_peak_in_track", "cycle_index", "cycle_count_track"}

SETS = {
    "fast": KIN + RHYTHM + POSE,                              # миллисекунды на цикл: только поза и время; детекторов и моделей нет
    "fast+vlm": KIN + RHYTHM + POSE + VLM,                    # + VLM (≈ 9 с на цикл): пакет для серой зоны без дорогого детектора предмета
    "kin": KIN,
    "kin+rhythm": KIN + RHYTHM,
    "kin+pose": KIN + POSE,
    "kin+obj": KIN + OBJ,
    "cheap": KIN + RHYTHM + POSE + OBJ,                       # без тяжёлых моделей: считается по позе и кропам за доли секунды на цикл
    "cheap+photo": KIN + RHYTHM + POSE + OBJ + PHOTO,
    "cheap+video": KIN + RHYTHM + POSE + OBJ + VIDEO,         # + VideoMAE/X-CLIP (≈ 6 с на цикл)
    "cheap+vlm": KIN + RHYTHM + POSE + OBJ + VLM,             # + VLM (≈ 9 с на цикл)
    "full": KIN + RHYTHM + POSE + OBJ + VIDEO + VLM,
}


def available(cols: list[str], tab: pd.DataFrame, min_frac: float = 0.5) -> list[str]:
    """Только колонки таблицы, у которых заполнено не меньше min_frac значений (иначе импутация медианой превращает признак в шум), без признаков клипа."""
    return [c for c in dict.fromkeys(cols) if c in tab.columns and c not in CONFOUNDS and tab[c].notna().mean() >= min_frac]


def members_for(feats: list[str], kinds=("lr", "gb", "nn")) -> dict[str, dict]:
    spec: dict[str, dict] = {}
    small = [c for c in SMALL if c in feats] or feats[:6]
    if "lr" in kinds:
        spec["lr_small"] = dict(kind="logreg", features=small, params=dict(C=0.3))
        if len(feats) > len(small):
            spec["lr_all"] = dict(kind="logreg", features=feats, params=dict(C=0.02))
    if "gb" in kinds:
        spec["gb"] = dict(kind="lgbm", features=feats)
    if "nn" in kinds:
        spec["nn"] = dict(kind="mlp", features=feats, params=dict(hidden=8, alpha=1.0))
    return spec


def _boot_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n: int = 400, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    v = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        a = _auc(y[pick], s[pick])
        if np.isfinite(a):
            v.append(a)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (float("nan"), float("nan"))


def compare(tab: pd.DataFrame, sets: dict[str, list[str]] | None = None, repeats: int = 5, n_splits: int = 5, kinds=("lr", "gb", "nn"), n_boot: int = 300) -> pd.DataFrame:
    """Для каждого набора признаков: AUC out-of-fold каждого члена и ансамбля (среднее по перемешиваниям фолдов) + 95% интервал по видео для ансамбля."""
    y, g = tab.y.to_numpy(int), tab.video.to_numpy()
    rows = []
    for name, cols in (sets or SETS).items():
        feats = available(cols, tab)
        if len(feats) < 2:
            continue
        spec = members_for(feats, kinds)
        oof = B.oof_members(tab.reset_index(drop=True), y, g, spec, n_splits, repeats)
        ens = np.mean([oof[k] for k in spec], axis=0)
        lo, hi = _boot_ci(y, ens, g, n_boot)
        row = dict(набор=name, признаков=len(feats), **{f"AUC {k}": _auc(y, v) for k, v in oof.items()}, **{"AUC ансамбль": _auc(y, ens), "ансамбль lo": lo, "ансамбль hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def train(tab: pd.DataFrame, set_name: str, out_dir, meta: dict | None = None, repeats: int = 5, kinds=("lr", "gb", "nn")) -> dict:
    feats = available(SETS[set_name], tab)
    spec = members_for(feats, kinds)
    return B.train_bundle(tab.reset_index(drop=True), tab.y.to_numpy(int), tab.video.to_numpy(), spec, out_dir, meta=dict(feature_set=set_name, **(meta or {})), repeats=repeats)
