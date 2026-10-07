"""Признаки жеста по треку: d_t (запястье–рот в ширинах плеч), скорость, угол локтя, наклон головы, видимость.

Все пространственные признаки нормированы на масштаб тела `s` (ширина плеч), все временные — в секундах (руководство §4.2, §5.3).
Пропуски — NaN (бустинг их понимает), малые пропуски (<= interp_gap_sec) интерполируются.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

# индексы COCO-17
NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHIP, RHIP = range(13)

COCO_SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                 (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def _odd(n: float, lo: int = 3) -> int:
    n = max(int(round(n)), lo)
    return n if n % 2 == 1 else n + 1


def _mean2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Среднее двух массивов, игнорируя NaN (без предупреждений numpy)."""
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2))


def interp_small_gaps(t: np.ndarray, y: np.ndarray, max_gap: float) -> np.ndarray:
    """Линейно заполняет серии NaN, если они ограничены валидными точками и длятся <= max_gap секунд."""
    y = y.copy()
    isn = np.isnan(y)
    if not isn.any() or isn.all():
        return y
    idx = np.flatnonzero(isn)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        a, b = run[0], run[-1]
        if a == 0 or b == len(y) - 1:
            continue  # на краях не экстраполируем
        if t[b + 1] - t[a - 1] <= max_gap:
            y[a:b + 1] = np.interp(t[a:b + 1], [t[a - 1], t[b + 1]], [y[a - 1], y[b + 1]])
    return y


def smooth_segments(y: np.ndarray, med_n: int, sg_n: int, poly: int) -> np.ndarray:
    """Медианный фильтр + Савицкий–Голей отдельно на каждом непрерывном (без NaN) участке."""
    out = np.full_like(y, np.nan, dtype=np.float64)
    ok = ~np.isnan(y)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        seg = y[run].astype(np.float64)
        n = len(seg)
        odd_n = n if n % 2 == 1 else n - 1
        if n >= 3:
            seg = median_filter(seg, size=min(med_n, odd_n), mode="nearest")
        w = min(sg_n, odd_n)
        if w >= poly + 2 and w >= 3:
            seg = savgol_filter(seg, window_length=w, polyorder=poly, mode="interp")
        out[run] = seg
    return out


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Угол в точке b между отрезками b->a и b->c, градусы; a,b,c: (n,2)."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    cos = (v1 * v2).sum(1) / np.maximum(n1 * n2, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def kp_threshold(cfg: dict) -> float:
    """Порог видимости ключевой точки. У RTMPose (метод уточнения rtmpose-*) шкала уверенности другая, чем у YOLO — свой порог."""
    r = cfg["pose"]["refine"]
    if r["enabled"] and str(r.get("method", "yolo")).startswith("rtmpose"):
        return float(r.get("kp_conf_min", cfg["pose"]["kp_conf_min"]))
    return float(cfg["pose"]["kp_conf_min"])


def track_grid(frame_t: pd.DataFrame, f0: int, f1: int) -> pd.DataFrame:
    """Все обработанные кадры между первым и последним кадром трека (пропуски детекций станут NaN)."""
    return frame_t[(frame_t.frame >= f0) & (frame_t.frame <= f1)].sort_values("frame").reset_index(drop=True)


def build_series(grid: pd.DataFrame, rows: pd.DataFrame, kp: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Временной ряд признаков одного трека.

    grid — кадры [frame, t] без пропусков; rows — детекции трека [frame, x1,y1,x2,y2, score]; kp — (len(rows), K, 3).
    """
    f, thr = cfg["features"], kp_threshold(cfg)
    n = len(grid)
    pos = pd.Series(np.arange(len(rows)), index=rows.frame.values)
    take = pos.reindex(grid.frame.values).values  # индекс строки в rows или NaN
    have = ~np.isnan(take)
    ti = take[have].astype(int)
    P = np.full((n, kp.shape[1], 3), np.nan, np.float32)
    P[have] = kp[ti]
    box = np.full((n, 4), np.nan, np.float32)
    box[have] = rows[["x1", "y1", "x2", "y2"]].values[ti]
    t = grid.t.values.astype(np.float64)
    dt = float(np.median(np.diff(t))) if n > 2 else 0.1
    win1s = int(max(3, round(1.0 / dt)))

    xy, cf = P[..., :2], P[..., 2]
    vis = (cf >= thr) & np.isfinite(xy).all(-1)

    def pt(i: int) -> np.ndarray:
        return np.where(vis[:, i:i + 1], xy[:, i], np.nan)

    H = (box[:, 3] - box[:, 1]).astype(np.float64)
    Hs = pd.Series(H).rolling(win1s, min_periods=1, center=True).median().values  # устойчивая высота человека

    lsho, rsho, nose = pt(LSHO), pt(RSHO), pt(NOSE)
    sho_w = np.linalg.norm(lsho - rsho, axis=1)
    s = np.clip(sho_w, f["scale_min_h"] * Hs, f["scale_max_h"] * Hs)  # NaN остаётся NaN
    s = np.where(np.isnan(s), f["fallback_scale_h"] * Hs, s)
    s = pd.Series(s).rolling(win1s, min_periods=1, center=True).median().values  # масштаб стабилен во времени

    both_sho = ~np.isnan(lsho).any(1) & ~np.isnan(rsho).any(1)
    mid_sho = np.where(both_sho[:, None], (lsho + rsho) / 2, np.nan)
    u = mid_sho - nose
    u = np.where(np.isnan(u), np.stack([np.zeros(n), 0.12 * Hs], 1), u)  # плечи не видны: вниз на ~12% роста
    mouth = nose + f["mouth_offset"] * u

    med_n, sg_n, poly = _odd(f["median_sec"] / dt), _odd(f["smooth_sec"] / dt), f["smooth_poly"]
    k_ext = float(f.get("hand_extend", 0.0))
    dd, hand_xy = {}, {}
    for name, wi, ei in (("L", LWRI, LELB), ("R", RWRI, RELB)):
        w, e = pt(wi), pt(ei)
        # точка кисти: запястье, продлённое вдоль предплечья (приближает положение пальцев с сигаретой); без локтя — просто запястье
        h = np.where(np.isnan(e).any(1, keepdims=True), w, w + k_ext * (w - e)) if k_ext else w
        hand_xy[name] = h
        d_raw = np.linalg.norm(h - mouth, axis=1) / np.maximum(s, 1e-6)
        dd[f"{name}_raw"] = d_raw
        dd[name] = smooth_segments(interp_small_gaps(t, d_raw, f["interp_gap_sec"]), med_n, sg_n, poly)
    dL, dR = dd["L"], dd["R"]
    d = np.fmin(dL, dR)
    d_raw_min = np.fmin(dd["L_raw"], dd["R_raw"])
    hand = np.where(np.isnan(d), -1, np.where(np.nan_to_num(dL, nan=1e9) <= np.nan_to_num(dR, nan=1e9), 0, 1))
    v = np.full(n, np.nan)
    ok = ~np.isnan(d)
    if ok.sum() >= 3:
        v[ok] = np.gradient(d[ok], t[ok])

    ang = {}
    for name, (sh, el, wr) in (("L", (LSHO, LELB, LWRI)), ("R", (RSHO, RELB, RWRI))):
        a = _angle(pt(sh), pt(el), pt(wr))
        ang[name] = smooth_segments(interp_small_gaps(t, a, f["interp_gap_sec"]), 3, 3, 1)

    ear_y = _mean2(np.where(vis[:, LEAR], xy[:, LEAR, 1], np.nan), np.where(vis[:, REAR], xy[:, REAR, 1], np.nan))
    tilt = (nose[:, 1] - ear_y) / np.maximum(s, 1e-6)  # < 0: нос выше линии ушей (голова запрокинута), > 0: наклон вперёд
    sho_y = _mean2(lsho[:, 1], rsho[:, 1])
    wL, wR = pt(LWRI), pt(RWRI)
    hL, hR = hand_xy["L"], hand_xy["R"]
    wrist_h = np.where(hand == 0, (wL[:, 1] - sho_y) / s, np.where(hand == 1, (wR[:, 1] - sho_y) / s, np.nan))  # <0: выше плеча

    return pd.DataFrame({
        "frame": grid.frame.values, "t": t, "H": H, "s": s, "have_det": have,
        "mouth_x": mouth[:, 0], "mouth_y": mouth[:, 1], "nose_x": nose[:, 0], "nose_y": nose[:, 1],
        "d": d, "d_raw": d_raw_min, "dL": dL, "dR": dR, "hand": hand.astype(int), "v": v,
        "elbowL": ang["L"], "elbowR": ang["R"], "head_tilt": tilt, "wrist_h": wrist_h,
        "wrist_conf": np.fmax(cf[:, LWRI], cf[:, RWRI]), "nose_conf": cf[:, NOSE],
        "wristL_x": wL[:, 0], "wristL_y": wL[:, 1], "wristR_x": wR[:, 0], "wristR_y": wR[:, 1],
        "handL_x": hL[:, 0], "handL_y": hL[:, 1], "handR_x": hR[:, 0], "handR_y": hR[:, 1],   # = запястье при hand_extend=0
    })


def visibility_fraction(series: pd.DataFrame) -> float:
    """Доля кадров с валидным d (запястье и нос видны). < min_visible_frac => кандидат в IGNORE."""
    return float(np.isfinite(series["d"]).mean()) if len(series) else 0.0
