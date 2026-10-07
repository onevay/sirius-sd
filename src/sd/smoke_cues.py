"""«Дым» как признак цикла (в запросе: расположения точек, время, угол локтя, ответы VLM, дым): проверка простыми и дешёвыми признаками на кропах рта.

Кадры кропа (6 на цикл, ≈ 0.5 с друг от друга вокруг пика, `photo_video`) упорядочены по времени: 0–1 — до затяжки, 3–5 — после (выдох). Дым появляется ПОСЛЕ затяжки,
поэтому главные признаки — разность «поздние минус ранние кадры» внутри цикла: сцена и человек остаются теми же, разность убирает их влияние.
  * классическая «дымка»: падение резкости (дисперсия лапласиана), низкочастотного контраста, плотности границ и насыщенности, рост доли «молочных» пикселей
    (низкая насыщенность, высокая яркость); знак `d_*` выбран так, чтобы ожидаемое направление «есть дым» = больше;
  * CLIP zero-shot: сходство кадра с «cigarette smoke» минус сходство с «no smoke»: максимум по кадрам, среднее и разность «поздний максимум − ранний средний».
Направления и агрегаты заданы ДО просмотра результата; оцениваются все признаки сразу (без выбора лучшего). Результат на кропах ваших видео — `ANALYSIS.md`, раздел 9.4.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

KEY = ["video", "tid", "start"]
EARLY, LATE = slice(0, 2), slice(3, 6)
SMOKE_PROMPTS = {"s0": "a photo of a person exhaling cigarette smoke", "s1": "a photo with a cloud of smoke in front of a person's face", "s2": "a photo of visible cigarette smoke in the air"}
NONE_PROMPTS = {"n0": "a photo of a person with no smoke in the air", "n1": "a photo of clear air with no smoke"}
HAZE_COLS = ["smoke_d_sharp", "smoke_d_low", "smoke_d_edge", "smoke_d_sat", "smoke_d_white"]
CLIP_COLS = ["smoke_clip_max", "smoke_clip_mean", "smoke_clip_delta"]
ALL_COLS = HAZE_COLS + CLIP_COLS


def frame_measures(rgb: np.ndarray) -> dict[str, float]:
    """Контраст/«молочность» кадра: резкость (дисперсия лапласиана), низкочастотный контраст, плотность границ, насыщенность, доля светлых малонасыщенных пикселей."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return dict(sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var()), low=float(cv2.GaussianBlur(gray, (0, 0), 8).std()), edge=float((cv2.Canny(gray, 60, 140) > 0).mean()),
                sat=float(hsv[..., 1].mean() / 255.0), white=float(((hsv[..., 1] < 50) & (hsv[..., 2] > 140)).mean()))


def haze_deltas(frames: list[np.ndarray]) -> dict[str, float]:
    """Относительные изменения «поздние кадры − ранние» по кадрам цикла (6 штук); NaN, если кадров меньше шести."""
    if len(frames) < 6:
        return {c: float("nan") for c in HAZE_COLS}
    m = pd.DataFrame([frame_measures(f) for f in frames[:6]])
    e, l = m.iloc[EARLY].mean(), m.iloc[LATE].mean()
    drop = lambda k: float(-(l[k] - e[k]) / (e[k] + 1e-6))   # noqa: E731 — падение величины: больше = «дымнее»
    return dict(smoke_d_sharp=drop("sharp"), smoke_d_low=drop("low"), smoke_d_edge=drop("edge"), smoke_d_sat=drop("sat"), smoke_d_white=float(l["white"] - e["white"]))


def clip_cues(emb: np.ndarray, text_smoke: np.ndarray, text_none: np.ndarray) -> dict[str, float]:
    """CLIP zero-shot по кадрам цикла: эмбеддинги кадров (n×512, L2-нормированные) и текстов → max, среднее и «поздний максимум − ранний средний»."""
    if len(emb) < 6:
        return {c: float("nan") for c in CLIP_COLS}
    sc = (emb @ text_smoke.T).mean(1) - (emb @ text_none.T).mean(1)
    return dict(smoke_clip_max=float(sc.max()), smoke_clip_mean=float(sc.mean()), smoke_clip_delta=float(sc[LATE].max() - sc[EARLY].mean()))


def cycle_cues(cycles: pd.DataFrame, with_clip: bool = True, progress=None) -> pd.DataFrame:
    """Признаки «дыма» для циклов (`video, tid, start`), у которых есть кропы рта (`sd analyze-cycles --what photo` или `photo_video.extract_crops`); без кропа — строки нет."""
    from . import photo_feats as PF
    from . import photo_video as PV

    ts = tn = cache = None
    if with_clip:
        text = PF.clip_text_embeddings({**SMOKE_PROMPTS, **NONE_PROMPTS}, cache_name="clip_text_smoke")
        ts, tn = text[:len(SMOKE_PROMPTS)], text[len(SMOKE_PROMPTS):]
        cache = PF.load_cache("clip")
    rows = []
    for i, r in enumerate(cycles[KEY].itertuples(index=False)):
        d = PV.cycle_dir(r.video, r.tid, r.start)
        files = sorted(d.glob("*.jpg")) if d.exists() else []
        if len(files) < 6:
            continue
        frames = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files[:6]]
        row = dict(video=r.video, tid=int(r.tid), start=round(float(r.start), 3), **haze_deltas(frames))
        if with_clip:
            vec = [cache.get(PF._key(str(f))) for f in files[:6]]
            row.update(clip_cues(np.stack(vec), ts, tn) if all(v is not None for v in vec) else {c: float("nan") for c in CLIP_COLS})
        rows.append(row)
        if progress:
            progress(i + 1, len(cycles))
    return pd.DataFrame(rows)


def evaluate(tab: pd.DataFrame, cols: list[str] = ALL_COLS, n_boot: int = 400) -> pd.DataFrame:
    """AUC каждого признака «курение против остальных» (`y`, интервал бутстрэпом по видео), AUC внутри папки «курение» и «папка среди негативов» — две проверки на смешение со сценой."""
    from .model_tools import auc_with_ci

    folder = tab.video.str.split("__").str[0]
    rows = []
    for c in [c for c in cols if c in tab.columns]:
        a, lo, hi = auc_with_ci(tab.y.to_numpy(), tab[c].to_numpy(), tab.video.to_numpy(), n_boot=n_boot)
        inside = tab[folder == "курение"]
        a_in = auc_with_ci(inside.y.to_numpy(), inside[c].to_numpy(), inside.video.to_numpy(), n_boot=n_boot // 2)[0] if inside.y.nunique() == 2 else float("nan")
        neg = tab[tab.y == 0]
        fl = (neg.video.str.split("__").str[0] == "курение").astype(int)
        a_f = auc_with_ci(fl.to_numpy(), neg[c].to_numpy(), neg.video.to_numpy(), n_boot=n_boot // 2)[0] if fl.nunique() == 2 else float("nan")
        rows.append(dict(признак=c, AUC=a, lo=lo, hi=hi, внутри_папки_курение=a_in, папка_среди_негативов=a_f, n=int(tab[c].notna().sum())))
    return pd.DataFrame(rows)
