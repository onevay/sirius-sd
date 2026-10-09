"""Фото-классификатор на видео: кропы рта и активной кисти вокруг каждого цикла → эмбеддинги → оценка «курит» по кадрам цикла.

Это проверка переноса домена: модель учится на портретах/сценах из фото-наборов, а применяется к кропам 224 px из записей камер.
Кропы (6 кадров на цикл, те же, что видит VLM) сохраняются как JPEG в outputs/photo/crops/<ключ цикла>/ — повторный прогон их не пересчитывает,
а эмбеддинги берутся из общего кэша `photo_feats`.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import photo_clf as PC
from . import photo_feats as PF
from .paths import OUTPUTS

CROPS = OUTPUTS / "photo" / "crops"
KEY = ["video", "tid", "start"]


def cycle_dir(video: str, tid: int, start: float) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(video))
    return CROPS / f"{safe}__{int(tid)}__{float(start):.3f}"


def extract_crops(tab: pd.DataFrame, frames: int = 6, window: float = 2.5, size: int = 224, scale: float = 2.5, progress=None, ctx: dict | None = None) -> pd.DataFrame:
    """Сохраняет кропы рта для каждого цикла таблицы `tab` (video, tid, start, peak_t, run). Возвращает таблицу (ключ, число кадров, каталог).

    `ctx` — {run: (треки, конфиг, ряд признаков)}, если вызывающий уже держит их в памяти (тот же конфиг, что у циклов)."""
    from . import stages
    from .calibrate import cfg_current_for_run
    from .evidence import mouth_frames
    from .tracks import Tracks

    rows, cache = [], {}
    for i, r in enumerate(tab.sort_values(["run", "start"]).itertuples()):
        d = cycle_dir(r.video, r.tid, r.start)
        have = sorted(d.glob("*.jpg")) if d.exists() else []
        if len(have) >= frames:
            rows.append(dict(video=r.video, tid=int(r.tid), start=float(r.start), n=len(have), dir=str(d)))
        else:
            if ctx and r.run in ctx:
                tr, cfg, ser = ctx[r.run]
            else:
                if r.run not in cache:
                    rd = Path(r.run)
                    tr0, cfg0 = Tracks.load(rd / "pose"), cfg_current_for_run(rd)
                    cache = {r.run: (tr0, cfg0, stages.stage_features(tr0, cfg0, rd))}   # держим один запуск: память дороже
                tr, cfg, ser = cache[r.run]
            t0, t1 = float(r.peak_t) - window / 2, float(r.peak_t) + window / 2
            fr, _ = mouth_frames(tr.meta["video"], tr, ser, int(r.tid), t0, t1, cfg, n=frames, size=size, scale=scale)
            d.mkdir(parents=True, exist_ok=True)
            for j, f in enumerate(fr):
                cv2.imwrite(str(d / f"{j}.jpg"), cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(dict(video=r.video, tid=int(r.tid), start=float(r.start), n=len(fr), dir=str(d)))
        if progress:
            progress(i + 1, len(tab))
    return pd.DataFrame(rows)


def frame_table(crops: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на кадр кропа: ключ цикла + путь к файлу."""
    out = []
    for r in crops.itertuples():
        for p in sorted(Path(r.dir).glob("*.jpg")):
            out.append(dict(video=r.video, tid=r.tid, start=r.start, path=str(p)))
    return pd.DataFrame(out, columns=[*KEY, "path"])          # колонки есть и у пустой таблицы: циклы без кропов (рот не виден) не должны ронять разбор


def embed_crops(ft: pd.DataFrame, backbones=("clip", "convnext"), backend: str = "auto", progress=None) -> dict[str, np.ndarray]:
    out = {}
    for bb in backbones:
        out[bb] = PF.embed_files(PF.make_embedder(bb, backend), ft.path.tolist(), progress=(lambda i, n, dt, bb=bb: progress(bb, i, n, dt)) if progress else None)
    return out


def aggregate(ft: pd.DataFrame, per_frame: dict[str, np.ndarray], prefix: str = "photo") -> pd.DataFrame:
    """Кадровые оценки → признаки цикла: среднее, максимум, среднее по двум лучшим кадрам, число кадров."""
    if ft.empty:
        return pd.DataFrame(columns=KEY)
    d = ft[KEY].copy()
    for name, s in per_frame.items():
        d[name] = s
    rows = []
    for k, g in d.groupby(KEY):
        row = dict(zip(KEY, k))
        for name in per_frame:
            v = np.sort(g[name].dropna().to_numpy())[::-1]
            if len(v):
                row.update({f"{prefix}_{name}_mean": float(v.mean()), f"{prefix}_{name}_max": float(v[0]), f"{prefix}_{name}_top2": float(v[:2].mean())})
        row[f"{prefix}_n_frames"] = int(g[list(per_frame)[0]].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def score_frames(bundle: dict, emb: dict[str, np.ndarray]) -> np.ndarray:
    """Оценка пакета фото-модели по кадрам; NaN, если у кадра нет эмбеддинга."""
    bbs = bundle["manifest"]["backbones"]
    ok = np.ones(len(next(iter(emb.values()))), bool)
    for bb in bbs:
        ok &= ~np.isnan(emb[bb]).any(1)
    s = np.full(len(ok), np.nan)
    if ok.any():
        s[ok] = PC.score_bundle(bundle, {bb: emb[bb][ok] for bb in bbs})
    return s
