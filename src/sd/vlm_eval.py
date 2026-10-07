"""Проверка VLM-верификатора на размеченных циклах: кадры «трубки» человека вокруг цикла → модель → оценка «затяжка» и действие.

Считается то же, что для остальных признаков (`feature_auc`): AUC против меток (интервал — бутстрэп по видео), AUC против каждого вида ложного жеста,
доля верных решений при пороге 0.5 и стоимость (время на цикл, prefill/decode отдельно). Результаты дописываются в parquet по ключу цикла
(video, tid, start) после КАЖДОГО окна: прогон в десятки минут на CPU можно прервать и продолжить, уже посчитанные циклы не пересчитываются.
Метки — суждения ассистента (см. `feature_auc`), выборка мала и смешана со сценой: результат — «различает ли модель вообще», не итоговое качество.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import feature_auc as FA
from . import stages
from .calibrate import cfg_current_for_run
from .evidence import mouth_frames
from .tracks import Tracks
from .tube import tube_frames
from .vlm import smoke_score

KEY = ["video", "tid", "start"]


def pick_subset(tab: pd.DataFrame, per_class: int | None, seed: int = 0) -> pd.DataFrame:
    """Детерминированная выборка для быстрого скрининга: поровну затяжек и не-затяжек, по кругу по видео (чтобы не набрать всё из одного клипа)."""
    if not per_class:
        return tab
    rng = np.random.default_rng(seed)
    parts = []
    for y in (1, 0):
        g = tab[tab.y == y]
        by_v = {v: list(rng.permutation(gg.index.values)) for v, gg in g.groupby("video")}
        vids = list(by_v)
        rng.shuffle(vids)
        order: list[int] = []
        want = min(per_class, len(g))
        while len(order) < want:
            for v in vids:
                if by_v[v] and len(order) < want:
                    order.append(by_v[v].pop())
        parts.append(g.loc[order])
    return pd.concat(parts)


def run_vlm(tab: pd.DataFrame, vlms: dict, out: Path, frames: int = 6, window: float = 2.5, progress: Callable[[int, int], None] | None = None,
            crop: str = "tube", mouth_scale: float = 2.5, size: int = 224, ctx: dict | None = None) -> pd.DataFrame:
    """Прогоняет VLM по циклам таблицы `tab` (колонки video, tid, start, peak_t, run) и дописывает результат в `out` после каждого окна.

    `crop`: `tube` — верх тела человека целиком (`tube.tube_frames`), `mouth` — крупные кропы вокруг рта и активной кисти (`evidence.mouth_frames`, сторона `mouth_scale`
    ширин плеч). `vlms` — {имя режима: клиент}; режимы идут подряд на одних кадрах (кэш запроса между разными текстами не работает — каждый режим стоит полный prefill). Колонки результата:
    `score_<имя>`, `action_<имя>`, `ms_<имя>`, `prefill_ms_<имя>`, `cached_<имя>`, `tokens_<имя>`; `score`/`ms`/`action` — копии первого режима.
    """
    names = list(vlms)
    rows = pd.read_parquet(out).to_dict("records") if out.exists() else []
    done = {(r["video"], int(r["tid"]), float(r["start"])) for r in rows}
    cache: dict[str, tuple] = {}
    for i, r in enumerate(tab.itertuples()):
        key = (r.video, int(r.tid), float(r.start))
        if key in done:
            if progress:
                progress(i + 1, len(tab))
            continue
        if ctx and r.run in ctx:      # готовые (треки, конфиг, ряд признаков) вызывающего: пороги те же, что у циклов в таблице (recognize, UI)
            tr, cfg, ser = ctx[r.run]
        else:
            if r.run not in cache:
                rd = Path(r.run)
                tr0, cfg0 = Tracks.load(rd / "pose"), cfg_current_for_run(rd)
                cache = {r.run: (tr0, cfg0, stages.stage_features(tr0, cfg0, rd) if crop == "mouth" else None)}   # держим один запуск: память дороже
            tr, cfg, ser = cache[r.run]
        t0, t1 = float(r.peak_t) - window / 2, float(r.peak_t) + window / 2
        if crop == "mouth":
            fr, _ = mouth_frames(tr.meta["video"], tr, ser, int(r.tid), t0, t1, cfg, n=frames, size=size, scale=mouth_scale)
        else:
            fr, _, _ = tube_frames(tr.meta["video"], tr, int(r.tid), t0, t1, cfg, n=frames)
        row = dict(video=r.video, tid=int(r.tid), start=float(r.start), ok=bool(fr))
        if fr:
            for name in names:
                ans = vlms[name].verify(fr, t1 - t0)
                row.update({f"score_{name}": smoke_score(ans), f"action_{name}": ans["action"], f"ms_{name}": float(ans["ms"]), f"prefill_ms_{name}": ans.get("prefill_ms"),
                            f"decode_ms_{name}": ans.get("decode_ms"), f"cached_{name}": ans.get("cached_tokens"), f"tokens_{name}": ans.get("prompt_tokens"),
                            f"parsed_{name}": bool(ans.get("parsed")), f"raw_{name}": str(ans["raw"])[:120]})
            row.update(score=row[f"score_{names[0]}"], action=row[f"action_{names[0]}"], ms=row[f"ms_{names[0]}"], parsed=row[f"parsed_{names[0]}"])
        rows.append(row)
        pd.DataFrame(rows).to_parquet(out, index=False)
        if progress:
            progress(i + 1, len(tab))
    return pd.DataFrame(rows)


def modes_in(res: pd.DataFrame) -> list[str]:
    """Имена режимов, посчитанных в таблице результатов (по колонкам `score_<имя>`)."""
    return [c[len("score_"):] for c in res.columns if c.startswith("score_")]


def evaluate(tab: pd.DataFrame, res: pd.DataFrame, n_boot: int = 500, mode: str | None = None) -> dict:
    """Сводка качества режима `mode` (по умолчанию первый): AUC оценки с интервалом по видео, AUC против каждого ложного жеста, решение при 0.5, стоимость."""
    sfx = f"_{mode}" if mode else ""
    ok = res[res.ok.astype(bool)]
    m = tab.merge(ok, on=KEY, how="inner")
    out: dict = dict(n=len(m), n_pos=int(m.y.sum()), n_neg=int((1 - m.y).sum()), n_failed=int((~res.ok.astype(bool)).sum()))
    if m.empty or m.y.nunique() < 2:
        return out
    m = m.assign(_score=m[f"score{sfx}"].astype(float))
    a, lo, hi = FA.auc_by_video_bootstrap(m.dropna(subset=["_score"]), "_score", n_boot)
    out.update(auc=a, auc_lo=lo, auc_hi=hi, n_scored=int(m._score.notna().sum()))
    for nc in FA.NEG:
        s = m[m.label.isin(FA.POS + (nc,))].dropna(subset=["_score"])
        if (s.label == nc).sum() >= 3 and s.y.nunique() == 2:
            out[f"auc_vs_{nc}"] = FA._auc(s.y.values, s._score.values)
    pred = (m._score >= 0.5)
    out.update(recall=float(pred[m.y == 1].mean()), fp_rate=float(pred[m.y == 0].mean()), accuracy=float((pred == (m.y == 1)).mean()), answer_yes_share=float(pred.mean()))
    # лучший порог по Юдену (на этих же данных — только чтобы увидеть, есть ли вообще порог с пользой; в manifest не идёт)
    ths = np.unique(m._score.dropna())
    if len(ths) > 1:
        best = max(((float(((m._score >= t) & (m.y == 1)).sum() / max((m.y == 1).sum(), 1) - ((m._score >= t) & (m.y == 0)).sum() / max((m.y == 0).sum(), 1)), float(t))
                    for t in ths), key=lambda x: x[0])
        out.update(youden_j=best[0], youden_threshold=best[1])
    ms = m[f"ms{sfx}"] if f"ms{sfx}" in m else m["ms"]
    out.update(ms_median=float(ms.median()), ms_mean=float(ms.mean()), ms_max=float(ms.max()))
    for c in ("prefill_ms", "decode_ms", "tokens", "cached"):
        col = f"{c}{sfx}"
        if col in m and m[col].notna().any():
            out[f"{c}_median"] = float(m[col].median())
    pc = f"parsed{sfx}"
    out["parsed_share"] = float(m[pc].astype(bool).mean()) if pc in m else float("nan")
    ac = f"action{sfx}"
    if ac in m:
        out["actions"] = m[ac].value_counts().to_dict()
    return out
