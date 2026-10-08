"""Эталон событий из размеченных кандидатов-жестов: затяжки одного человека → эпизоды курения (POSITIVE), одиночные затяжки → IGNORE.

Метки жестов ставятся глазами по монтажам (`sd start-eval --montages`, формат `docs/review/gesture_gt.csv`). Эпизод — затяжки одного трека, расстояние между соседними
не больше `merge_gap` (как склейка событий в регламенте, 15 с); эпизод из ≥ `min_puffs` затяжек — POSITIVE, одиночная затяжка — IGNORE (из одного клипа нельзя понять,
эпизод это или случайный жест: событие по регламенту такое не требует, и штрафовать за него нельзя). Все строки получают `labeler` — по нему видно, что это разметка
ассистента, а не организаторов или ваша; править их можно на странице «Разметка».
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import gt as GT


def puff_episodes(puffs: pd.DataFrame, merge_gap: float = 15.0, min_puffs: int = 2, pad: float = 0.5) -> pd.DataFrame:
    """puffs: video, tid, start, end, peak_t → эпизоды: video, tid, start, end, peak, n, label (POSITIVE | IGNORE)."""
    rows = []
    for (v, t), g in puffs.sort_values("start").groupby(["video", "tid"]):
        cur = None
        for r in g.itertuples():
            if cur is not None and r.start - cur["end"] <= merge_gap:
                cur["end"], cur["n"] = max(cur["end"], r.end), cur["n"] + 1
                cur["peaks"].append(r.peak_t)
            else:
                if cur is not None:
                    rows.append(cur)
                cur = dict(video=v, tid=int(t), start=r.start, end=r.end, n=1, peaks=[r.peak_t])
        if cur is not None:
            rows.append(cur)
    out = pd.DataFrame(rows, columns=["video", "tid", "start", "end", "n", "peaks"])
    if out.empty:
        return out.assign(peak=[], label=[]).drop(columns="peaks")
    out["start"] = (out.start - pad).clip(lower=0.0)
    out["end"] = out.end + pad
    out["peak"] = out.peaks.map(lambda p: sorted(p)[len(p) // 2])
    out["label"] = ["POSITIVE" if n >= min_puffs else "IGNORE" for n in out.n]
    return out.drop(columns="peaks").reset_index(drop=True)


def write_gt(episodes: pd.DataFrame, durations: dict[str, float], box_at: Callable[[str, int, float], tuple | None], labeler: str = "assistant",
             reviewed_clips: list[str] | None = None, path=None) -> dict:
    """Записывает эпизоды в эталон (`gt.add`), рамка человека — из трека в пике. Клипы из `reviewed_clips` без POSITIVE получают «весь клип без курения».
    Прежние строки того же `labeler` по этим клипам заменяются; чужие (ваши) не трогаются."""
    clips = set(reviewed_clips or []) | set(episodes.video)
    df = GT.load(path)
    drop = df[(df.labeler == labeler) & df.clip_id.isin(clips)].id.tolist()
    if drop:
        GT.delete(drop, path)
    n_pos = n_ign = n_neg = 0
    for r in episodes.itertuples():
        box = box_at(r.video, int(r.tid), float(r.peak))
        if r.label == "POSITIVE" and box is None:
            raise ValueError(f"нет рамки человека для {r.video} трек {r.tid} в {r.peak:.1f} с")
        GT.add(r.video, r.start, min(r.end, durations.get(r.video, r.end)), r.label, person=f"t{int(r.tid)}", peak=r.peak, box=box if r.label == "POSITIVE" else None,
               note=f"{int(r.n)} затяжк." + ("и" if r.n > 1 else "а") + " (по меткам жестов)", labeler=labeler, path=path)
        n_pos += r.label == "POSITIVE"
        n_ign += r.label == "IGNORE"
    have_pos = set(episodes[episodes.label == "POSITIVE"].video)
    for c in sorted(clips - have_pos):
        if c in durations and not (set(episodes[episodes.video == c].label) & {"IGNORE"}) :
            GT.mark_clean(c, durations[c], labeler=labeler, path=path)
            n_neg += 1
    return dict(positive=int(n_pos), ignore=int(n_ign), clean_clips=int(n_neg))


def rebuild_gt(pool: pd.DataFrame, durations: dict[str, float], box_at: Callable[[str, int, float], tuple | None], *, labeler: str = "assistant", path=None,
               negative_prefixes: tuple[str, ...] = ("лжекурение__",), positive_prefixes: tuple[str, ...] = ("курение__",)) -> dict:
    """Эталон событий целиком из размеченного пула кандидатов (`pool`: video, tid, start, end, peak_t, label).

    * затяжки (`smoke`) одного человека → эпизоды (POSITIVE при ≥ 2 затяжек, иначе IGNORE);
    * «неясные» жесты вне POSITIVE-эпизодов → IGNORE (событие там не штрафуется);
    * клипы из папок `negative_prefixes` без эпизодов → «весь клип без курения»;
    * клипы из папок `positive_prefixes` без подтверждённого POSITIVE → IGNORE на весь клип (папка говорит, что курение есть, но я его не подтвердил: клип вне оценки, а не негатив).
    Строки этого разметчика по перечисленным клипам заменяются, чужие не трогаются — повторный вызов безопасен."""
    smoke = pool[pool.label == "smoke"]
    ep = puff_episodes(smoke[["video", "tid", "start", "end", "peak_t"]]) if len(smoke) else puff_episodes(pd.DataFrame(columns=["video", "tid", "start", "end", "peak_t"]))
    pos = set(ep[ep.label == "POSITIVE"].video)
    clips = set(durations)
    neg = sorted(c for c in clips if c.startswith(negative_prefixes) and c not in pos)
    neutral = sorted(c for c in clips if c.startswith(positive_prefixes) and c not in pos)
    df = GT.load(path)
    drop = df[(df.labeler == labeler) & df.clip_id.isin(clips)].id.tolist()
    if drop:
        GT.delete(drop, path)
    ep_use = ep[ep.video.isin(clips) & ~ep.video.isin(neutral)]
    r = write_gt(ep_use, durations, box_at, labeler=labeler, reviewed_clips=[], path=path)
    n_unsure = 0
    for u in pool[(pool.label == "unsure") & pool.video.isin(clips) & ~pool.video.isin(neutral)].itertuples():
        inside = ep_use[(ep_use.video == u.video) & (ep_use.tid == u.tid) & (ep_use.start <= u.peak_t) & (ep_use.end >= u.peak_t) & (ep_use.label == "POSITIVE")]
        if len(inside):
            continue
        GT.add(u.video, max(u.start - 0.3, 0.0), min(u.end + 0.3, durations[u.video]), "IGNORE", person=f"t{int(u.tid)}", note="неясный жест (по меткам жестов)", labeler=labeler, path=path)
        n_unsure += 1
    for c in neg:
        GT.mark_clean(c, durations[c], labeler=labeler, path=path)
    for c in neutral:
        GT.add(c, 0.0, durations[c], "IGNORE", note="клип из папки «курение»: затяжки не подтверждены — вне оценки", labeler=labeler, path=path)
    return dict(positive=r["positive"], ignore_episodes=r["ignore"], ignore_unsure=n_unsure, clean_clips=len(neg), neutral_clips=len(neutral))
