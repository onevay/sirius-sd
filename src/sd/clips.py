"""Прогон конвейера по папке клипов вида «папка = класс»: ваши `data/` и скачанные открытые датасеты в `data_external/`.

Для каждого клипа: поза+трекинг → признаки → циклы → события (кэш этапов тот же, что у `sd pose/cycles/events`), затем сводка по КЛАССАМ:
какая доля клипов даёт хотя бы один цикл/событие и сколько циклов в минуту на человека. Это быстрая проверка «конвейер работает и различает классы»
на любом наборе без ручной разметки (метка клипа — имя папки). Циклы клипов складываются в таблицу для обучения (группа = клип).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import stages
from .dataset import build_cycle_table
from .paths import OUTPUTS, class_label, list_videos, video_id
from .video_io import probe


def pick_clips(root: Path, per_class: int | None = None, only_class: set[str] | None = None, seed: int = 0) -> list[Path]:
    """Клипы под `root`, сгруппированные по папке-классу; не больше `per_class` на класс (детерминированная случайная выборка)."""
    by: dict[str, list[Path]] = {}
    for v in list_videos(root):
        by.setdefault(v.parent.name, []).append(v)
    rng = np.random.default_rng(seed)
    out: list[Path] = []
    for cls, vs in sorted(by.items()):
        if only_class and cls not in only_class:
            continue
        vs = sorted(vs)
        if per_class and len(vs) > per_class:
            vs = [vs[i] for i in sorted(rng.choice(len(vs), per_class, replace=False))]
        out += vs
    return out


def run_clips(root: Path, cfg: dict, per_class: int | None = None, max_sec: float = 15.0, start: float = 0.0, only_class: set[str] | None = None,
              positive: set[str] | None = None, tag: str | None = None, force: bool = False,
              progress: Callable[[int, int, str], None] | None = None, bundle=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (таблица клипов, таблица циклов) и пишет их в outputs/external/<tag>/. Сломанный клип не останавливает прогон.

    `bundle` — загруженный `bundle.Bundle` классификатора цикла: тогда события собираются по его оценкам (признаки «быстрого» набора: кинематика, ритм, поза), а в таблице клипов
    появляются `events` (по классификатору) и `events_rules` (по правилу длительности паузы) — разница показывает, что даёт классификатор на этом наборе."""
    from .pipeline import fast_cycle_features
    from .pose_feats import pose_rows

    clips = pick_clips(root, per_class, only_class)
    min_h = cfg["video"]["min_person_height_px"]
    rows, tables = [], []
    for i, vp in enumerate(clips):
        t0 = time.perf_counter()
        row = dict(video=video_id(vp), cls=vp.parent.name, label=class_label(vp.parent.name, positive))
        try:
            dur = probe(vp).duration
            end = None if dur <= start + max_sec + 0.5 else round(start + max_sec, 2)   # короткий клип целиком: тот же ключ кэша, что `<начало>-ends`
            tr, rd = stages.stage_pose(vp, cfg, start, end, force=force)
            ser = stages.stage_features(tr, cfg, rd)
            cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
            df, _ = stages.stage_events(tr, ser, cyc, cfg, rd, "cam_local", vp.stem)
            s = tr.summary(min_h)
            ok = s[~s.ignore_small]
            main = ser[ser.tid == ser.groupby("tid").size().idxmax()] if len(ser) else ser    # самый длинный трек клипа
            reach = dict(d_min_track=float(main.d.min()) if len(main) else np.nan, wrist_conf=float(main.wrist_conf.mean()) if len(main) else np.nan)
            cyc_ok = cyc[cyc.tid.isin(ok.tid)]
            tab = build_cycle_table(vp, tr, ser, cyc)
            n_rules = len(df)
            if bundle is not None and len(tab):
                feats = fast_cycle_features(tab, pd.DataFrame(pose_rows(video_id(vp), tr, cfg, ser, cyc)))
                sc = bundle.score(feats)
                scores = {(int(t), round(float(st), 3)): float(v) for t, st, v in zip(feats.tid, feats.start, sc)}
                df, _ = stages.assemble_events(tr, ser, cyc, cfg, "cam_local", vp.stem, cycle_scores=scores)
                tab = tab.assign(score=sc)
            row.update(video_sec=round(dur, 1), window_sec=round(min(max_sec, dur - start), 1), tracks=len(s), tracks_ge80=len(ok), h_med=float(ok.h_med.median()) if len(ok) else np.nan,
                       person_min=float(ok.dur.sum() / 60.0), cycles=len(cyc_ok), events=len(df), events_rules=n_rules,
                       ev_conf_max=float(df.confidence.max()) if len(df) else np.nan, wall_s=round(time.perf_counter() - t0, 1), **reach)
            if len(tab):
                tab = tab[tab.tid.isin(ok.tid)]
                tables.append(tab.assign(source=tag or root.name, cls=vp.parent.name))
        except Exception as e:   # битый файл, нет кадров, нет людей
            row["error"] = f"{type(e).__name__}: {str(e)[:100]}"
        rows.append(row)
        if progress:
            progress(i + 1, len(clips), row["video"])
    df = pd.DataFrame(rows)
    cyc_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    out = OUTPUTS / "external" / (tag or root.name)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "clips.csv", index=False, encoding="utf-8-sig")
    if len(cyc_df):
        cyc_df.to_parquet(out / "cycles.parquet", index=False)
    return df, cyc_df


def summarize(df: pd.DataFrame, th_in: float = 0.65) -> pd.DataFrame:
    """По классам: клипов, ошибок, доля с циклом, доля с событием, циклов в минуту на человека (медиана и по сумме), медианный рост человека;
    если в таблице есть диагностика позы — доля клипов, где запястье подходит ко рту ближе `th_in` (порог входа автомата; иначе цикла быть не может), и медианная достоверность запястий."""
    d = df.copy()
    if "error" not in d:
        d["error"] = np.nan
    ok = d[d.error.isna() & d["cycles"].notna()]
    agg = dict(clips=("video", "size"), cycles_sum=("cycles", "sum"), person_min=("person_min", "sum"),
               with_cycle=("cycles", lambda x: float((x > 0).mean())), with_event=("events", lambda x: float((x > 0).mean())), h_med_px=("h_med", "median"))
    if "events_rules" in ok:   # прогон с классификатором: доля клипов с событием по правилу длительности паузы — для сравнения
        agg["with_event_rules"] = ("events_rules", lambda x: float((x > 0).mean()))
    if "d_min_track" in ok:   # диагностика позы: почему нет циклов — рука не достаёт до рта в оценке позы или автомат отверг жест
        agg["reach_th_in"] = ("d_min_track", lambda x: float((x < th_in).mean()))
        agg["wrist_conf_med"] = ("wrist_conf", "median")
    out = ok.groupby(["cls", "label"]).agg(**agg)
    out["cycles_per_person_min"] = out.cycles_sum / out.person_min.clip(lower=1e-9)
    out = out.drop(columns=["cycles_sum", "person_min"])
    out["errors"] = d.groupby(["cls", "label"]).error.apply(lambda x: int(x.notna().sum()))
    return out.reset_index()


def class_auc(clips: pd.DataFrame, cycles: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Качество оценки цикла на наборе «папка = класс» (метка клипа = класс, слабая: в положительном клипе не каждый цикл — затяжка).

    * уровень клипа: оценка клипа = максимальная оценка цикла в клипе (клип без циклов = 0, как «автомат ничего не нашёл»); AUC «курение против остальных классов» с 95% интервалом
      (бутстрэп по клипам) и против каждого отрицательного класса отдельно; доля клипов, где оценка ≥ 0.5;
    * уровень цикла: AUC оценки цикла против метки клипа (вторичный: метки шумные), число циклов по классам.
    Нужна колонка `score` в таблице циклов (прогон с `--cycle-bundle`)."""
    from sklearn.metrics import roc_auc_score

    ok = clips[clips["error"].isna()] if "error" in clips else clips
    sc = cycles.groupby("video").score.max() if len(cycles) and "score" in cycles else pd.Series(dtype=float)
    d = ok[["video", "cls", "label"]].copy()
    d["score"] = d.video.map(sc).fillna(0.0)
    d["y"] = (d.label == "smoking").astype(int)
    out: dict = dict(clips=int(len(d)), positives=int(d.y.sum()), clip_auc=float("nan"), lo=float("nan"), hi=float("nan"), per_class=[], cycle_auc=float("nan"), cycles=int(len(cycles)))
    if d.y.nunique() < 2:
        return out
    out["clip_auc"] = float(roc_auc_score(d.y, d.score))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        if b.y.nunique() == 2:
            bs.append(roc_auc_score(b.y, b.score))
    if bs:
        out["lo"], out["hi"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
    pos = d[d.y == 1]
    for c, g in d[d.y == 0].groupby("cls"):
        out["per_class"].append(dict(cls=c, n=int(len(g)), auc=float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(g))], np.r_[pos.score, g.score])),
                                     score_median=float(g.score.median()), share_ge_05=float((g.score >= 0.5).mean())))
    out["positive_score_median"] = float(pos.score.median())
    out["positive_share_ge_05"] = float((pos.score >= 0.5).mean())
    if len(cycles) and "score" in cycles:
        lab = cycles.video.map(d.set_index("video").y)
        m = lab.notna()
        if m.sum() and lab[m].nunique() == 2:
            out["cycle_auc"] = float(roc_auc_score(lab[m].astype(int), cycles.score[m]))
        out["cycles_by_class"] = {str(k): int(v) for k, v in cycles.groupby("cls").size().items()}
    return out

