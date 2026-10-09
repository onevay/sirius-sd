"""Честная оценка на событиях: оценка цикла — out-of-fold (классификатор обучается на ДРУГИХ видео), события собираются по регламенту, метрики — протокол организаторов.

Зачем отдельно от `sd eval`: пакет `cycle_fast`/`cycle_cheap` обучен на размеченных кандидатах тех же видео, поэтому прогон `sd eval` с ним — «на обучающих». Здесь для каждого видео
оценку даёт модель, не видевшая это видео (фолды по видео, повторные перемешивания), а ТОЧКИ подбора — только порог оценки цикла `events.cycle_th` и порог события — помечены
как оптимистичные (подбираются на тех же 31 клипах): цифра показывает потолок на этих данных, не ожидание на скрытом наборе.

Вход — готовые запуски `sd recognize` (каталог `outputs/recognize/<клип>/0-ends_*/analysis`: признаки циклов) и метки жестов (`docs/review/gesture_gt.csv`/пул). Поза не пересчитывается.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import bundle as B
from . import cycle_models as CM
from . import evaluation as EV
from . import gt as GT
from . import stages
from .calibrate import cfg_current_for_run
from .paths import OUTPUTS, from_portable
from .tracks import Tracks

POS_LABELS = ("smoke",)
NEG_LABELS = ("no_gesture", "touch_face", "drink", "phone", "eat", "other_neg")
KEY = ["video", "tid", "start"]


@dataclass
class ClipData:
    video: str
    run_dir: Path
    out_dir: Path
    tab: pd.DataFrame            # признаки циклов (video, tid, start, peak_t, …) — то же, что получает пакет в `sd recognize`


SCENE_RULES = (("noyabrsk", ("smoking_noyabrsk",)), ("jar", ("smoking_Jar", "drinking_Jar")), ("woman_1106", ("2025_11_06_",)), ("drink_ru", ("распитие",)))


def scene_group(video: str) -> str:
    """Группа «сцена» для фолдов: клипы одной камеры/одного человека не должны оказываться и в обучении, и в проверке (иначе модель узнаёт место, а не жест).
    Правила по именам клипов (`SCENE_RULES`); остальные клипы — каждый сам по себе."""
    for name, keys in SCENE_RULES:
        if any(k in video for k in keys):
            return name
    return video


def collect(root: Path = OUTPUTS / "recognize", videos: Sequence[str] | None = None, window: str = "0-end", runs_like: str | None = None) -> dict[str, ClipData]:
    """Последний полный запуск `sd recognize` по каждому клипу: признаки циклов из его `analysis/`. `runs_like` — только запуски, в имени каталога позы которых есть эта подстрока
    (например `_10fps` или `5fps`): так классификатор обучается и оценивается на признаках ОДНОЙ конфигурации, а не на «последнем, что считалось»."""
    from . import feature_auc as FA
    from .pipeline import _with_end

    out = {}
    for vd in sorted(Path(root).iterdir()):
        if not vd.is_dir() or (videos is not None and vd.name not in videos):
            continue
        runs = sorted(d for d in vd.glob(f"{window}*") if (d / "result.json").exists())
        if runs_like:
            runs = [d for d in runs if runs_like in str(json.loads((d / "result.json").read_text(encoding="utf-8")).get("run_dir", ""))]
        if not runs:
            continue
        d = runs[-1]
        meta = json.loads((d / "result.json").read_text(encoding="utf-8"))
        if (d / "analysis" / "dataset_cycles.parquet").exists():
            tab = _with_end(FA.all_cycles_features(an=d / "analysis", vlm=None, dataset=d / "analysis" / "dataset_cycles.parquet"))
            tab = tab.assign(start=tab.start.round(3))
        else:   # клип без циклов: остаётся в оценке (событий нет), признаков нет
            tab = pd.DataFrame({"video": pd.Series(dtype=object), "tid": pd.Series(dtype="int64"), "start": pd.Series(dtype=float), "peak_t": pd.Series(dtype=float)})
        out[vd.name] = ClipData(vd.name, from_portable(meta["run_dir"]), d, tab)
    return out


def match_gestures(cands: pd.DataFrame, gestures: pd.DataFrame, tol: float = 1.0) -> pd.DataFrame:
    """Метка жеста для каждой строки `cands` (video, tid, peak_t): ближайший по peak_t жест того же видео в пределах `tol` с. Номер трека НЕ обязателен — он меняется между запусками
    (окно 0–40 с и полное видео дают разные ID одного человека); при двух жестах в окне предпочитается тот же трек, затем ближайший по времени. Колонка `label`, `start_quality`."""
    lab, sq = [], []
    by = {v: g for v, g in gestures.groupby("video")}
    for r in cands.itertuples():
        g = by.get(r.video)
        m = g[(g.peak_t - r.peak_t).abs() <= tol] if g is not None else pd.DataFrame()
        if len(m):
            same = m[m.tid == r.tid]
            m = same if len(same) else m
            b = m.iloc[int((m.peak_t - r.peak_t).abs().to_numpy().argmin())]
            lab.append(b.label)
            sq.append(b.get("start_quality"))
        else:
            lab.append(None)
            sq.append(None)
    return cands.assign(label=lab, start_quality=sq)


def attach_labels(clips: dict[str, ClipData], gestures: pd.DataFrame, tol: float = 1.0) -> pd.DataFrame:
    """Все циклы всех клипов + `label` (см. `match_gestures`) и `y` (1 — затяжка, 0 — другой жест/не жест, NaN — неясно или не размечено)."""
    tab = pd.concat([c.tab for c in clips.values() if len(c.tab)], ignore_index=True)
    tab = match_gestures(tab, gestures, tol).drop(columns="start_quality")
    tab["y"] = np.where(tab.label.isin(POS_LABELS), 1.0, np.where(tab.label.isin(NEG_LABELS), 0.0, np.nan))
    return tab


def oof_scores(tab: pd.DataFrame, set_name: str = "fast", n_splits: int = 5, repeats: int = 3, seed: int = 0, kinds=("lr", "gb", "nn"), cols: Sequence[str] | None = None, by_scene: bool = False) -> np.ndarray:
    """Оценка КАЖДОГО цикла моделью, не видевшей его видео (`by_scene` — и не видевшей его сцену: клипы одной камеры/человека в одном фолде): фолды по видео, обучение только на размеченных (y не NaN) циклах обучающих видео, усреднение по перемешиваниям и членам ансамбля.
    `cols` — явный список признаков вместо набора `set_name` (для коротких моделей: на сотне циклов длинный список переобучается)."""
    feats = CM.available(list(cols) if cols else CM.SETS[set_name], tab)
    spec = CM.members_for(feats, kinds)
    X = tab.reset_index(drop=True)
    y = X.y.to_numpy(float)
    groups = X.video.map(scene_group).to_numpy() if by_scene else X.video.to_numpy()
    uniq = np.array(sorted(set(groups)))
    score, cnt = np.zeros(len(X)), np.zeros(len(X))
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        perm = {v: i for i, v in enumerate(rng.permutation(uniq))}
        fold_of = np.array([perm[v] % n_splits for v in groups])           # случайное, но воспроизводимое разбиение видео на фолды
        for k in range(n_splits):
            te = np.flatnonzero(fold_of == k)
            tr = np.flatnonzero((fold_of != k) & np.isfinite(y))
            if len(te) == 0 or len(np.unique(y[tr])) < 2:
                continue
            ens = np.zeros(len(te))
            for s in spec.values():
                st = B.fit_member(s["kind"], X.iloc[tr], y[tr].astype(int), s["features"], seed=seed + r, **s.get("params", {}))
                ens += B.predict_member(st, X.iloc[te])
            score[te] += ens / len(spec)
            cnt[te] += 1
    return score / np.maximum(cnt, 1)


def auc_report(tab: pd.DataFrame, score: np.ndarray, n_boot: int = 300) -> dict:
    """AUC oof-оценки на размеченных циклах: общий (с интервалом по видео) и против каждого негативного класса."""
    from .model_tools import auc_with_ci

    m = np.isfinite(tab.y.to_numpy(float))
    d = tab[m].assign(s=score[m])
    a, lo, hi = auc_with_ci(d.y.to_numpy(int), d.s.to_numpy(), d.video.to_numpy(), n_boot=n_boot)
    per = {}
    pos = d[d.y == 1]
    for lab, g in d[d.y == 0].groupby("label"):
        if len(g) >= 5 and len(pos):
            from sklearn.metrics import roc_auc_score

            per[lab] = dict(n=int(len(g)), auc=float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(g))], np.r_[pos.s, g.s])))
    return dict(n=int(len(d)), positives=int(d.y.sum()), clips=int(d.video.nunique()), auc=a, lo=lo, hi=hi, per_class=per)


def track_box_at(clips: dict[str, ClipData]):
    """box_at(clip, person, t) по трекам запуска: person_gt_id вида `t<N>` — трек N. Нужен, чтобы человек, сместившийся за эпизод, не терял совпадение по IoU (рамка эталона в протоколе — траектория)."""
    cache: dict[str, Tracks] = {}

    def fn(clip: str, person: str, t: float):
        if clip not in clips or not str(person).startswith("t"):
            return None
        if clip not in cache:
            cache[clip] = Tracks.load(clips[clip].run_dir / "pose")
        try:
            return cache[clip].box_at(int(str(person)[1:]), t)
        except ValueError:
            return None

    return fn


def assemble(clips: dict[str, ClipData], score: np.ndarray, tab: pd.DataFrame, cycle_th: float, camera: str = "cam_local", overrides: dict | None = None) -> pd.DataFrame:
    """События всех клипов по оценкам цикла `score` (позиционно соответствует `tab`) и порогу цикла `cycle_th`; остальные параметры — текущий конфиг."""
    sc_all = tab[KEY].assign(score=score)
    rows = []
    for name, c in clips.items():
        rd = c.run_dir
        cfg = cfg_current_for_run(rd)
        cfg["events"]["cycle_th"] = float(cycle_th)
        for k, v in (overrides or {}).items():
            cfg["events"][k] = v
        tr = Tracks.load(rd / "pose")
        ser = stages.stage_features(tr, cfg, rd)
        cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
        s = sc_all[sc_all.video == name]
        scores = {(int(r.tid), round(float(r.start), 3)): float(r.score) for r in s.itertuples()}
        df, _ = stages.assemble_events(tr, ser, cyc, cfg, camera, name, cycle_scores=scores)
        rows.append(df.assign(clip_id=name))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_events(events: pd.DataFrame, clips: dict[str, ClipData], durations: dict[str, float], gt_df: pd.DataFrame | None = None, target_f1: float = 0.8,
                    use_track_boxes: bool = True, n_boot: int = 300) -> EV.Report:
    """Отчёт `evaluation.build_report` по событиям и эталону из `labels/events_gt.csv` (только клипы с эталоном)."""
    gt_df = GT.load() if gt_df is None else gt_df
    reviewed = {c for c in durations if c in set(gt_df.clip_id)}
    dur = {c: durations[c] for c in reviewed}
    gts = GT.to_eval_gts(gt_df, reviewed, track_box_at(clips) if use_track_boxes else None)
    preds = [p for c in reviewed for p in EV.to_preds(events[events.clip_id == c], c)] if len(events) else []
    return EV.build_report(preds, gts, dur, EV.Settings(target_f1=target_f1), n_boot=n_boot)


def grid(clips: dict[str, ClipData], tab: pd.DataFrame, score: np.ndarray, durations: dict[str, float], cycle_ths: Sequence[float] = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
         gt_df: pd.DataFrame | None = None, use_track_boxes: bool = True) -> pd.DataFrame:
    """Лучший F1 по порогу события для каждого порога цикла (кривая и плато — в отчёте); таблица для выбора `events.cycle_th`."""
    rows = []
    for th in cycle_ths:
        ev = assemble(clips, score, tab, th)
        rep = evaluate_events(ev, clips, durations, gt_df, use_track_boxes=use_track_boxes, n_boot=50)
        c = rep.curve
        b = c.loc[c.f1.idxmax()] if c.f1.notna().any() else None
        pl = rep.plateau or {}
        rows.append(dict(cycle_th=th, events=len(ev), best_f1=float(b.f1) if b is not None else np.nan, event_th=float(b.threshold) if b is not None else np.nan,
                         precision=float(b.precision) if b is not None else np.nan, recall=float(b.recall) if b is not None else np.nan,
                         tp=int(b.tp) if b is not None else 0, fp=int(b.fp) if b is not None else 0, fn=int(b.fn) if b is not None else 0,
                         plateau_center=pl.get("center"), plateau_f1=pl.get("best_f1")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ уровень эпизода (события)
EP_FEATS = ["n_cycles", "s_mean", "s_max", "s_min", "span", "hold_mean", "log_cycles"]


def assemble_detail(clips: dict[str, ClipData], score: np.ndarray, tab: pd.DataFrame, cycle_th: float, camera: str = "cam_local") -> pd.DataFrame:
    """Как `assemble`, но с признаками эпизода для каждого события: число циклов, средняя/максимальная/минимальная оценка циклов, длительность, средняя пауза у рта."""
    sc_all = tab[KEY].assign(score=score)
    rows = []
    for name, c in clips.items():
        rd = c.run_dir
        cfg = cfg_current_for_run(rd)
        cfg["events"]["cycle_th"] = float(cycle_th)
        tr = Tracks.load(rd / "pose")
        ser = stages.stage_features(tr, cfg, rd)
        cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
        s = sc_all[sc_all.video == name]
        scores = {(int(r.tid), round(float(r.start), 3)): float(r.score) for r in s.itertuples()}
        df, events = stages.assemble_events(tr, ser, cyc, cfg, camera, name, cycle_scores=scores)
        evs = sorted(events, key=lambda e: e.start)
        feats = []
        for e in evs:
            ss = [g.score for g in e.cycles]
            feats.append(dict(n_cycles=len(ss), s_mean=float(np.mean(ss)), s_max=float(np.max(ss)), s_min=float(np.min(ss)), span=float(e.end - e.start),
                              hold_mean=float(np.mean([g.cycle.hold for g in e.cycles])), log_cycles=float(np.log(len(ss)))))
        if len(df):
            rows.append(pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1).assign(clip_id=name))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def label_events(events: pd.DataFrame, clips: dict[str, ClipData], durations: dict[str, float], gt_df: pd.DataFrame | None = None, use_track_boxes: bool = True) -> np.ndarray:
    """1 — событие совпало с эталоном по протоколу (TP, один к одному), 0 — иначе. Нужно для калибровки уверенности эпизода."""
    from .evaluate import evaluate

    gt_df = GT.load() if gt_df is None else gt_df
    reviewed = {c for c in durations if c in set(gt_df.clip_id)}
    gts = GT.to_eval_gts(gt_df, reviewed, track_box_at(clips) if use_track_boxes else None)
    preds, idx = [], []
    for i, r in enumerate(events.itertuples()):
        if r.clip_id in reviewed:
            preds.append(EV.to_preds(events.iloc[[i]], r.clip_id)[0])
            idx.append(i)
    res = evaluate(preds, gts, **EV.Settings().kw())
    y = np.zeros(len(events))
    for pi, _ in res.matches:
        y[idx[pi]] = 1.0
    return y


def calibrate_cv(events: pd.DataFrame, y: np.ndarray, n_splits: int = 5, C: float = 0.5, seed: int = 0, feats: Sequence[str] = ("s_mean", "s_max", "log_cycles", "span")) -> np.ndarray:
    """Уверенность эпизода = out-of-fold вероятность логрегрессии по признакам эпизода (фолды по клипам); вместо заглушки `episode_confidence` с ручными весами."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = events[list(feats)].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0)
    groups = events.clip_id.to_numpy()
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    fold = {v: i % n_splits for i, v in enumerate(rng.permutation(uniq))}
    f = np.array([fold[g] for g in groups])
    out = np.full(len(events), np.nan)
    for k in range(n_splits):
        te, tr = np.flatnonzero(f == k), np.flatnonzero(f != k)
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        out[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return np.where(np.isnan(out), events.confidence.to_numpy(float), out)


def nested_thresholds(events_by_th: dict[float, pd.DataFrame], clips: dict[str, ClipData], durations: dict[str, float], gt_df: pd.DataFrame | None = None, n_splits: int = 5,
                      seed: int = 0, event_ths: Sequence[float] = tuple(np.round(np.arange(0.1, 0.801, 0.025), 3)), use_track_boxes: bool = True, by_scene: bool = False) -> dict:
    """Честная оценка ПРОЦЕДУРЫ подбора порогов: пара (порог цикла, порог события) выбирается по максимуму F1 на клипах-обучении и применяется к отложенным клипам (фолды по клипам);
    итоговые TP/FP/FN суммируются по отложенным частям. Цифра ниже, чем «лучший F1 на всех», на величину оптимизма подбора."""
    gt_df = GT.load() if gt_df is None else gt_df
    reviewed = sorted(c for c in durations if c in set(gt_df.clip_id))
    gts_all = GT.to_eval_gts(gt_df, set(reviewed), track_box_at(clips) if use_track_boxes else None)
    st = EV.Settings()
    rng = np.random.default_rng(seed)
    units = sorted({scene_group(c) if by_scene else c for c in reviewed})            # единицы разбиения: клипы или сцены
    perm = list(rng.permutation(units))
    fold_of = {u: i % n_splits for i, u in enumerate(perm)}
    folds = [[c for c in reviewed if fold_of[scene_group(c) if by_scene else c] == k] for k in range(n_splits)]

    def score(th_c: float, th_e: float, subset: Sequence[str]) -> tuple[int, int, int]:
        ev = events_by_th[th_c]
        d = {c: durations[c] for c in subset}
        preds = [p for c in subset for p in EV.to_preds(ev[ev.clip_id == c], c)] if len(ev) else []
        m, _, _ = EV.metrics_at(preds, [g for g in gts_all if g.clip_id in d], st, d, th_e)
        return m["tp"], m["fp"], m["fn"]

    tp = fp = fn = 0
    chosen = []
    for k, test in enumerate(folds):
        train = [c for c in reviewed if c not in test]
        best, arg = -1.0, None
        for th_c in events_by_th:
            for th_e in event_ths:
                a, b, c_ = score(th_c, th_e, train)
                f = EV.f1_of(a, b, c_)
                if f > best:
                    best, arg = f, (th_c, th_e)
        a, b, c_ = score(arg[0], arg[1], test)
        tp, fp, fn = tp + a, fp + b, fn + c_
        chosen.append(dict(fold=k, cycle_th=arg[0], event_th=arg[1], train_f1=round(best, 3), test_tp=a, test_fp=b, test_fn=c_))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=EV.f1_of(tp, fp, fn), folds=chosen)


# ------------------------------------------------------------------------------------------ исследование наборов признаков
def study(clips: dict[str, ClipData], tab: pd.DataFrame, durations: dict[str, float], sets: Sequence[str], kinds=("lr", "gb"), repeats: int = 3,
          cycle_ths: Sequence[float] = (0.2, 0.3, 0.4, 0.5), nested_seeds: int = 3, progress=None, by_scene: bool = False) -> tuple[pd.DataFrame, dict]:
    """Для каждого набора признаков: AUC циклов out-of-fold (с интервалом по видео и по классам жестов), лучший F1 события «на всех клипах» (оптимистичный: пороги подобраны по ним же)
    и F1 с вложенным подбором порогов (честная оценка процедуры). Возвращает (таблица, {набор: оценки циклов})."""
    rows, scores = [], {}
    for i, name in enumerate(sets):
        feats = CM.available(CM.SETS[name], tab)
        sc = oof_scores(tab, name, repeats=repeats, kinds=kinds, by_scene=by_scene)
        scores[name] = sc
        a = auc_report(tab, sc, n_boot=300)
        g = grid(clips, tab, sc, durations, cycle_ths=cycle_ths)
        best = g.loc[g.best_f1.idxmax()]
        ev = {th: assemble(clips, sc, tab, th) for th in cycle_ths}
        nested = [nested_thresholds(ev, clips, durations, seed=s, by_scene=by_scene)["f1"] for s in range(nested_seeds)]
        rows.append(dict(набор=name, признаков=len(feats), auc=a["auc"], auc_lo=a["lo"], auc_hi=a["hi"], **{f"auc_{k}": v["auc"] for k, v in a["per_class"].items()},
                         f1_лучший=float(best.best_f1), порог_цикла=float(best.cycle_th), порог_события=float(best.event_th), P=float(best.precision), R=float(best.recall),
                         tp=int(best.tp), fp=int(best.fp), fn=int(best.fn), f1_вложенный_мин=float(np.min(nested)), f1_вложенный_медиана=float(np.median(nested)), f1_вложенный_макс=float(np.max(nested))))
        if progress:
            progress(i + 1, len(sets), name)
    return pd.DataFrame(rows), scores
