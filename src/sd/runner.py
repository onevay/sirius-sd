"""Прогон выбранных папок профилем моделей и оценка: «оценка только на выбранных директориях».

    evaluate_dirs(dirs, profile, mode="events")  →  видео этих папок → события (кэш по отпечатку профиля) → метрики с опорой на целевую F1 → запись в журнал экспериментов

Режимы:
  events — по ручному эталону событий (`gt.py`, протокол организаторов): TP/FP/FN, P/R/F1, ложные тревоги в час, задержка. Оцениваются только размеченные клипы выбранных папок.
  clips  — слабая оценка по имени папки (курение / не курение), если события ещё не размечены: грубая проверка, что система различает клипы.

Роль набора: `validation` — порог можно выбирать по кривой (центр плато); `hidden` — порог берётся ТОЛЬКО из профиля (зафиксирован заранее), кривая показывается справочно.
Предсказания всех клипов сохраняются ДО фильтра по порогу, поэтому смена порога не требует повторного прогона.

Тяжёлая часть (`pipeline.recognize`) и чтение метаданных видео подставляются параметрами `recognize_fn`/`probe_fn` — на этом держатся тесты без моделей и видео.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import evaluation as EV
from . import experiments as XP
from . import gt as GT
from . import library as LIB
from . import roi as ROI
from .config import stable_hash
from .paths import OUTPUTS, from_portable, portable, video_id, weak_label
from .profiles import Profile

CACHE_DIR = OUTPUTS / "eval_cache"
Progress = Callable[[int, int, str], None] | None
EVENT_COLUMNS = ["camera_id", "clip_id", "event_id", "start_sec", "end_sec", "confidence", "label", "person_track_id", "peak_sec", "x1", "y1", "x2", "y2"]


@dataclass
class ClipRun:
    clip_id: str
    video: Path
    duration: float                       # длительность окна, с (по ней считаются «в час»)
    window: tuple[float, float | None]
    events: pd.DataFrame                  # ВСЕ события клипа (до фильтра по порогу), clip_id = уникальный id клипа
    cycles: int = 0
    people: int = 0
    seconds: float = 0.0
    cached: bool = False
    run_dir: str | None = None            # каталог позы (tracks.parquet) — для просмотра
    out_dir: str | None = None
    error: str | None = None


@dataclass
class Outcome:
    report: EV.Report
    runs: list[ClipRun]
    meta: dict
    run_id: str | None = None
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    gt: pd.DataFrame = field(default_factory=pd.DataFrame)


def _default_recognize(video, start, end, profile: Profile):
    from . import pipeline as P

    return P.recognize(video, start, end, profile.cfg(), profile.options_obj(render=False))


def _default_probe(video):
    from .video_io import probe

    return probe(video)


def window_for(duration: float, window: tuple[float, float | None] = (0.0, None), max_sec: float | None = None) -> tuple[float, float | None]:
    """Окно анализа: (начало, конец|None). `max_sec` ограничивает длину окна (пробные прогоны на длинных клипах)."""
    a, b = float(window[0] or 0.0), window[1]
    if b is None and max_sec and duration > a + max_sec + 0.5:
        b = round(a + max_sec, 2)
    return a, (float(b) if b is not None else None)


def _key(profile: Profile, video: Path, win: tuple[float, float | None]) -> str:
    st = video.stat()
    return stable_hash(dict(fp=profile.fingerprint(), v=portable(video), size=st.st_size, mt=st.st_mtime_ns, w=list(win)), 12)


def run_clips(videos: Sequence[Path], profile: Profile, *, window: tuple[float, float | None] = (0.0, None), max_sec: float | None = None,
              zones: list | None = None, use_cache: bool = True, recognize_fn: Callable | None = None, probe_fn: Callable | None = None, progress: Progress = None,
              cache_root: Path | None = None) -> list[ClipRun]:
    """События по каждому клипу. Ошибка одного клипа не останавливает прогон: клип получает пустой список событий и текст ошибки (в метриках это пропуск, а не «чисто»)."""
    recognize_fn = recognize_fn or _default_recognize
    probe_fn = probe_fn or _default_probe
    root = cache_root or CACHE_DIR
    out: list[ClipRun] = []
    for i, v in enumerate(videos):
        cid = video_id(v)
        t0 = time.perf_counter()
        run = ClipRun(cid, v, 0.0, window, pd.DataFrame(columns=EVENT_COLUMNS))
        try:
            dur = float(probe_fn(v).duration)
            win = window_for(dur, window, max_sec)
            run.window = win
            run.duration = float((win[1] if win[1] is not None else dur) - win[0])
            d = root / _key(profile, v, win)
            f = d / "events.csv"
            if use_cache and f.exists() and (d / "meta.json").exists():
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                ev = pd.read_csv(f, encoding="utf-8-sig")
                run.cached = True
            else:
                res = recognize_fn(v, win[0], win[1], profile)
                ev = res.events.copy()
                meta = dict(run_dir=portable(res.meta.get("run_dir")), out_dir=portable(res.meta.get("out_dir")), cycles=res.meta["counts"]["cycles"], people=res.meta["counts"]["people"],
                            warnings=res.meta.get("warnings", []))
                d.mkdir(parents=True, exist_ok=True)
                ev.to_csv(f, index=False, encoding="utf-8-sig")
                (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            ev = ev.assign(clip_id=cid) if len(ev) else pd.DataFrame(columns=EVENT_COLUMNS)
            run.events = ROI.filter_events(ev, zones)
            run.cycles, run.people, run.run_dir, run.out_dir = int(meta.get("cycles", 0)), int(meta.get("people", 0)), meta.get("run_dir"), meta.get("out_dir")
        except Exception as e:   # битый файл, нет кадров, нет модели
            run.error = f"{type(e).__name__}: {str(e)[:160]}"
        run.seconds = round(time.perf_counter() - t0, 2)
        out.append(run)
        if progress:
            progress(i + 1, len(videos), cid + (" (кэш)" if run.cached else "") + (" — ОШИБКА" if run.error else ""))
    return out


def restrict_gts(gts: Sequence, windows: dict[str, tuple[float, float | None]], durations: dict[str, float]) -> list:
    """Эталон в пределах окна анализа: интервалы целиком вне окна не считаются пропусками (иначе пробный прогон «первые 60 с» штрафовался бы за всё остальное)."""
    keep = []
    for g in gts:
        a, b = windows.get(g.clip_id, (0.0, None))
        end = b if b is not None else a + durations.get(g.clip_id, 0.0)
        if g.end > a and g.start < end:
            keep.append(g)
    return keep


def _weak_frame(runs: Iterable[ClipRun]) -> pd.DataFrame:
    rows = []
    for r in runs:
        lab = weak_label(r.video)
        if lab is None:
            continue
        rows.append(dict(clip_id=r.clip_id, y=int(lab == "smoking"), duration=r.duration, events=[float(c) for c in r.events.confidence] if len(r.events) else []))
    return pd.DataFrame(rows, columns=["clip_id", "y", "duration", "events"])


def with_classifier(profile: Profile, classifier: str | Path | None) -> Profile:
    """Копия профиля с выбранным классификатором цикла (выбор классификатора — часть проверки, а не скрытая деталь профиля)."""
    if not classifier:
        return profile
    return Profile(profile.name, profile.description, profile.base, dict(profile.config), {**profile.options, "cycle_bundle": str(classifier)})


def evaluate_dirs(dirs: Sequence[str | Path], profile: Profile, *, classifier: str | Path | None = None, mode: str = "events", role: str = "validation", policy: str = "plateau", target_f1: float = 0.80,
                  only_labeled: bool = True, window: tuple[float, float | None] = (0.0, None), max_sec: float | None = None, roi_path: str | Path | None = None,
                  use_cache: bool = True, name: str | None = None, save: bool = True, gt_df: pd.DataFrame | None = None, n_boot: int = 300,
                  recognize_fn: Callable | None = None, probe_fn: Callable | None = None, progress: Progress = None, cache_root: Path | None = None,
                  exp_root: Path | None = None, settings: EV.Settings | None = None) -> Outcome:
    """Полный цикл: клипы выбранных папок → события → отчёт → журнал. Исключения: `ValueError` — нечего оценивать (нет размеченных клипов / нет клипов)."""
    if mode not in ("events", "clips"):
        raise ValueError("mode ∈ {events, clips}")
    from . import solver as SV

    profile = with_classifier(profile, classifier)
    errs = SV.errors(profile, devices=recognize_fn is None)           # до долгого прогона: нет классификатора, нет нужных экстракторов, неверный fusion
    if errs:
        raise ValueError("профиль не готов к оценке:\n  - " + "\n  - ".join(e.text for e in errs))
    if role not in ("validation", "hidden"):
        raise ValueError("role ∈ {validation, hidden}")
    if role == "hidden" and policy != "fixed":
        raise ValueError("на скрытом наборе порог фиксируется заранее (policy='fixed'): подбирать его по скрытым меткам запрещено")
    videos = LIB.videos_in(dirs)
    if not videos:
        raise ValueError("в выбранных папках нет видео")
    ids = LIB.clip_ids(videos)
    gt_all = GT.load() if gt_df is None else gt_df
    notes: list[str] = []
    skipped: list[str] = []
    if mode == "events" and only_labeled:
        status = GT.clip_status(gt_all, ids)
        reviewed = set(status[status.reviewed].clip_id)
        skipped = [c for c in ids if c not in reviewed]
        videos = [ids[c] for c in ids if c in reviewed]
        if not videos:
            raise ValueError("в выбранных папках нет размеченных клипов: разметьте события на странице «Разметка» или переключите режим на «по папкам»")
        if skipped:
            notes.append(f"пропущено неразмеченных клипов: {len(skipped)} из {len(ids)}")
    if mode == "clips":
        keep = [v for v in videos if weak_label(v)]
        if len(keep) < len(videos):
            notes.append(f"без метки по папке пропущено клипов: {len(videos) - len(keep)}")
        videos = keep
        if not videos or len({weak_label(v) for v in videos}) < 2:
            raise ValueError("для слабой оценки нужны клипы обоих классов («курение» и без курения) в выбранных папках")
    zones = ROI.load(roi_path) if roi_path else None
    prof = profile
    st = settings or EV.Settings(threshold=prof.threshold, target_f1=target_f1)
    runs = run_clips(videos, prof, window=window, max_sec=max_sec, zones=zones, use_cache=use_cache, recognize_fn=recognize_fn, probe_fn=probe_fn, progress=progress, cache_root=cache_root)
    failed = [r for r in runs if r.error]
    if save and failed and len(failed) == len(runs):      # ни один клип не обработан: «F1 = 0» в журнале выглядел бы как плохая модель, а это сбой запуска (нет весов, CUDA, битый файл)
        raise RuntimeError("ни один клип не обработан, прогон не сохранён. Первая ошибка: " + failed[0].error + (f" (всего {len(failed)} клипов с ошибкой)" if len(failed) > 1 else ""))
    if failed:
        notes.append(f"клипов с ошибкой обработки: {len(failed)} (посчитаны как «событий нет»): " + "; ".join(f"{r.clip_id}: {r.error}" for r in failed[:3]))
    gt_end = gt_all.groupby("clip_id").end_sec.max().to_dict() if len(gt_all) else {}
    # клип, у которого не удалось даже прочитать длительность, остаётся в оценке (событий нет → пропуски эталона = FN); иначе сбой молча улучшал бы метрику
    durations = {r.clip_id: (r.duration if r.duration > 0 else max(float(gt_end.get(r.clip_id, 0.0)), 1.0)) for r in runs}
    events_all = pd.concat([r.events.assign(clip_id=r.clip_id) for r in runs if len(r.events)], ignore_index=True) if any(len(r.events) for r in runs) else pd.DataFrame(columns=EVENT_COLUMNS)
    policy_eff = "fixed" if role == "hidden" else policy
    if mode == "events":
        gts = restrict_gts(GT.to_eval_gts(gt_all, set(durations)), {r.clip_id: r.window for r in runs}, durations)
        preds = [p for r in runs for p in EV.to_preds(r.events, r.clip_id)]
        rep = EV.finalize_report(preds, gts, durations, st, policy_eff, n_boot=n_boot, weak={r.clip_id: (weak_label(r.video) or "") for r in runs})
        gt_used = gt_all[gt_all.clip_id.isin(set(durations))]
    else:
        wf = _weak_frame(runs)
        rep = EV.build_weak_report(wf, st, n_boot=n_boot)
        if policy_eff != "fixed":
            th = EV.choose_threshold(rep, policy_eff, st.threshold)
            if abs(th - st.threshold) > 1e-9:
                rep = EV.build_weak_report(wf, replace(st, threshold=th), n_boot=n_boot)
                rep.notes.append(f"порог {th:.3f} выбран по кривой этого же набора: цифра оптимистична")
        gt_used = pd.DataFrame(columns=GT.COLUMNS)
    if role == "hidden":
        rep.notes.append("скрытый набор: порог взят из профиля и не подбирался; кривая — только справочно")
    rep.notes = notes + rep.notes
    meta = dict(name=name or prof.name, profile=prof.to_dict(), describe=prof.describe(), fingerprint=prof.fingerprint(), mode=mode, role=role, policy=policy_eff,
                dirs=[str(d) for d in dirs], clips=[dict(clip_id=r.clip_id, path=portable(r.video), duration=r.duration, window=list(r.window), cached=r.cached, seconds=r.seconds,
                                                         run_dir=r.run_dir, out_dir=r.out_dir, error=r.error, cycles=r.cycles, people=r.people) for r in runs],
                skipped_unlabeled=skipped, gt_fingerprint=GT.fingerprint(gt_all, set(durations)) if mode == "events" else None, gt_rows=int(len(gt_used)),
                roi=str(roi_path) if roi_path else None, max_sec=max_sec, target_f1=target_f1, total_seconds=round(sum(r.seconds for r in runs), 1))
    rid = None
    if save:
        rid = XP.save_run(rep, meta, events_all, gt_used, root=exp_root).name
    return Outcome(rep, runs, meta, rid, events_all, gt_used)
