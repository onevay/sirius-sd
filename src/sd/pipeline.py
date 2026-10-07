"""Полный цикл распознавания одной командой: видео → поза+трекинг → циклы → признаки → оценка цикла → события по регламенту (+ видео с разметкой).

Один код для `sd recognize` и вкладки «Распознавание» веб-интерфейса. Все промежуточные таблицы пишутся в каталог запуска
`outputs/recognize/<видео>/<окно>_<время>/`, поэтому результат проверяется без повторного расчёта:
  result.json        — что запускалось: версии пакетов моделей (sha256), пороги, время этапов, предупреждения;
  cycles_scored.csv  — все циклы с признаками и оценками (score_cheap, vlm_yesno, score);
  events.csv/.json   — события по регламенту (поля как в протоколе оценки);
  analysis/          — таблицы признаков по группам (те же имена колонок, что при обучении — `feature_auc`, `bundle`);
  overlay.mp4        — видео с точками, состояниями автомата, оценками циклов и событиями (если render=True).

Каскад «дёшево → дорого»: дешёвые признаки (кинематика, поза, предмет, фото-модель) считаются для всех циклов; VLM (≈ 9 с на цикл на iGPU) — для всех
(`vlm_mode=all`), только для «серой зоны» оценки дешёвого пакета (`grey`) или не считается (`off`).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import analysis as A
from . import feature_auc as FA
from . import stages
from .config import stable_hash
from .paths import OUTPUTS, ROOT, video_id

Progress = Callable[[str, int, int], None] | None
KEY = ["video", "tid", "start"]


@dataclass
class Options:
    cycle_bundle: str | None = None          # models/cycle/<имя>: пакет без VLM-признаков («дешёвый»)
    cycle_bundle_full: str | None = None     # пакет с VLM-признаком (для серой зоны / all)
    photo_bundle: str | None = None          # models/photo/<имя>
    objects: tuple = ()                      # id детекторов предмета (smoking_yolo11m_beehzod ...)
    vlm_model: str | None = None             # тег Ollama, например qwen3.5:2b-q4_K_M
    vlm_mode: str = "off"                    # off | grey | all
    grey: tuple = (0.3, 0.8)
    render: bool = False
    camera_id: str = "cam_local"
    backend: str = "auto"                    # рантайм эмбеддингов фото-модели: auto | ov | torch


@dataclass
class Result:
    out_dir: Path
    cycles: pd.DataFrame
    events: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")) if path.is_dir() else [path]:
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def bundle_info(p: str | Path | None, kind: str) -> dict | None:
    if not p:
        return None
    d = _abs(p)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return dict(path=str(d), kind=kind, sha16=_sha(d), format=man.get("format"), created=man.get("created"), n=man.get("n"),
                cv=man.get("cv") or man.get("oof_auc"), features=len(man.get("features", [])) or None)


def _with_end(tab: pd.DataFrame) -> pd.DataFrame:
    """В таблице признаков конец цикла хранится как длительность `dur` (колонка `end` убирается в `_cycles_with_durations`); для плеера/рендера нужен `end`."""
    if "end" not in tab.columns and {"start", "dur"} <= set(tab.columns):
        tab = tab.assign(end=tab["start"] + tab["dur"])
    return tab


def fast_cycle_features(tab: pd.DataFrame, pose_df: pd.DataFrame) -> pd.DataFrame:
    """Признаки «быстрого» пакета (кинематика + ритм + поза) для таблицы циклов `dataset.build_cycle_table` + `pose_feats.pose_rows`: добавляет `dur`, `mouth_dur`
    и присоединяет колонки группы H по ключу (video, tid, start). Детекторов и моделей не требует — миллисекунды на цикл; нужен для `clips-run --cycle-bundle`."""
    t = tab.copy()
    t["dur"] = t["end"] - t["start"]
    t["mouth_dur"] = t["mouth_out"] - t["mouth_in"]
    if len(pose_df):
        p = pose_df.assign(start=pose_df["start"].round(3))
        t = t.assign(start=t["start"].round(3)).merge(p, on=KEY, how="left", suffixes=("", "_dup"))
        t = t.drop(columns=[c for c in t.columns if c.endswith("_dup")])
    return t


def grey_mask(score: np.ndarray, grey: tuple[float, float]) -> np.ndarray:
    lo, hi = grey
    return (score >= lo) & (score <= hi)


def cascade(score_cheap: np.ndarray, vlm_available: np.ndarray, score_full: np.ndarray | None) -> np.ndarray:
    """Итоговая оценка: где есть ответ VLM и пакет с VLM — оценка полного пакета, иначе — дешёвого."""
    out = np.asarray(score_cheap, float).copy()
    if score_full is not None:
        use = np.asarray(vlm_available, bool) & np.isfinite(score_full)
        out[use] = score_full[use]
    return out


def recognize(video: str | Path, start: float, end: float | None, cfg: dict, opts: Options, out_root: Path = OUTPUTS / "recognize",
              progress: Progress = None, force_pose: bool = False) -> Result:
    from .bundle import Bundle
    from .dataset import build_cycle_table
    from .pose_feats import pose_rows

    video = Path(video)
    clip = video.stem
    t_all = time.perf_counter()
    timing: dict[str, float] = {}
    warnings: list[str] = []

    def tick(name: str, t0: float) -> None:
        timing[name] = round(time.perf_counter() - t0, 2)

    def pr(stage: str):
        return (lambda i, n: progress(stage, i, n)) if progress else None

    # ---- 1. поза + трекинг, признаки ряда, циклы (кэш по параметрам: повторный запуск почти мгновенный)
    t0 = time.perf_counter()
    tr, rd = stages.stage_pose(video, cfg, start, end, force=force_pose, progress=pr("поза+трекинг"))
    tick("pose", t0)
    t0 = time.perf_counter()
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, st_ = stages.stage_cycles(ser, cfg, rd)
    tick("features_cycles", t0)
    vname = rd.parent.name
    out_dir = out_root / vname / f"{start:g}-{'end' if end is None else format(end, 'g')}s_{time.strftime('%Y%m%d_%H%M%S')}"
    an = out_dir / "analysis"
    an.mkdir(parents=True, exist_ok=True)
    cycles = A.cycles_frame(vname, rd, tr, cfg, ser, cyc)
    ctx = {str(rd): (tr, cfg, ser)}
    tab = pd.DataFrame()
    score = np.array([])
    scored = cycles.copy()

    if len(cycles):
        cycles.to_parquet(an / "cycles_all.parquet", index=False)
        # ---- 2. признаки по группам (имена колонок те же, что при обучении)
        t0 = time.perf_counter()
        ds = build_cycle_table(video, tr, ser, cyc)
        ds.to_parquet(an / "dataset_cycles.parquet", index=False)
        pd.DataFrame(pose_rows(vname, tr, cfg, ser, cyc)).to_parquet(an / "pose_cycles.parquet", index=False)
        tick("feat_kinematics_pose", t0)
        if opts.objects:
            t0 = time.perf_counter()
            pd.DataFrame(A.evidence_rows(cycles, tr, ser, cfg, list(opts.objects))).to_parquet(an / "evidence_cycles.parquet", index=False)
            tick("feat_object", t0)
        if opts.photo_bundle:
            t0 = time.perf_counter()
            A.photo_all(cycles, _abs(opts.photo_bundle), opts.backend, out=an / "photo_cycles.parquet", ctx=ctx,
                        progress=(lambda tag, i, n: progress(f"фото-модель · {tag}", i, n)) if progress else None)
            tick("feat_photo", t0)
        tab = _with_end(FA.all_cycles_features(an=an, vlm=None, dataset=an / "dataset_cycles.parquet"))
        # ---- 3. оценка цикла: дешёвый пакет
        if opts.cycle_bundle:
            b = Bundle(_abs(opts.cycle_bundle))
            miss = b.missing(tab)
            if miss:
                warnings.append(f"в таблице нет признаков пакета {len(miss)} шт. ({', '.join(miss[:6])}…): они считаются как пропуски; подключите экстракторы (--photo-bundle, --objects)")
            t0 = time.perf_counter()
            tab["score_cheap"] = b.score(tab)
            tick("score_cheap", t0)
        else:
            from .cycles import Cycle
            from .events import heuristic_cycle_score

            warnings.append("пакет классификатора цикла не задан: оценка — эвристика-заглушка по длительности паузы (events.heuristic_cycle_score)")
            q = {int(t): stages.track_quality(ser, int(t)) for t in tab.tid.unique()}
            tab["score_cheap"] = [heuristic_cycle_score(Cycle(int(r.tid), r.start, r.mouth_in, r.mouth_out, r.end, r.hold, int(r.hand), r.d_min), q[int(r.tid)])
                                  for r in cycles.merge(tab[KEY], on=KEY).itertuples()] if len(tab) == len(cycles) else np.nan
        score = tab["score_cheap"].to_numpy(float)
        # ---- 4. VLM (дорого): все циклы или серая зона
        if opts.vlm_model and opts.vlm_mode in ("all", "grey"):
            from .vlm import OllamaVLM
            from .vlm_eval import run_vlm

            sel = np.ones(len(tab), bool) if opts.vlm_mode == "all" else grey_mask(score, opts.grey)
            sub = tab[sel][KEY].merge(cycles, on=KEY, how="left")
            t0 = time.perf_counter()
            if len(sub):
                run_vlm(sub, {"yesno": OllamaVLM(opts.vlm_model, size=224, mode="yesno")}, an / "vlm.parquet", 6, 2.5, pr("VLM по циклам"), "mouth", 2.5, 224, ctx=ctx)
                tab = _with_end(FA.all_cycles_features(an=an, vlm=an / "vlm.parquet" if (an / "vlm.parquet").exists() else None, dataset=an / "dataset_cycles.parquet"))
                tab["score_cheap"] = score
            tick("vlm", t0)
            tab["vlm_called"] = sel
            if opts.cycle_bundle_full and "vlm_yesno" in tab:
                bf = Bundle(_abs(opts.cycle_bundle_full))
                avail = tab["vlm_yesno"].notna().to_numpy()
                s_full = np.full(len(tab), np.nan)
                if avail.any():
                    s_full[avail] = bf.score(tab[avail])
                score = cascade(score, avail, s_full)
            elif "vlm_yesno" in tab and tab["vlm_yesno"].notna().any():
                warnings.append("VLM посчитан, но пакет с VLM-признаком (cycle_bundle_full) не задан: итоговая оценка = оценка дешёвого пакета; ответы VLM показаны в таблице")
        tab["score"] = score
        scored = tab
    scores = {(int(r.tid), round(float(r.start), 3)): float(r.score) for r in scored.itertuples() if hasattr(r, "score") and np.isfinite(r.score)} if len(scored) and "score" in scored else {}

    # ---- 5. события по регламенту
    t0 = time.perf_counter()
    ev_df, _ = stages.assemble_events(tr, ser, cyc, cfg, opts.camera_id, clip, cycle_scores=scores or None)
    tick("events", t0)
    if len(scored):
        scored.to_csv(out_dir / "cycles_scored.csv", index=False, encoding="utf-8-sig")
        scored.to_parquet(out_dir / "cycles_scored.parquet", index=False)
    ev_df.to_csv(out_dir / "events.csv", index=False)
    (out_dir / "events.json").write_text(json.dumps(json.loads(ev_df.to_json(orient="records")), ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 6. видео с разметкой
    if opts.render:
        from .render import Renderer, render_video

        t0 = time.perf_counter()
        cols = [c for c in ("tid", "start", "end", "score", "score_cheap", "vlm_yesno", "photo_p_mean") if c in scored.columns]
        extras = {"cycle_scores": scored[cols].copy()} if len(scored) else {}
        r = Renderer(tr, cfg, series=ser, cycles=cyc, states=st_, events=ev_df, extras=extras)
        render_video(video, tr, cfg, out_dir / "overlay.mp4", renderer=r, progress=pr("видео с разметкой"))
        tick("render", t0)

    meta = dict(video=str(video), clip=clip, window=[start, end], run_dir=str(rd), out_dir=str(out_dir), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                counts=dict(cycles=int(len(cycles)), rejected=int(len(rej)), events=int(len(ev_df)), people=int(len(tr.tids)),
                            vlm_calls=int(scored["vlm_called"].sum()) if "vlm_called" in scored else 0),
                options={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(opts).items()},
                models=dict(cycle=bundle_info(opts.cycle_bundle, "cycle"), cycle_full=bundle_info(opts.cycle_bundle_full, "cycle"), photo=bundle_info(opts.photo_bundle, "photo"),
                            objects=list(opts.objects), vlm=opts.vlm_model, pose=tr.meta.get("pose")),
                config_hash=stable_hash({k: cfg[k] for k in ("cycles", "events", "features", "evidence") if k in cfg}),
                timing_sec=dict(timing, total=round(time.perf_counter() - t_all, 2)), warnings=warnings)
    (out_dir / "result.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return Result(out_dir, scored, ev_df, meta)


def list_results(root: Path = OUTPUTS / "recognize") -> pd.DataFrame:
    """Все сохранённые прогоны распознавания (для вкладки «Плеер»): по result.json."""
    rows = []
    for f in sorted(root.glob("*/*/result.json")) if root.exists() else []:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(dict(dir=str(f.parent), clip=m.get("clip"), window=f"{m['window'][0]:g}–{'конец' if m['window'][1] is None else format(m['window'][1], 'g')} с",
                         created=m.get("created"), cycles=m["counts"]["cycles"], events=m["counts"]["events"], overlay=(f.parent / "overlay.mp4").exists(),
                         total_sec=m["timing_sec"].get("total")))
    return pd.DataFrame(rows).sort_values("created", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["dir", "clip", "window", "created", "cycles", "events", "overlay", "total_sec"])
