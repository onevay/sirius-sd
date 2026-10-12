"""Анализ прогонов по кэшу: таблица «клип × метрика» и автоматический поиск слабых мест — где и почему модель ошибается, без повторного прогона моделей.

Источники (всё уже лежит на диске после `sd eval`):
    outputs/experiments/<прогон>/  meta.json (клипы, время, запуски), per_clip.csv, errors.csv, events.csv
    запуск распознавания клипа     result.json (время этапов), cycles_scored.parquet (циклы с оценкой, рост человека H, достоверность точек)
    кэш позы                       pose/meta.json (кадры/с, мс на кадр), pose/tracks.parquet (рамки, рост h)
    метки жестов                   что за жест был рядом с ложной тревогой (питьё, телефон…)

Правила слабых мест детерминированы и перечислены в `RULES` — их можно проверить и оспорить, это не «чёрный ящик».
Сводка по клипам кэшируется в `<прогон>/clip_details.csv` (пересчитывается, если прогон новее).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import experiments as XP
from .paths import from_portable, weak_label

DETAILS = "clip_details.csv"
RULES = {
    "error": "клип не обработан (ошибка конвейера) — в метриках считается «событий нет»",
    "small_people": "люди мелкие (медиана роста трека < 150 px) и есть пропуск — нужен вход 1280–1600, тайлинг или ROI",
    "no_cycles": "в клипе с курением автомат не нашёл ни одного цикла — точки рук/рта не видны или поза не держит человека",
    "missed": "пропуск: цикл не найден, трек потерян или оценка цикла ниже порога — смотрите дорожку «Циклы»",
    "wrong_box": "время совпало, но рамка не того человека (IoU < 0.3) — склейка треков / два человека рядом",
    "below_threshold": "событие найдено, но уверенность ниже порога — признаки предмета, калибровка",
    "similar_gesture": "ложная тревога на похожем жесте (по меткам жестов) — нужны жёсткие негативы этого типа",
    "false_alarm": "ложная тревога вне эталона",
    "weak_conflict": "метка папки противоречит эталону — исправьте слабую метку или переложите файл",
    "low_kp": "низкая достоверность точек запястья в циклах (< 0.4) — свет, ракурс, размытие",
    "slow": "медленнее реального времени — 4K/плотная сцена: снизить fps, ROI, ускоритель",
    "leak": "клип был в обучении классификатора — его метрика оптимистична",
}
LEVEL = {"error": "error", "small_people": "error", "no_cycles": "error", "missed": "error", "wrong_box": "warning", "below_threshold": "warning", "similar_gesture": "warning",
         "false_alarm": "warning", "weak_conflict": "warning", "low_kp": "info", "slow": "info", "leak": "info"}


def _json(f: Path) -> dict:
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f and f.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def clip_details(meta: dict) -> pd.DataFrame:
    """Сводка по каждому клипу прогона из файлов на диске: время, скорость позы, рост людей, циклы, достоверность точек, время этапов."""
    rows = []
    for c in meta.get("clips", []):
        out, rd = from_portable(c.get("out_dir")), from_portable(c.get("run_dir"))
        res = _json(out / "result.json") if out else {}
        pose = _json(rd / "pose" / "meta.json") if rd else {}
        dur, sec = float(c.get("duration") or 0), float(c.get("seconds") or 0)
        r = dict(clip_id=c["clip_id"], folder=Path(str(from_portable(c.get("path")) or "")).parent.name, weak=weak_label(from_portable(c["path"])) if c.get("path") else None,
                 duration=dur, seconds=sec, cached=bool(c.get("cached")), error=c.get("error"), cycles=c.get("cycles"), people=c.get("people"),
                 pose_wall=pose.get("wall_sec"),
                 pose_fps=pose.get("fps_steady"), pose_ms=((pose.get("ms") or {}).get("total") or {}).get("median"), proc_fps=pose.get("proc_fps"),
                 width=(pose.get("video_info") or {}).get("width"), height=(pose.get("video_info") or {}).get("height"))
        r["pose_rt"] = round(r["pose_fps"] / r["proc_fps"], 2) if r["pose_fps"] and r["proc_fps"] else np.nan     # успевает ли поза за частотой обработки
        for k, v in (res.get("timing_sec") or {}).items():
            r[f"t_{k}"] = v
        # скорость конвейера по замерам кэша: поза — время её реального расчёта (pose/meta.json), остальное — время этапов последнего распознавания.
        # Время самого прогона оценки (seconds) для этого не годится: поза и признаки обычно берутся из кэша.
        win = c.get("window") or [0.0, None]
        pdur = float((pose.get("end") or dur) - (pose.get("start") or 0.0)) if pose else dur
        rest = max(0.0, float((res.get("timing_sec") or {}).get("total", 0.0)) - float((res.get("timing_sec") or {}).get("pose", 0.0)))
        if pose.get("wall_sec"):
            compute = float(pose["wall_sec"]) * (dur / pdur if pdur > 0 else 1.0) + rest
            r["rt_factor"] = round(dur / compute, 2) if compute > 0 else np.nan
        else:
            r["rt_factor"] = np.nan
        r["window"] = f"{win[0]:g}–{'конец' if win[1] is None else f'{win[1]:g}'}"
        if rd and (rd / "pose" / "tracks.parquet").exists():
            try:
                tr = pd.read_parquet(rd / "pose" / "tracks.parquet", columns=["tid", "h"])
                per = tr.groupby("tid").h.median()
                r["h_med"], r["h_p10"] = float(per.median()) if len(per) else np.nan, float(np.percentile(tr.h, 10)) if len(tr) else np.nan
            except Exception:       # noqa: BLE001 — старый формат кэша
                pass
        if out and (out / "cycles_scored.parquet").exists():
            try:
                cy = pd.read_parquet(out / "cycles_scored.parquet")
                r["wrist_conf"] = float(cy.wrist_conf.median()) if "wrist_conf" in cy and len(cy) else np.nan
                sc = next((k for k in ("score", "score_cheap", "score_final") if k in cy), None)
                r["score_max"] = float(cy[sc].max()) if sc and len(cy) else np.nan
                r["cycle_h"] = float(cy.H.median()) if "H" in cy and len(cy) else np.nan
            except Exception:       # noqa: BLE001
                pass
        rows.append(r)
    return pd.DataFrame(rows)


def run_table(rid: str, root: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Клип × метрика для прогона: per_clip (F1, TP/FP/FN, статус) + сводка `clip_details` + число ошибок каждого типа."""
    base = (root or XP.EXP_DIR) / rid
    rep, meta, _, gt = XP.load_run(rid, root)
    f = base / DETAILS
    if f.exists() and not refresh and f.stat().st_mtime >= (base / "meta.json").stat().st_mtime:
        det = pd.read_csv(f, encoding="utf-8-sig")
    else:
        det = clip_details(meta)
        det.to_csv(f, index=False, encoding="utf-8-sig")
    pc = rep.per_clip.copy()
    tab = det.merge(pc.drop(columns=[c for c in ("duration",) if c in pc]), on="clip_id", how="left") if len(pc) else det
    if len(rep.errors):
        cnt = rep.errors.pivot_table(index="clip_id", columns="cause", values="kind", aggfunc="count", fill_value=0)
        tab = tab.merge(cnt.add_prefix("err_"), left_on="clip_id", right_index=True, how="left")
    if len(gt) and "label" in gt:
        pos = gt[gt.label == "POSITIVE"].groupby("clip_id").size().rename("gt_pos")
        tab = tab.merge(pos, left_on="clip_id", right_index=True, how="left")
    leak = meta.get("leak")
    if leak is None and meta.get("profile"):        # старые прогоны: проверка утечки задним числом по пакету профиля
        from . import profiles as PR
        from .runner import training_leak

        try:
            pr = meta["profile"]
            leak = training_leak(PR.Profile(pr.get("name", "run"), pr.get("description", ""), pr.get("base"), pr.get("config") or {}, pr.get("options") or {}), list(tab.clip_id))
        except Exception:       # noqa: BLE001
            leak = {}
    tab["leak"] = tab.clip_id.isin(set((leak or {}).get("clips") or []))
    for c in [c for c in tab.columns if c.startswith("err_")] + ["gt_pos"]:
        if c in tab:
            tab[c] = tab[c].fillna(0).astype(int)
    return tab


def _gesture_near(gest: pd.DataFrame | None, clip: str, t: float, tol: float = 3.0) -> str | None:
    if gest is None or not len(gest) or not np.isfinite(t):
        return None
    g = gest[(gest.video == clip) & ((gest.peak_t - t).abs() <= tol)]
    g = g[~g.label.isin(["smoke", "unsure", "no_gesture"])]
    return str(g.iloc[int((g.peak_t - t).abs().to_numpy().argmin())].label) if len(g) else None


def weak_spots(tab: pd.DataFrame, errors: pd.DataFrame, gestures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Слабые места по правилам `RULES`: одна строка = клип + правило + пояснение. Сортировка: ошибки, предупреждения, справка."""
    out = []

    def add(clip, rule, detail=""):
        out.append(dict(level=LEVEL[rule], clip_id=clip, rule=rule, text=RULES[rule], detail=detail))

    err = errors if errors is not None and len(errors) else pd.DataFrame(columns=["clip_id", "kind", "cause", "peak", "start"])
    for r in tab.itertuples():
        e = err[err.clip_id == r.clip_id]
        fn = int((e.kind == "FN").sum()) if len(e) else 0
        if isinstance(r.error, str) and r.error and r.error != "nan":
            add(r.clip_id, "error", r.error[:160])
        h = getattr(r, "h_med", np.nan)
        if fn and np.isfinite(h) and h < 150:
            add(r.clip_id, "small_people", f"медиана роста {h:.0f} px, пропусков {fn}")
        if getattr(r, "gt_pos", 0) and (r.cycles == 0 or pd.isna(r.cycles)):
            add(r.clip_id, "no_cycles", f"позитивов в эталоне {r.gt_pos}")
        for cause, rule in (("missed", "missed"), ("wrong_box", "wrong_box"), ("wrong_person", "wrong_box"), ("below_threshold", "below_threshold")):
            n = int((e.cause == cause).sum()) if len(e) else 0
            if n and not (rule == "missed" and getattr(r, "gt_pos", 0) and r.cycles == 0):
                add(r.clip_id, rule, f"{n} шт., моменты: " + ", ".join(f"{t:.0f} с" for t in e[e.cause == cause].get("peak", e[e.cause == cause].start).head(4)))
        fps = e[e.kind == "FP"] if len(e) else e
        for f in fps.itertuples():
            t = getattr(f, "peak", np.nan)
            g = _gesture_near(gestures, r.clip_id, float(t) if t == t else np.nan)
            if g:
                add(r.clip_id, "similar_gesture", f"{t:.0f} с — рядом размечен жест «{g}»")
            else:
                add(r.clip_id, "false_alarm", f"{t:.0f} с" if t == t else "")
        w, pos = getattr(r, "weak", None), getattr(r, "gt_pos", 0)
        if w == "fake" and pos:
            add(r.clip_id, "weak_conflict", f"папка «лжекурение», а в эталоне {pos} эпизод(а) курения")
        wc = getattr(r, "wrist_conf", np.nan)
        if np.isfinite(wc) and wc < 0.4:
            add(r.clip_id, "low_kp", f"медиана достоверности запястья {wc:.2f}")
        rt = getattr(r, "rt_factor", np.nan)
        if np.isfinite(rt) and rt < 1:
            add(r.clip_id, "slow", f"{rt:.2f}× реального времени" + (f", кадр {int(r.width)}×{int(r.height)}" if np.isfinite(getattr(r, "width", np.nan) or np.nan) else ""))
        if getattr(r, "leak", False):
            add(r.clip_id, "leak")
    df = pd.DataFrame(out, columns=["level", "clip_id", "rule", "text", "detail"])
    order = {"error": 0, "warning": 1, "info": 2}
    return df.sort_values(["level", "clip_id"], key=lambda s: s.map(order) if s.name == "level" else s, kind="stable").reset_index(drop=True)


def summary(spots: pd.DataFrame) -> pd.DataFrame:
    """Сколько клипов затронуто каждым правилом (что чинить в первую очередь)."""
    if spots.empty:
        return pd.DataFrame(columns=["правило", "уровень", "клипов", "что значит"])
    g = spots.groupby("rule").agg(клипов=("clip_id", "nunique"), уровень=("level", "first")).reset_index()
    g["что значит"] = g.rule.map(RULES)
    order = {"error": 0, "warning": 1, "info": 2}
    return g.rename(columns={"rule": "правило"}).sort_values(["уровень", "клипов"], key=lambda s: s.map(order) if s.name == "уровень" else -s).reset_index(drop=True)


MATRIX_COLS = {"f1": ("F1", 1), "fn": ("FN", -1), "fp": ("FP", -1), "cycles": ("циклов", 0), "score_max": ("макс. оценка цикла", 1), "h_med": ("рост людей, px", 1),
               "wrist_conf": ("точки запястья", 1), "rt_factor": ("× реального времени", 1)}


def matrix(tab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Для тепловой карты: (значения, оценка «хорошо/плохо» в [-1, 1]). Знак: +1 — больше лучше, −1 — больше хуже, 0 — нейтрально (цвет по отклонению от медианы)."""
    cols = [c for c in MATRIX_COLS if c in tab]
    vals = tab.set_index("clip_id")[cols].apply(pd.to_numeric, errors="coerce")
    good = pd.DataFrame(index=vals.index)
    for c in cols:
        v = vals[c]
        sign = MATRIX_COLS[c][1]
        if c == "f1":
            good[c] = (v.fillna(1.0) - 0.5) * 2
        elif c in ("fn", "fp"):
            good[c] = -np.clip(v.fillna(0), 0, 3) / 3
        else:
            med, spread = v.median(), (v.quantile(0.9) - v.quantile(0.1)) or 1.0
            good[c] = np.clip((v - med) / spread * (sign or 0), -1, 1).fillna(0)
    vals.columns = [MATRIX_COLS[c][0] for c in cols]
    good.columns = vals.columns
    return vals, good


def clip_history(clip_id: str, root: Path | None = None) -> pd.DataFrame:
    """F1 и ошибки клипа во всех прогонах журнала (как менялось от конфигурации к конфигурации)."""
    base = root or XP.EXP_DIR
    rows = []
    for d in sorted(base.glob("*/per_clip.csv")) if base.exists() else []:
        try:
            pc = pd.read_csv(d, encoding="utf-8-sig")
            m = json.loads((d.parent / "meta.json").read_text(encoding="utf-8"))
        except Exception:      # noqa: BLE001
            continue
        r = pc[pc.clip_id == clip_id]
        if len(r):
            x = r.iloc[0]
            rows.append(dict(прогон=d.parent.name, имя=m.get("name"), создан=m.get("created"), F1=x.get("f1"), TP=x.get("tp"), FP=x.get("fp"), FN=x.get("fn"), статус=x.get("status"),
                             fps=(m.get("describe") or {}).get("fps"), классификатор=(m.get("describe") or {}).get("cycle_model")))
    return pd.DataFrame(rows)
