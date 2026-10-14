"""Каталог данных для Dev Studio: что за видео есть, что из них размечено, где разметка противоречит сама себе и чего не хватает.

По каждому видео: папка, id клипа, размер, свойства (длительность, к/с, разрешение — `video_io.probe`, кэш outputs/catalog/probe.json по пути, размеру и
времени изменения), слабая метка по имени папки, состояние эталона событий (просмотрен, POSITIVE/NEGATIVE/IGNORE), метки жестов (сколько окон, сколько затяжек),
есть ли готовые запуски распознавания, отпечаток для поиска дублей (размер + sha1 первого мегабайта).

`issues()` — проверки, которые раньше всплывали только после оценки:
    weak_conflict   папка «лжекурение», а в эталоне есть POSITIVE (случай smoking_noyabrsk_2/3) — слабая метка неверна
    smoke_no_pos    папка «курение», клип просмотрен, но POSITIVE нет — проверить (сигарета не видна? не тот файл?)
    gestures_no_gt  затяжки размечены как жесты, а эталона событий для клипа нет — `sd gt-from-gestures` или разметка
    unlabeled       клип не размечен (в оценку не попадёт)
    duplicate       одинаковые файлы под разными именами (утечка между обучением и проверкой)
    unreadable      видео не открывается
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import gt as GT
from . import library as LIB
from .paths import OUTPUTS, portable, video_id, weak_label

PROBE_CACHE = OUTPUTS / "catalog" / "probe.json"
ISSUE_TEXT = {
    "weak_conflict": ("warning", "папка «лжекурение», а в эталоне есть курение — слабая метка папки неверна (переложите файл или учитывайте только эталон)"),
    "smoke_no_pos": ("info", "папка «курение», клип размечен, но эпизодов курения нет — проверьте (сигарета не видна? курение вне кадра?)"),
    "gestures_no_gt": ("warning", "затяжки размечены как жесты, а эталона событий нет — постройте эталон из жестов или разметьте события"),
    "unlabeled": ("info", "клип не размечен — в оценку по эталону не попадёт"),
    "duplicate": ("error", "одинаковые файлы под разными именами — один и тот же ролик может попасть и в обучение, и в проверку"),
    "unreadable": ("error", "видео не открывается или длительность неизвестна"),
}


def _fast_hash(p: Path, n: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return f"{p.stat().st_size}:{h.hexdigest()[:12]}"


def _probe_cache() -> dict:
    try:
        return json.loads(PROBE_CACHE.read_text(encoding="utf-8")) if PROBE_CACHE.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def probe_cached(paths: list[Path], progress=None) -> dict[str, dict]:
    """Свойства видео (длительность, к/с, ширина, высота, кодек) с кэшем на диске. Ключ — переносимый путь; запись устаревает при смене размера/времени файла."""
    from .video_io import probe

    cache = _probe_cache()
    out, changed = {}, False
    for i, p in enumerate(paths):
        key = portable(p)
        st = p.stat()
        stamp = f"{st.st_size}:{st.st_mtime_ns}"
        rec = cache.get(key)
        if not rec or rec.get("stamp") != stamp:
            try:
                info = probe(p)
                rec = dict(stamp=stamp, duration=round(float(info.duration), 2), fps=round(float(info.fps), 2), width=int(info.width), height=int(info.height),
                           codec=getattr(info, "codec", None), ok=bool(info.duration and info.duration > 0))
            except Exception as e:      # noqa: BLE001 — битый файл не роняет каталог
                rec = dict(stamp=stamp, ok=False, error=f"{type(e).__name__}: {e}"[:200])
            cache[key] = rec
            changed = True
        out[key] = rec
        if progress:
            progress(i + 1, len(paths), p.name)
    if changed:
        PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PROBE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return out


def videos_table(dirs: list[str | Path] | None = None, with_probe: bool = False, gestures: pd.DataFrame | None = None, gt: pd.DataFrame | None = None,
                 progress=None) -> pd.DataFrame:
    """Таблица видео выбранных папок (по умолчанию — всех найденных). `with_probe` — прочитать свойства видео (медленно при первом вызове, дальше из кэша)."""
    from .paths import OUTPUTS as OUT

    if dirs is None:
        d = LIB.discover_dirs()
        dirs = d.path.tolist() if len(d) else []
    vids = LIB.videos_in(dirs)
    gt = GT.load() if gt is None else gt
    if gestures is None:
        from . import start_eval as SE

        gestures = pd.read_csv(SE.GT_CSV, encoding="utf-8-sig") if Path(SE.GT_CSV).exists() else pd.DataFrame(columns=["video", "label"])
    g_all = gestures.groupby("video").size() if len(gestures) else pd.Series(dtype=int)
    g_smoke = gestures[gestures.label == "smoke"].groupby("video").size() if len(gestures) else pd.Series(dtype=int)
    status = GT.clip_status(gt, [video_id(v) for v in vids]).set_index("clip_id") if vids else pd.DataFrame()
    probes = probe_cached(vids, progress) if with_probe else {}
    rows = []
    for v in vids:
        cid = video_id(v)
        s = status.loc[cid] if cid in status.index else None
        pr = probes.get(portable(v), {})
        rows.append(dict(clip_id=cid, folder=v.parent.name, file=v.name, path=str(v), size_mb=round(v.stat().st_size / 1e6, 1), weak=weak_label(v) or "",
                         reviewed=bool(s.reviewed) if s is not None else False, pos=int(s.positive) if s is not None else 0, neg=int(s.negative) if s is not None else 0,
                         ign=int(s.ignore) if s is not None else 0, gestures=int(g_all.get(cid, 0)), smokes=int(g_smoke.get(cid, 0)),
                         runs=int(sum(1 for x in (OUT / "recognize" / cid).glob("0-end*") if (x / "result.json").exists())) if (OUT / "recognize" / cid).exists() else 0,
                         duration=pr.get("duration"), fps=pr.get("fps"), width=pr.get("width"), height=pr.get("height"), readable=pr.get("ok") if pr else None,
                         dup_key=_fast_hash(v)))
    return pd.DataFrame(rows)


def issues(tab: pd.DataFrame) -> pd.DataFrame:
    """Проблемы данных (см. docstring модуля): уровень, клип, код, текст."""
    out = []

    def add(r, code, detail=""):
        lvl, txt = ISSUE_TEXT[code]
        out.append(dict(level=lvl, clip_id=r.clip_id, folder=r.folder, code=code, text=txt, detail=detail))

    if tab.empty:
        return pd.DataFrame(columns=["level", "clip_id", "folder", "code", "text", "detail"])
    dups = tab[tab.duplicated("dup_key", keep=False)]
    groups = dups.groupby("dup_key").clip_id.apply(list).to_dict()
    for r in tab.itertuples():
        if r.weak == "fake" and r.pos > 0:
            add(r, "weak_conflict", f"POSITIVE в эталоне: {r.pos}")
        if r.weak == "smoking" and r.reviewed and r.pos == 0:
            add(r, "smoke_no_pos", f"IGNORE {r.ign}, NEGATIVE {r.neg}")
        if r.smokes > 0 and not r.reviewed:
            add(r, "gestures_no_gt", f"затяжек в метках жестов: {r.smokes}")
        if not r.reviewed:
            add(r, "unlabeled")
        if r.dup_key in groups:
            add(r, "duplicate", "совпадает с: " + ", ".join(c for c in groups[r.dup_key] if c != r.clip_id))
        if r.readable is False:
            add(r, "unreadable")
    order = {"error": 0, "warning": 1, "info": 2}
    df = pd.DataFrame(out, columns=["level", "clip_id", "folder", "code", "text", "detail"])
    return df.sort_values("level", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


def coverage(tab: pd.DataFrame) -> pd.DataFrame:
    """Сводка по папкам: видео, минут, размечено, эпизодов курения, клипов с метками жестов, с готовыми запусками."""
    if tab.empty:
        return pd.DataFrame()
    g = tab.groupby("folder").agg(видео=("clip_id", "count"), минут=("duration", lambda s: round(s.fillna(0).sum() / 60, 1)), размечено=("reviewed", "sum"),
                                  эпизодов_курения=("pos", "sum"), с_жестами=("gestures", lambda s: int((s > 0).sum())), с_запусками=("runs", lambda s: int((s > 0).sum())),
                                  метка_папки=("weak", "first")).reset_index().rename(columns={"folder": "папка"})
    return g


def feedback_status() -> dict:
    """Решения оператора (хранилище мониторинга), которые ещё не перенесены в эталон: сколько подтверждённых и ложных, сколько уже перенесено."""
    try:
        from .realtime.store import open_store

        df = open_store().list_alerts(status=["confirmed", "false"], limit=100000)
    except Exception as e:          # noqa: BLE001 — базы мониторинга может не быть
        return dict(ok=False, error=str(e), confirmed=0, false=0, exported=0)
    gt = GT.load()
    done = set(gt.note.str.extract(r"alert:(\d+)")[0].dropna().astype(int)) if len(gt) else set()
    if df is None or not len(df):
        return dict(ok=True, confirmed=0, false=0, exported=0, pending=0)
    pend = df[~df.id.isin(done)]
    return dict(ok=True, confirmed=int((df.status == "confirmed").sum()), false=int((df.status == "false").sum()), exported=int(df.id.isin(done).sum()), pending=int(len(pend)))
