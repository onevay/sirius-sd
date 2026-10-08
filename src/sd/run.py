"""Единая команда запуска (руководство §10.3):

    python -m sd.run --input <папка|файл> --out preds.csv [--profile имя] [--roi roi.json] [--threshold 0.55] [--render out_videos/]

Пишет `preds.csv` в формате организаторов (события выше порога) и рядом `<out>.manifest.json`: профиль моделей, sha256 весов/пакетов (отпечаток), порог, время, коммит.
Порог по умолчанию — из профиля (`events.confidence_threshold`); все события до порога пишутся в `<out>.all.csv` для разбора. Без `--profile` — `configs/default.yaml` с эвристикой цикла.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from . import experiments as XP
from . import preds_check as PC
from . import library as LIB
from . import profiles as PR
from . import roi as ROI
from . import runner as RN


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sd.run", description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="папка с видео или один файл")
    ap.add_argument("--out", required=True, type=Path, help="куда писать preds.csv")
    ap.add_argument("--profile", default=None, help="имя профиля из configs/experiments или путь к YAML (по умолчанию default)")
    ap.add_argument("--roi", default=None, help="roi.json: события вне зоны отбрасываются")
    ap.add_argument("--threshold", type=float, default=None, help="порог confidence (по умолчанию из профиля)")
    ap.add_argument("--render", type=Path, default=None, help="каталог для видео с разметкой")
    ap.add_argument("--max-sec", type=float, default=None, help="обрабатывать только первые N секунд каждого клипа")
    ap.add_argument("--no-cache", action="store_true", help="не брать результаты из кэша")
    a = ap.parse_args(argv)

    src = Path(a.input)
    videos = [src] if src.is_file() else LIB.videos_in([src]) or LIB.videos_in([p for p in src.rglob("*") if p.is_dir()])
    if not videos:
        print(f"в {src} нет видео", file=sys.stderr)
        return 2
    prof = PR.load(a.profile) if a.profile else PR.default_profile()
    th = prof.threshold if a.threshold is None else a.threshold
    zones = ROI.load(a.roi) if a.roi else None
    rec = None
    if a.render:
        a.render.mkdir(parents=True, exist_ok=True)

        def rec(video, start, end, profile):   # прогон с отрисовкой: копия overlay.mp4 в каталог --render
            from . import pipeline as P

            res = P.recognize(video, start, end, profile.cfg(), profile.options_obj(render=True))
            f = Path(res.meta["out_dir"]) / "overlay.mp4"
            if f.exists():
                shutil.copyfile(f, a.render / f"{Path(video).stem}.mp4")
            return res

    t0 = time.perf_counter()
    runs = RN.run_clips(videos, prof, max_sec=a.max_sec, zones=zones, use_cache=not a.no_cache and rec is None, recognize_fn=rec,
                        progress=lambda i, n, m: print(f"[{i}/{n}] {m}", flush=True))
    allev = pd.concat([r.events for r in runs if len(r.events)], ignore_index=True) if any(len(r.events) for r in runs) else pd.DataFrame(columns=RN.EVENT_COLUMNS)
    if len(allev):   # организаторам нужен clip_id в их нумерации — имя файла; camera_id — папка
        stem = {r.clip_id: r.video.stem for r in runs}
        cam = {r.clip_id: r.video.parent.name for r in runs}
        allev = allev.assign(camera_id=allev.clip_id.map(cam), clip_id=allev.clip_id.map(stem))
        allev["event_id"] = [f"{c}_{i:04d}" for c, i in zip(allev.clip_id, allev.groupby("clip_id").cumcount() + 1)]
    out = allev[allev.confidence >= th][RN.EVENT_COLUMNS] if len(allev) else allev
    sizes = {}
    for r in runs:                      # размер кадра нужен проверке «рамка внутри кадра»; не открывается — проверку пропускаем
        try:
            from .video_io import probe

            i = probe(r.video)
            sizes[r.video.stem] = (i.width, i.height)
        except Exception:
            pass
    issues = PC.validate(out, sizes)
    for w in issues:
        print(f"! формат: {w}", file=sys.stderr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    allev.to_csv(a.out.with_suffix(".all.csv"), index=False)
    man = dict(profile=prof.to_dict(), describe=prof.describe(), fingerprint=prof.fingerprint(), confidence_threshold=th, roi=a.roi, input=str(src), clips=len(videos),
               events_total=int(len(allev)), format_issues=issues, events_written=int(len(out)), errors={r.clip_id: r.error for r in runs if r.error}, seconds=round(time.perf_counter() - t0, 1),
               per_clip_seconds={r.clip_id: r.seconds for r in runs}, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), git=XP.git_state())
    a.out.with_suffix(".manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"события: {len(out)} из {len(allev)} (порог {th}); {a.out}")
    return 1 if man["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
