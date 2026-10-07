"""Пакетная визуальная проверка циклов: монтажи кадров по всем запускам с текущими порогами из configs/default.yaml.

Результат: outputs/analysis/review/<video>_cycles_NN.png (по 8 циклов на изображение) и <video>_cycles.csv — таблица в том же порядке
(индекс `k` в подписи изображения = номер строки). По этим монтажам циклы размечаются глазами: затяжка / питьё / телефон / касание / другое.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import pandas as pd

from . import stages
from .calibrate import cfg_current_for_run, list_runs
from .paths import OUTPUTS
from .review import cycle_montage
from .tracks import Tracks


def make_all(out_dir: Path = OUTPUTS / "analysis" / "review", per_video: int = 8, pose_sig: str | None = "rtmpose", progress=None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    runs = list_runs(pose_sig=pose_sig)
    for i, (vname, rd) in enumerate(runs):
        tr = Tracks.load(rd / "pose")
        cfg = cfg_current_for_run(rd)
        ser = stages.stage_features(tr, cfg, rd)
        cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
        summ = tr.summary(cfg["video"]["min_person_height_px"])
        cyc = cyc[cyc.tid.isin(summ[~summ.ignore_small].tid)]
        if cyc.empty:
            rows.append(dict(video=vname, cycles=0, montages=0))
            continue
        sub = cyc.sort_values("start").reset_index(drop=True)
        if len(sub) > per_video:   # равномерная выборка по времени, а не первые N
            sub = sub.iloc[[int(round(x)) for x in pd.Series(range(per_video)).mul((len(sub) - 1) / max(per_video - 1, 1))]].reset_index(drop=True)
        sub.insert(0, "k", range(len(sub)))
        sub.insert(1, "video", vname)
        sub.to_csv(out_dir / f"{vname}_cycles.csv", index=False)
        paths = cycle_montage(tr.meta["video"], tr, sub, out_dir / f"{vname}_cycles", series=ser, title=vname)
        rows.append(dict(video=vname, cycles=len(cyc), shown=len(sub), montages=len(paths)))
        if progress:
            progress(i + 1, len(runs))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "index.csv", index=False)
    return df
