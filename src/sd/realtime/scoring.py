"""Оценка цикла классификатором в потоке и в имитации потока по файлу.

`ClassifierScorer` считает ТЕ ЖЕ признаки, что offline (быстрые: пауза, ритм, положение точек; а при заданном `source_path` — ещё предмет на кропах и фото-модель из исходного файла),
и подаёт их в пакет классификатора без изменений: вектор признаков пакета берётся из его manifest.json. Если пакету нужны признаки, которых в этом режиме нет (предмет/фото в настоящем потоке),
`strict=True` не даёт запуститься: молчаливые пропуски вместо признаков тихо снижали бы качество.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..solver import describe_bundle

KEY = ["video", "tid", "start"]


class FeatureUnavailable(RuntimeError):
    """Классификатору нужны признаки, которые в этом режиме посчитать нельзя."""


class ClassifierScorer:
    name = "classifier"

    def __init__(self, bundle_dir: str | Path, cfg: dict, camera_id: str = "cam", *, objects: tuple | list = (), photo_bundle: str | Path | None = None,
                 source_path: str | Path | None = None, backend: str = "auto", strict: bool = True, fusion: dict | None = None):
        from ..bundle import Bundle
        from ..paths import repo_path

        self.path = repo_path(bundle_dir)
        self.b = Bundle(self.path)
        self.cfg, self.cam = cfg, camera_id
        self.objects, self.photo_bundle = list(objects or []), (repo_path(photo_bundle) if photo_bundle else None)
        self.source, self.backend = (Path(source_path) if source_path else None), backend
        self.info = describe_bundle(self.path)
        req = self.info["requires"]
        self.need_objects = [d for d in req["objects"]]
        self.need_photo = bool(req["photo"])
        if strict:
            if req["vlm"] or req["tube"]:
                raise FeatureUnavailable("классификатор использует признаки VLM/видео-моделей: в потоке они не считаются")
            if (self.need_objects or self.need_photo) and self.source is None:
                raise FeatureUnavailable("классификатору нужны признаки предмета/фото-модели, а исходного файла нет (настоящий поток): используйте классификатор на быстрых признаках "
                                         "или имитацию потока по файлу")
            lack = [d for d in self.need_objects if d not in self.objects]
            if lack:
                raise FeatureUnavailable(f"не подключены детекторы предмета {lack}, которые нужны классификатору")
            if self.need_photo and self.photo_bundle is None:
                raise FeatureUnavailable("классификатору нужна фото-модель (photo_bundle), она не задана")
        self.missing_last: list[str] = []
        from ..fusion import FusionSpec

        self.fusion = FusionSpec.from_dict(fusion)          # те же поправки, что в offline: в replay работают сигналы предмета и фото-модели

    def features(self, cycles: pd.DataFrame, tr, ser: pd.DataFrame) -> pd.DataFrame:
        from ..dataset import build_cycle_table
        from ..pipeline import fast_cycle_features
        from ..pose_feats import pose_rows

        tab = build_cycle_table(Path(self.cam), tr, ser, cycles)
        if tab.empty:
            return tab
        feats = fast_cycle_features(tab, pd.DataFrame(pose_rows(tab.video.iloc[0], tr, self.cfg, ser, cycles)))
        if self.source is not None and (self.need_objects or self.need_photo):
            from .. import analysis as A

            tr.meta["video"] = str(self.source)
            feats = feats.assign(start=feats["start"].round(3))
            cyc = feats[["video", "tid", "start", "end", "mouth_in", "mouth_out", "peak_t", "hold", "d_min", "hand"]].copy()
            cyc["start"] = cyc["start"].round(3)
            cyc["run"], cyc["idx"] = "stream", range(len(cyc))
            if self.need_objects:
                ev = pd.DataFrame(A.evidence_rows(cyc, tr, ser, self.cfg, self.need_objects))
                ev["start"] = ev["start"].round(3)
                feats = feats.assign(start=feats["start"].round(3)).merge(ev, on=KEY, how="left", suffixes=("", "_ev"))
            if self.need_photo:
                import tempfile

                with tempfile.TemporaryDirectory() as td:
                    ph = A.photo_all(cyc, self.photo_bundle, self.backend, out=Path(td) / "photo.parquet", ctx={"stream": (tr, self.cfg, ser)})
                ph["start"] = ph["start"].round(3)
                feats = feats.merge(ph, on=KEY, how="left", suffixes=("", "_ph"))
        return feats

    def score(self, cycles: pd.DataFrame, tr, ser: pd.DataFrame) -> dict:
        feats = self.features(cycles, tr, ser)
        if feats.empty:
            return {}
        self.missing_last = self.b.missing(feats)
        sc = self.b.score(feats)
        if self.fusion.active:
            from ..fusion import fuse

            sc, _ = fuse(sc, feats, self.fusion)
        return {(int(t), round(float(s), 3)): float(v) for t, s, v in zip(feats.tid, feats.start, sc)}
