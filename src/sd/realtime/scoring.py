"""Оценка цикла классификатором в потоке и в имитации потока по файлу.

`ClassifierScorer` считает ТЕ ЖЕ признаки, что offline (быстрые: пауза, ритм, положение точек; а при заданном `source_path` — ещё предмет на кропах и фото-модель из исходного файла),
и подаёт их в пакет классификатора без изменений: вектор признаков пакета берётся из его manifest.json. Если пакету нужны признаки, которых в этом режиме нет (предмет/фото в настоящем потоке),
`strict=True` не даёт запуститься: молчаливые пропуски вместо признаков тихо снижали бы качество.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..feature_auc import _derived
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
        # `obj_any_*` не называет детектор: тогда нужны ВСЕ детекторы профиля (так же считает offline); без этого признаки предмета в потоке не считались вовсе
        self.need_objects = list(req["objects"]) or (list(self.objects) if req.get("objects_any") else [])
        self.need_photo = bool(req["photo"])
        if strict:
            if req["vlm"] or req["tube"]:
                raise FeatureUnavailable("классификатор использует признаки VLM/видео-моделей: в потоке они не считаются")
            if req.get("objects_any") and not self.objects:
                raise FeatureUnavailable("классификатор использует признаки предмета (obj_any_*), а детекторы предмета в профиле не заданы")
            if (self.need_objects or self.need_photo) and self.source is None:
                raise FeatureUnavailable("классификатору нужны признаки предмета/фото-модели, а исходного файла нет (настоящий поток): используйте классификатор на быстрых признаках "
                                         "или имитацию потока по файлу")
            lack = [d for d in self.need_objects if d not in self.objects]
            if lack:
                raise FeatureUnavailable(f"не подключены детекторы предмета {lack}, которые нужны классификатору")
            if self.need_photo and self.photo_bundle is None:
                raise FeatureUnavailable("классификатору нужна фото-модель (photo_bundle), она не задана")
        self.missing_last: list[str] = []
        self._ev, self._ph = pd.DataFrame(), pd.DataFrame()      # признаки предмета/фото по уже обработанным циклам
        self._done: set[tuple[int, float]] = set()
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
            feats = self._with_extra(feats, tr, ser)
        return feats

    def _to_source_frames(self, tr, ser: pd.DataFrame, cyc: pd.DataFrame):
        """Кадры в потоке нумеруются подряд (0, 1, 2 …), а детекторам предмета и кропам рта нужны номера кадров ФАЙЛА: без пересчёта они брали не те кадры (или ни одного — признаки
        предмета были нулями при любой сигарете). Соответствие — по времени: ближайший кадр файла, только в окне новых циклов (вся запись не просматривается)."""
        from dataclasses import replace

        from ..video_io import window_frame_index

        ft = tr.frame_t
        t0, t1 = float(cyc["start"].min()) - 3.0, float(cyc["end"].max()) + 3.0
        try:
            idx, ts = window_frame_index(self.source, max(t0, 0.0), t1)
        except OSError:          # файла нет/не открывается: признаки предмета сами сообщат об ошибке (громко), а не получат «пустые» кадры
            return replace(tr, meta={**tr.meta, "video": str(self.source)}), ser
        if not len(ft):
            return tr, ser
        tt = ft["t"].to_numpy(float)
        if not len(ts):          # в окне нет кадров файла: любые номера потока были бы чужими кадрами — уводим их за пределы файла
            m = {int(f): int(f) + 10 ** 9 for f in ft["frame"].astype(int)}
            return (replace(tr, df=tr.df.assign(frame=tr.df["frame"].map(m)), frame_t=ft.assign(frame=ft["frame"].map(m)), meta={**tr.meta, "video": str(self.source)}),
                    ser.assign(frame=ser["frame"].map(m)) if "frame" in ser.columns else ser)
        pos = np.clip(np.searchsorted(ts, tt), 1, len(ts) - 1) if len(ts) > 1 else np.zeros(len(tt), int)
        pos = np.where(np.abs(ts[pos] - tt) < np.abs(ts[np.maximum(pos - 1, 0)] - tt), pos, np.maximum(pos - 1, 0))
        ok = np.abs(ts[pos] - tt) < 0.5
        m = {int(f): (int(idx[p]) if o else int(f) + 10 ** 9) for f, p, o in zip(ft["frame"].astype(int), pos, ok)}      # вне окна — номер, которого нет в файле
        meta = {**tr.meta, "video": str(self.source)}
        tr2 = replace(tr, df=tr.df.assign(frame=tr.df["frame"].map(m)), frame_t=ft.assign(frame=ft["frame"].map(m)), meta=meta)
        ser2 = ser.assign(frame=ser["frame"].map(m)) if "frame" in ser.columns else ser
        return tr2, ser2

    def _with_extra(self, feats: pd.DataFrame, tr, ser: pd.DataFrame) -> pd.DataFrame:
        """Признаки предмета и фото-модели. Считаются ОДИН раз на цикл и запоминаются: движок оценивает буфер каждую секунду, и без кэша каждая оценка заново прогоняла бы детекторы по
        всем циклам буфера — время росло бы с каждым циклом и «реальное время» превращалось в минуты на секунду видео. Пустой результат (нет кропов) — не ошибка: признаки остаются пропусками."""
        from .. import analysis as A

        feats = feats.assign(start=pd.to_numeric(feats["start"]).round(3))
        key = list(zip(feats["tid"].astype(int), feats["start"]))
        new = [k not in self._done for k in key]
        if any(new):
            tr, ser = self._to_source_frames(tr, ser, feats.loc[new])
            cyc = feats.loc[new, ["video", "tid", "start", "end", "mouth_in", "mouth_out", "peak_t", "hold", "d_min", "hand"]].copy()
            cyc["run"], cyc["idx"] = "stream", range(len(cyc))
            if self.need_objects:
                ev = pd.DataFrame(A.evidence_rows(cyc, tr, ser, self.cfg, self.need_objects))
                if len(ev):
                    ev["start"] = pd.to_numeric(ev["start"]).round(3)
                    self._ev = pd.concat([self._ev, ev], ignore_index=True)
            if self.need_photo:
                import tempfile

                with tempfile.TemporaryDirectory() as td:
                    ph = A.photo_all(cyc, self.photo_bundle, self.backend, out=Path(td) / "photo.parquet", ctx={"stream": (tr, self.cfg, ser)},
                                     only_zsd=bool(self.info["requires"].get("photo_only_zsd")))
                if len(ph):
                    ph["start"] = pd.to_numeric(ph["start"]).round(3)
                    self._ph = pd.concat([self._ph, ph], ignore_index=True)
            self._done.update(k for k, n in zip(key, new) if n)
        for extra, suffix in ((self._ev, "_ev"), (self._ph, "_ph")):
            if len(extra):
                feats = feats.merge(extra.drop_duplicates(KEY), on=KEY, how="left", suffixes=("", suffix))
        # производные признаки (`obj_any_*` — максимум по детекторам) считаются тем же кодом, что offline (`feature_auc._derived`): без них пакет получал пропуски вместо
        # признака предмета, оценки циклов сваливались к ≈0.3 и тревог в потоке не было вовсе, хотя offline та же модель даёт F1 0.82
        return _derived(feats)

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
