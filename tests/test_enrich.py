"""Кэш признаков предмета: отпечаток параметров и весов, покрытие циклов, защита чужого кэша."""
import copy

import pandas as pd

from sd import enrich as EN


def _cfg():
    return {"evidence": {"imgsz": 320, "conf": 0.15, "crop_scale": 1.2, "detectors": ["a"]}}


def _cycles():
    return pd.DataFrame([dict(video="v", tid=1, start=1.0), dict(video="v", tid=1, start=5.0)])


def _rows(starts=(1.0, 5.0)):
    return pd.DataFrame([dict(video="v", tid=1, start=s, obj_x_max_conf=0.5 + i / 10, obj_x_hit_frames=i, n_det_frames=5) for i, s in enumerate(starts)])


def test_cache_roundtrip_and_invalidation(tmp_path, monkeypatch):
    monkeypatch.setattr(EN, "evidence_signature", lambda cfg, dets: f"{cfg['evidence']['imgsz']}-{'-'.join(sorted(dets))}")
    d = tmp_path / "v"
    assert EN.cached_evidence(_cycles(), _cfg(), ["x"], d) is None                                # кэша нет
    EN.save_evidence(_rows(), _cfg(), ["x"], d)
    got = EN.cached_evidence(_cycles(), _cfg(), ["x"], d)
    assert got is not None and list(got.obj_x_hit_frames) == [0, 1]
    c2 = copy.deepcopy(_cfg()); c2["evidence"]["imgsz"] = 640
    assert EN.cached_evidence(_cycles(), c2, ["x"], d) is None                                    # другой размер входа — другой отпечаток
    assert EN.cached_evidence(pd.concat([_cycles(), pd.DataFrame([dict(video="v", tid=2, start=9.0)])]), _cfg(), ["x"], d) is None   # цикл не покрыт
    EN.save_evidence(_rows((9.0,)), c2, ["x"], d)                                                 # чужой отпечаток не затирает кэш
    assert EN.cached_evidence(_cycles(), _cfg(), ["x"], d) is not None
    EN.save_evidence(_rows((9.0,)).assign(tid=2), _cfg(), ["x"], d)                               # свой отпечаток дополняет
    both = pd.concat([_cycles(), pd.DataFrame([dict(video="v", tid=2, start=9.0)])])
    assert EN.cached_evidence(both, _cfg(), ["x"], d) is not None


def test_missing_is_key_based():
    done = _rows((1.0,))
    assert list(EN.missing(_cycles(), done).start) == [5.0]


def test_photo_cache_signature(tmp_path):
    import json

    b = tmp_path / "bundle"
    b.mkdir()
    (b / "manifest.json").write_text(json.dumps({"v": 1}), encoding="utf-8")
    (b / "logreg.json").write_text(json.dumps({"w": [1, 2]}), encoding="utf-8")
    d = tmp_path / "clip"
    d.mkdir()
    rows = pd.DataFrame([dict(video="v", tid=1, start=1.0, photo_zsd_mean=0.1, photo_zsd_max=0.2), dict(video="v", tid=1, start=5.0, photo_zsd_mean=0.3, photo_zsd_max=0.4)])
    rows.to_parquet(d / "photo_cycles.parquet", index=False)
    assert EN.cached_photo(_cycles(), b, d) is None                      # без отпечатка кэш не используется
    EN._stamp_photo(d, b)
    got = EN.cached_photo(_cycles(), b, d)
    assert got is not None and list(got.photo_zsd_max) == [0.2, 0.4]
    (b / "logreg.json").write_text(json.dumps({"w": [9, 9]}), encoding="utf-8")   # другие веса фото-модели → кэш недействителен
    assert EN.cached_photo(_cycles(), b, d) is None


def test_attach_replaces_stale_feature_columns(tmp_path):
    """В таблице уже есть obj_*-колонки (из analysis/ другого запуска) — присоединение из кэша не должно оставлять их пустыми (раньше ломало слияние)."""
    tab = pd.DataFrame([dict(video="v", tid=1, start=1.0, hold=1.0, obj_x_max_conf=float("nan")), dict(video="w", tid=1, start=2.0, hold=2.0, obj_x_max_conf=0.9)])
    d = tmp_path / "v"
    d.mkdir()
    pd.DataFrame([dict(video="v", tid=1, start=1.0, n_det_frames=5, obj_a_max_conf=0.7, obj_a_hit_frames=3, obj_a_hit=True)]).to_parquet(d / "evidence_cycles.parquet", index=False)
    out = EN.attach(tab, ["v", "w"], root=tmp_path, vlm=False)
    v = out[out.video == "v"].iloc[0]
    assert v.obj_a_max_conf == 0.7 and v.obj_any_max_conf == 0.7 and "obj_x_max_conf" not in out.columns
    assert out[out.video == "w"].obj_any_max_conf.isna().all() if "obj_any_max_conf" in out else True       # клип без кэша остаётся без признаков


def test_evidence_cache_keeps_a_slot_per_configuration(tmp_path):
    """Кэш признаков предмета не принадлежит «первому записавшему»: другая конфигурация (детекторы, параметры) получает соседний слот и не вытесняет чужой."""
    import pandas as pd

    from sd import enrich as EN
    from sd.config import load_config

    cfg_a = load_config(None)
    cfg_b = load_config(None, ["evidence.conf=0.4"])
    rows = lambda v: pd.DataFrame(dict(video=["c"], tid=[1], start=[1.0], obj_det_x_max_conf=[v]))      # noqa: E731
    cyc = pd.DataFrame(dict(video=["c"], tid=[1], start=[1.0]))
    EN.save_evidence(rows(0.1), cfg_a, ["det_x"], tmp_path)
    EN.save_evidence(rows(0.9), cfg_b, ["det_x"], tmp_path)
    assert EN.cached_evidence(cyc, cfg_a, ["det_x"], tmp_path).obj_det_x_max_conf.iloc[0] == 0.1
    assert EN.cached_evidence(cyc, cfg_b, ["det_x"], tmp_path).obj_det_x_max_conf.iloc[0] == 0.9
    assert any(p.name.startswith("alt_") for p in tmp_path.iterdir())
