import numpy as np
import pytest

from sd import fusion as FU
from sd import profiles as PR
from sd import solver as SV


@pytest.mark.parametrize("n", ["..", ".", ".hidden"])
def test_dot_names_rejected(n):
    with pytest.raises(ValueError):
        SV._clean_name(n)
    with pytest.raises(ValueError):
        PR._check_name(n)


def test_fusion_validation_and_confident_scores():
    with pytest.raises(ValueError):
        FU.FusionSpec.from_dict({"weights": {"vlm": float("nan")}})
    with pytest.raises(ValueError):
        FU.FusionSpec.from_dict({"weights": {"vlm": 1}, "clip_logit": -1})
    import pandas as pd
    out, _ = FU.fuse(np.array([0.999, 0.001]), pd.DataFrame({"vlm_yesno": [0.5, 0.5]}), {"weights": {"vlm": 1.0}})
    assert out[0] > 0.99 and out[1] < 0.01          # опорное значение сигнала — поправки нет, уверенные оценки не сплющиваются


def test_fusion_is_applied_in_replay_and_only_warned_in_live():
    p = PR.heuristic_profile()
    p.options["fusion"] = {"weights": {"object": 1.0}}
    assert not any(i.code.startswith("fusion_") and i.level == "error" for i in SV.check(p, "replay"))
    assert any(i.code == "fusion_live" and i.level == "warn" for i in SV.check(p, "live"))


def test_photo_features_survive_cycles_without_crops(tmp_path, monkeypatch):
    """Регресс: «'DataFrame' object has no attribute 'path'» — у циклов без кадров рта таблица кадров была пустой и без колонок."""
    import pandas as pd

    from sd import analysis as A
    from sd import photo_clf as PC
    from sd import photo_video as PV

    assert list(PV.frame_table(pd.DataFrame(columns=["video", "tid", "start", "n", "dir"])).columns) == ["video", "tid", "start", "path"]
    assert PV.aggregate(PV.frame_table(pd.DataFrame()), {}).empty
    monkeypatch.setattr(PC, "load_bundle", lambda p: {"manifest": {"backbones": ["clip"]}})
    monkeypatch.setattr(PV, "extract_crops", lambda *a, **k: pd.DataFrame(columns=["video", "tid", "start", "n", "dir"]))
    cyc = pd.DataFrame(dict(video=["v"], tid=[1], start=[1.0], peak_t=[2.0], run=["r"]))
    out = A.photo_all(cyc, tmp_path, out=tmp_path / "p.parquet")
    assert out.empty and list(out.columns) == ["video", "tid", "start"]
