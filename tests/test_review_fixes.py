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


def test_fusion_blocked_in_stream():
    p = PR.heuristic_profile()
    p.options["fusion"] = {"weights": {"vlm": 1.0}}
    assert not any(i.code == "fusion_stream" for i in SV.check(p, "offline"))
