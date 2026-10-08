import pytest

from sd.config import load_config
from sd.ui.profile_editor import _flatten, _parse


def test_flatten_covers_default_config_scalars():
    flat = _flatten(load_config(None).to_dict())
    assert flat["pose.refine.device"] == "cpu" and flat["evidence.device"] == "cpu"
    assert flat["cycles.th_in"] == 0.65 and flat["pose.imgsz"] == 960


def test_parse_keeps_type_of_default():
    assert _parse("0.7", 0.65) == 0.7 and _parse("1280", 960) == 1280 and _parse("true", False) is True and _parse("cuda:0", "cpu") == "cuda:0"
    with pytest.raises(ValueError):
        _parse("maybe", True)
    with pytest.raises(ValueError):
        _parse("abc", 1.0)
