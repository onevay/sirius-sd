import pytest

from sd import profiles as PR


def test_profile_cfg_applies_overrides_and_rejects_typos(tmp_path):
    p = PR.Profile("t", config={"pose.weights": "yolo11m-pose", "events.confidence_threshold": 0.7, "cycles.hold_min_sec": 0.4})
    c = p.cfg()
    assert c["pose"]["weights"] == "yolo11m-pose" and p.threshold == 0.7 and c["cycles"]["hold_min_sec"] == 0.4
    with pytest.raises(KeyError):
        PR.Profile("bad", config={"pose.weigths": "x"}).cfg()
    with pytest.raises(KeyError):
        PR.Profile("bad", options={"cycle_bundel": "x"}).opts()
    with pytest.raises(KeyError):
        PR.save(PR.Profile("bad", config={"nope.key": 1}), tmp_path)      # битый профиль не пишется на диск
    assert not list(tmp_path.glob("*.yaml"))


def test_save_load_roundtrip_with_base_overlay(tmp_path):
    p = PR.Profile("exp-1", "описание", base="configs/profiles/fast.yaml", config={"pose.imgsz": 640}, options={"cycle_bundle": "models/cycle/x", "objects": ["a", "b"]})
    f = PR.save(p, tmp_path)
    q = PR.load("exp-1", tmp_path)
    assert q.to_dict() == p.to_dict() and PR.list_profiles(tmp_path) == ["exp-1"]
    assert q.cfg()["pose"]["refine"]["enabled"] is False and q.cfg()["pose"]["imgsz"] == 640        # накладка fast.yaml + точечное значение
    assert PR.delete("exp-1", tmp_path) and not f.exists() and not PR.delete("exp-1", tmp_path)


@pytest.mark.parametrize("name", ["", "a b", "a/b", "../x"])
def test_bad_profile_names(name, tmp_path):
    with pytest.raises(ValueError):
        PR.save(PR.Profile(name), tmp_path)


def test_fingerprint_depends_on_models_not_on_threshold(tmp_path):
    base = PR.Profile("a")
    same = PR.Profile("renamed", config={"events.confidence_threshold": 0.9})
    assert base.fingerprint() == same.fingerprint()                                           # порог применяется после прогона
    assert base.fingerprint() != PR.Profile("b", config={"pose.weights": "yolo11m-pose"}).fingerprint()
    assert base.fingerprint() != PR.Profile("c", options={"vlm_model": "qwen3.5:2b-q4_K_M", "vlm_mode": "grey"}).fingerprint()
    b1 = tmp_path / "bundle"
    b1.mkdir()
    (b1 / "manifest.json").write_text("{}")
    p = PR.Profile("d", options={"cycle_bundle": str(b1)})
    f1 = p.fingerprint()
    (b1 / "manifest.json").write_text('{"x": 1}')
    assert p.fingerprint() != f1                                                              # заменили файлы пакета — кэш инвалидируется


def test_describe_lists_selected_models():
    d = PR.Profile("x", options={"cycle_bundle": "models/cycle/cycle_fast", "vlm_model": "m", "vlm_mode": "grey", "objects": ["det1"]}).describe()
    assert d["cycle_model"] == "cycle_fast" and d["vlm"] == "m (grey)" and d["objects"] == "det1" and d["pose"]
    assert PR.default_profile().describe()["cycle_model"] == "эвристика"
