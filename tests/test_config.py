import pytest

from sd.config import deep_merge, load_config, section_hash, set_by_path, stable_hash


def test_default_has_expected_sections(cfg):
    for k in ("video", "pose", "tracking", "features", "cycles", "events", "evidence", "tube", "vlm", "render"):
        assert k in cfg
    assert cfg.cycles.th_in < cfg.cycles.th_out
    assert cfg.pose.refine.method.startswith("rtmpose")


def test_cli_override_and_typo_detection():
    c = load_config(overrides=["cycles.th_in=0.8", "pose.refine.enabled=false", "tube.prompts=[a, b]"])
    assert c.cycles.th_in == 0.8 and c.pose.refine.enabled is False and c.tube.prompts == ["a", "b"]
    with pytest.raises(KeyError):
        load_config(overrides=["cycles.th_inn=0.8"])          # опечатка в имени параметра
    with pytest.raises(ValueError):
        load_config(overrides=["cycles.th_in"])               # нет '='


def test_profile_overlay_merges_deeply(tmp_path):
    p = tmp_path / "prof.yaml"
    p.write_text("pose:\n  refine:\n    enabled: false\ncycles:\n  th_in: 0.7\n", encoding="utf-8")
    c = load_config(p)
    assert c.pose.refine.enabled is False and c.pose.refine.method.startswith("rtmpose")   # соседние ключи сохранились
    assert c.cycles.th_in == 0.7 and c.cycles.th_out == load_config().cycles.th_out


def test_profile_with_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("pose:\n  imgsize: 640\n", encoding="utf-8")          # опечатка: imgsz
    with pytest.raises(KeyError):
        load_config(p)


def test_shipped_profiles_load():
    from sd.paths import CONFIGS

    for name in ("fast.yaml", "cpu_torch.yaml"):
        c = load_config(CONFIGS / "profiles" / name)
        assert c.pose.weights == "yolo26n-pose"
    assert load_config(CONFIGS / "profiles" / "fast.yaml").pose.refine.enabled is False
    assert load_config(CONFIGS / "profiles" / "cpu_torch.yaml").pose.runtime == "torch"


def test_hashes_are_stable_and_sensitive():
    base = load_config()
    assert section_hash(base, "cycles") == section_hash(load_config(), "cycles")
    c2 = load_config(overrides=["cycles.th_in=0.9"])
    assert section_hash(base, "cycles") != section_hash(c2, "cycles")
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_default_thresholds_are_in_calibrated_plateau():
    """default.yaml хранит откалиброванные пороги (центр плато recall по визуальным интервалам: th_in 0.50–0.80, th_out 0.70–1.15)."""
    c = load_config()
    assert 0.50 <= c.cycles.th_in <= 0.80 and 0.70 <= c.cycles.th_out <= 1.15 and c.cycles.th_out > c.cycles.th_in
