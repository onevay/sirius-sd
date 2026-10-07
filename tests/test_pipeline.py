import json

import numpy as np
import pandas as pd

from sd import pipeline as P


def test_grey_mask_is_inclusive_on_both_ends():
    s = np.array([0.0, 0.3, 0.55, 0.8, 0.81, 1.0])
    assert list(P.grey_mask(s, (0.3, 0.8))) == [False, True, True, True, False, False]


def test_cascade_uses_full_score_only_where_vlm_answered():
    cheap = np.array([0.1, 0.4, 0.6, 0.9])
    avail = np.array([False, True, True, False])
    full = np.array([np.nan, 0.8, np.nan, np.nan])
    out = P.cascade(cheap, avail, full)
    assert np.allclose(out, [0.1, 0.8, 0.6, 0.9])                      # у третьего ответ VLM есть, но полная оценка NaN → остаётся дешёвая
    assert np.allclose(P.cascade(cheap, avail, None), cheap)           # нет пакета с VLM — итог = дешёвый пакет


def test_bundle_info_reads_manifest_and_hash_changes_with_content(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(dict(kind="cycle_classifier", format=2, created="t", n=5, features=["a", "b"], cv={"auc_ensemble": 0.8})), encoding="utf-8")
    (d / "members.json").write_text("{}", encoding="utf-8")
    a = P.bundle_info(str(d), "cycle")
    assert a["format"] == 2 and a["n"] == 5 and a["features"] == 2 and a["cv"]["auc_ensemble"] == 0.8 and len(a["sha16"]) == 16
    (d / "members.json").write_text('{"x": 1}', encoding="utf-8")
    assert P.bundle_info(str(d), "cycle")["sha16"] != a["sha16"]
    assert P.bundle_info(None, "cycle") is None


def test_list_results_reads_result_json(tmp_path):
    for i, (clip, created) in enumerate((("a", "2026-10-07 10:00:00"), ("b", "2026-10-07 11:00:00"))):
        d = tmp_path / clip / f"0-30s_{i}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(dict(clip=clip, window=[0, 30], created=created, counts=dict(cycles=3, events=1), timing_sec=dict(total=12.5))), encoding="utf-8")
    (tmp_path / "b" / "0-30s_1" / "overlay.mp4").write_bytes(b"x")
    df = P.list_results(tmp_path)
    assert list(df["clip"]) == ["b", "a"]                                   # свежие первыми
    assert list(df.overlay) == [True, False] and list(df.cycles) == [3, 3] and df.window.iloc[0].startswith("0–30")
    assert P.list_results(tmp_path / "нет").empty
