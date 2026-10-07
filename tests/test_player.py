import json
import re

import numpy as np
import pandas as pd

from sd.ui import player as PL


def _frames():
    cyc = pd.DataFrame(dict(tid=[1, 2], start=[2.0, 8.0], end=[4.5, 9.5], hold=[1.2, 0.8], hand=[0, 1], score=[0.9, np.nan], score_cheap=[0.7, 0.2],
                            photo_p_mean=[0.8, 0.1], vlm_yesno=[np.nan, 0.3], label=["smoke", None]))
    ev = pd.DataFrame(dict(person_track_id=[1], start_sec=[2.0], end_sec=[4.5], confidence=[0.88], label=["smoking_like"]))
    return cyc, ev


def test_build_data_handles_missing_values_and_collects_tracks():
    cyc, ev = _frames()
    d = PL.build_data(cyc, ev, 12.0, 15.0)
    assert d["tids"] == [1, 2] and len(d["cycles"]) == 2 and len(d["events"]) == 1
    assert d["cycles"][1]["score"] is None and d["cycles"][0]["vlm"] is None and d["cycles"][0]["label"] == "smoke" and d["cycles"][1]["label"] == ""
    assert d["events"][0]["conf"] == 0.88
    json.dumps(d)                                                  # сериализуется без NaN (иначе JS не разберёт)
    assert "NaN" not in json.dumps(d)


def test_build_data_empty_inputs_ok():
    d = PL.build_data(pd.DataFrame(), pd.DataFrame(), 5.0, 0.0)
    assert d["cycles"] == [] and d["events"] == [] and d["tids"] == [] and d["fps"] == 15.0


def test_player_html_embeds_video_and_data_or_warns(tmp_path):
    cyc, ev = _frames()
    d = PL.build_data(cyc, ev, 12.0, 15.0, offset=36.0)
    vid = tmp_path / "x.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    html, warn = PL.player_html(vid, d)
    assert warn is None and 'src="data:video/mp4;base64,' in html and "__DATA__" not in html and "__SRC__" not in html and "__H__" not in html
    data = json.loads(re.search(r"const D=(\{.*?\});const v=", html, re.S).group(1))
    assert data["cycles"][0]["start"] == 2.0 and data["offset"] == 36.0
    html2, warn2 = PL.player_html(None, d)
    assert warn2 is None and "data:video" not in html2
    old = PL.MAX_EMBED_MB
    try:
        PL.MAX_EMBED_MB = 0.00001
        _, warn3 = PL.player_html(vid, d)
        assert warn3 and "не вшито" in warn3
    finally:
        PL.MAX_EMBED_MB = old
