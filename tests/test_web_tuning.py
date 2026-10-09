"""Веб: проверка и подгонка моделей — оценка на папках, анализ порога по OOF, спецификация/диагностика/обучение классификатора."""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from rt_helpers import make_bundle
from sd import classifier as C
from sd import cycle_models as CM
from sd import experiments as XP
from sd import gt as GT
from sd import profiles as PR
from sd import runner as R
from sd.paths import video_id
from sd.realtime.store import AlertStore
from sd.web.api import WebApp
from sd.web.server import serve
from test_web_admin import mkvid

FAST = ["hold", "d_min", "mouth_dur", "p_wm_dist"]


def synth_table(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(int)
    cols = {f: rng.normal(size=n) + y * 1.2 for f in dict.fromkeys(CM.SETS["obj_hold_zsd"])}
    return pd.DataFrame(dict(video=[f"v{i % 12}" for i in range(n)], tid=1, start=np.arange(n, dtype=float), end=np.arange(n) + 1.0, peak_t=np.arange(n) + .5, y=y, **cols))


@pytest.fixture()
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "EXPERIMENTS_DIR", tmp_path / "profiles")
    monkeypatch.setattr(XP, "EXP_DIR", tmp_path / "exp")
    monkeypatch.setattr(C, "load_table", lambda enriched=True: synth_table())
    monkeypatch.setattr(C, "DIR", tmp_path / "specs")
    import sd.web.tuning as TU

    monkeypatch.setattr(TU, "MODELS", tmp_path / "models")
    monkeypatch.setattr(TU, "repo_path", lambda p: tmp_path / p)
    app = WebApp(AlertStore(tmp_path / "m.db"), tmp_path / "streams", tmp_path / "data", tmp_path / "out")
    srv = serve(app, port=0, block=False)
    yield app, f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()


def call(url, obj=None):
    req = urllib.request.Request(url, data=None if obj is None else json.dumps(obj).encode(), method="GET" if obj is None else "POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_op(base, oid, limit=600):
    for _ in range(limit):
        r = call(f"{base}/api/op/{oid}")[1]
        if r["state"] != "running":
            return r
        time.sleep(0.1)
    raise AssertionError("операция не завершилась")


def test_table_spec_validate_diagnose_train_and_attach(web):
    app, base, tmp = web
    PR.save(PR.Profile("p1", options={"allow_heuristic": True}))
    info = call(base + "/api/tune/table")[1]
    assert info["ok"] and info["n_labeled"] == 120 and info["n_pos"] == 60 and info["n_videos"] == 12 and "obj_hold_zsd" in info["sets"] and info["columns"][0]["filled"] == 1.0
    spec = call(base + "/api/tune/spec/default", {"set": "obj_hold_zsd", "kinds": ["lr", "gb"]})[1]
    assert spec["members"] and spec["problems"] == [] and call(base + "/api/tune/spec/default", {"set": "нет"})[0] == 400
    spec["cv"] = dict(n_splits=2, repeats=1, by_scene=False, seed=0)
    bad = dict(spec, members={"lr": dict(kind="неизвестно", features=["hold"])})
    assert call(base + "/api/tune/spec/validate", {"spec": bad})[1]["problems"]
    spec["name"] = "web_spec"
    assert call(base + "/api/tune/spec/save", {"spec": spec})[1]["file"] and call(base + "/api/tune/spec/web_spec")[1]["name"] == "web_spec"
    d = wait_op(base, call(base + "/api/tune/diagnose", {"spec": spec, "n_perm": 1, "importance": False})[1]["id"])
    assert d["state"] == "done", d["error"]
    assert d["result"]["n"] == 120 and d["result"]["members"] and d["result"]["verdict"]
    t = wait_op(base, call(base + "/api/tune/train", {"spec": spec, "name": "web_cycle", "attach": "p1"})[1]["id"])
    assert t["state"] == "done", t["error"]
    assert t["result"]["bundle"] == "models/cycle/web_cycle" and (tmp / "models" / "cycle" / "web_cycle" / "manifest.json").exists()
    assert PR.load("p1").options["cycle_bundle"] == "models/cycle/web_cycle"
    assert call(base + "/api/tune/train", {"spec": spec, "name": "../x"})[0] == 400


def test_bundle_threshold_analysis_from_oof(web):
    app, base, tmp = web
    b = make_bundle(tmp / "models" / "cycle" / "oofb", FAST)
    rng = np.random.default_rng(1)
    y = (np.arange(200) % 2).astype(int)
    pd.DataFrame(dict(y=y, group=np.arange(200) % 10, oof_a=np.clip(0.5 + (y - .5) * .5 + rng.normal(0, .15, 200), 0, 1), oof_ensemble=np.clip(0.5 + (y - .5) * .6 + rng.normal(0, .12, 200), 0, 1))).to_csv(b / "oof.csv", index=False)
    r = call(base + "/api/tune/bundle/oofb")[1]
    assert r["info"]["n_features"] == 4 and len(r["thresholds"]) == 19 and r["n"] == 200 and r["n_pos"] == 100
    best = r["best"]
    assert 0.3 <= best["suggested"] <= 0.7 and best["f1"] > 0.8 and best["plateau"][0] <= best["suggested"] <= best["plateau"][1]
    assert {m["name"] for m in r["members"]} == {"oof_a", "oof_ensemble"} and sum(r["hist"]["pos"]) == 100
    assert call(base + "/api/tune/bundle/..")[0] in (400, 404, 500)
    (make_bundle(tmp / "models" / "cycle" / "noof", FAST) / "oof.csv").unlink()
    assert "oof.csv" in call(base + "/api/tune/bundle/noof")[1]["note"]


def test_evaluate_profile_on_folder_and_read_back(web, monkeypatch):
    from sd import solver as SV

    monkeypatch.setattr(SV, "_device_issues", lambda p: [])
    app, base, tmp = web
    d = tmp / "data" / "курение"
    mkvid(d / "s1.mp4", n=20)
    f = tmp / "gt.csv"
    GT.add(video_id(d / "s1.mp4"), 0.2, 1.6, "POSITIVE", person=1, box=(0, 0, 100, 200), path=f)
    monkeypatch.setattr(GT, "default_path", lambda: f)

    def fake(video, start, end, profile):
        rows = [dict(camera_id="c", clip_id=video.stem, event_id="e", start_sec=0.3, end_sec=1.5, confidence=0.9, label="smoking_like", person_track_id=1, peak_sec=1, x1=0, y1=0, x2=100, y2=200)]
        return SimpleNamespace(events=pd.DataFrame(rows, columns=R.EVENT_COLUMNS), meta=dict(run_dir=None, out_dir=None, counts=dict(cycles=2, people=1), warnings=[]))

    monkeypatch.setattr(R, "_default_recognize", fake)
    monkeypatch.setattr(R, "CACHE_DIR", tmp / "cache")
    prof = PR.Profile("h", options={"allow_heuristic": True})
    app.load_profile = lambda name: prof
    folder = next(x for x in call(base + "/api/folders")[1] if x["name"] == "курение")
    op = call(base + "/api/tune/eval", {"profile": "h", "folders": [folder["id"]], "mode": "events", "policy": "fixed", "n_boot": 20})[1]
    r = wait_op(base, op["id"])
    assert r["state"] == "done", r["error"]
    res = r["result"]
    assert res["metrics"]["f1"] == 1.0 and res["metrics"]["tp"] == 1 and res["clips"][0]["video"] and len(res["curve"]) > 3 and res["dirs"] == ["курение"]
    rid = res["id"]
    assert call(base + "/api/tune/experiments")[1][0]["id"] == rid
    again = call(f"{base}/api/tune/experiment/{rid}")[1]
    assert again["metrics"]["f1"] == 1.0 and again["ci"] and again["fingerprint"]
    assert call(base + "/api/tune/experiment/no_such_run")[0] == 404
    assert call(base + "/api/tune/eval", {"profile": "h", "folders": []})[0] == 400
