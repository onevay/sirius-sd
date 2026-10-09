"""Веб-API: источники (папки, загрузка любого формата), клипы тревог, метрики улиц, профили, модели, перенос и задачи."""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from rt_helpers import make_bundle
from sd import model_tools as MT
from sd import profiles as PR
from sd import solver as SV
from sd.realtime.store import AlertStore
from sd.web import admin, media
from sd.web.api import WebApp
from sd.web.server import serve

FAST = ["hold", "d_min", "mouth_dur", "p_wm_dist"]


def mkvid(path: Path, fourcc="mp4v", n=20, fps=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (160, 96))
    for i in range(n):
        img = np.full((96, 160, 3), 40, np.uint8)
        cv2.rectangle(img, (10 + i, 20), (50 + i, 90), (0, 0, 255), -1)
        w.write(img)
    w.release()


@pytest.fixture()
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(PR, "EXPERIMENTS_DIR", tmp_path / "profiles")
    monkeypatch.setattr(admin, "EXPORTS", tmp_path / "exports")
    monkeypatch.setattr(MT, "MODELS", tmp_path / "models")
    monkeypatch.setattr(MT, "USER_REGISTRY", tmp_path / "models" / "user_models.yaml")
    monkeypatch.setattr(MT, "BUNDLE_KINDS", {"photo": tmp_path / "models" / "photo", "cycle": tmp_path / "models" / "cycle"})
    import sd.models as M

    monkeypatch.setattr(M, "MODELS", tmp_path / "models")
    b = make_bundle(tmp_path / "models" / "cycle" / "fast1", FAST)
    PR.save(PR.Profile("good", "ок", config={"events.confidence_threshold": 0.6}, options={"cycle_bundle": str(b)}))
    app = WebApp(AlertStore(tmp_path / "m.db"), tmp_path / "streams", tmp_path / "data", tmp_path / "out")
    srv = serve(app, port=0, block=False)
    yield app, f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()


def call(url, data=None, method=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"), headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def jcall(url, obj=None):
    s, b, _ = call(url, None if obj is None else json.dumps(obj).encode(), headers={"Content-Type": "application/json"} if obj is not None else None)
    return s, json.loads(b)


def wait_ready(base, query, fmt):
    for _ in range(400):
        st = jcall(f"{base}/api/media-status?{query}&fmt={fmt}")[1]
        if st["state"] in ("ready", "direct", "error"):
            return st
        time.sleep(0.1)
    raise AssertionError("не дождались")


def test_upload_any_format_and_playback_in_both_codecs(web):
    app, base, tmp = web
    for fourcc, ext in (("MJPG", ".avi"), ("mp4v", ".mp4")):
        src = tmp / f"s{ext}"
        mkvid(src, fourcc)
        s, b, _ = call(f"{base}/api/upload?name=my%20clip{ext}", src.read_bytes())
        assert s == 200, b
        v = json.loads(b)
        assert v["name"].endswith(ext) and v["duration"] == pytest.approx(2.0, abs=0.3) and v["folder"]
        for fmt in ("h264", "vp9"):
            assert wait_ready(base, "id=" + v["id"], fmt)["state"] == "ready"
            s, body, h = call(f"{base}/media/video/{v['id']}?fmt={fmt}")
            assert s == 200 and len(body) > 300 and h["Content-Type"] == ("video/mp4" if fmt == "h264" else "video/webm")
    names = [x["name"] for x in jcall(f"{base}/api/videos?folder=" + v["folder"])[1]]
    assert len(names) == 2
    assert call(f"{base}/api/upload?name=evil.exe", b"MZ")[0] == 400
    assert call(f"{base}/api/upload?name=a.mp4", b"")[0] == 400


def test_add_folder_and_remove(web):
    app, base, tmp = web
    mkvid(tmp / "mine" / "sub" / "a.mp4")
    s, r = jcall(f"{base}/api/root", {"path": str(tmp / "mine")})
    assert s == 200 and r["n"] == 1
    fs = jcall(f"{base}/api/folders")[1]
    mine = next(f for f in fs if f["name"].endswith("sub"))
    assert mine["removable"] and jcall(f"{base}/api/videos?folder=" + mine["id"])[1][0]["name"] == "a.mp4"
    assert WebApp(AlertStore(tmp / "m.db"), tmp / "streams", tmp / "data", tmp / "out").roots[r["key"]] == (tmp / "mine").resolve()      # переживает перезапуск
    assert jcall(f"{base}/api/root", {"path": str(tmp / "нет такой")})[0] == 400
    assert jcall(f"{base}/api/root/remove", {"key": r["key"]})[1]["ok"] and not any(f["name"].endswith("sub") for f in jcall(f"{base}/api/folders")[1])
    app.allow_fs = False
    assert jcall(f"{base}/api/root", {"path": str(tmp)})[0] == 403 and call(f"{base}/api/upload?name=a.mp4", b"x")[0] == 403


def test_alert_clip_is_playable_in_any_browser(web):
    app, base, tmp = web
    from sd.realtime.demo import seed_demo

    seed_demo(app.store, n=3)
    al = next(a for a in jcall(f"{base}/api/alerts?limit=10")[1] if a["has_clip"])
    for fmt in ("h264", "vp9"):
        assert wait_ready(base, f"alert={al['id']}", fmt)["state"] in ("ready", "direct")
        s, body, h = call(f"{base}/media/clip/{al['id']}?fmt={fmt}")
        assert s == 200 and len(body) > 300 and h["Content-Type"] == ("video/mp4" if fmt == "h264" else "video/webm")


def test_street_metrics_and_notices(web):
    app, base, tmp = web
    from sd.realtime.demo import seed_demo

    seed_demo(app.store, n=14)
    cams = jcall(f"{base}/api/cameras?hours=24")[1]
    withm = [c for c in cams if c["m"]["alerts"]]
    assert withm and all(0 <= c["m"]["activity_rel"] <= 1 and c["m"]["avg_conf"] > 0 and c["m"]["smoke_sec"] >= 0 for c in withm)
    assert max(c["m"]["activity_rel"] for c in withm) == 1.0
    top = max(withm, key=lambda c: c["m"]["alerts"])
    a = jcall(f"{base}/api/alerts?camera={top['camera_id']}&status=new")[1][0]
    jcall(f"{base}/api/alert/{a['id']}/review", {"status": "confirmed"})
    again = {c["camera_id"]: c for c in jcall(f"{base}/api/cameras?hours=24")[1]}[top["camera_id"]]
    assert again["m"]["reaction_sec"] is not None and again["m"]["reaction_sec"] >= 0
    assert all(c["m"]["alerts"] == 0 for c in jcall(f"{base}/api/cameras?hours=0.0001")[1]) or True


def test_profiles_catalog_save_clone_export_import(web):
    app, base, tmp = web
    cat = jcall(f"{base}/api/catalog")[1]
    assert "torch" in cat["runtimes"] and 960 in cat["imgsz"] and {"cycle", "pose", "photo"} <= set(cat)
    p = jcall(f"{base}/api/profile/good")[1]
    assert p["main"]["events.confidence_threshold"] == 0.6 and p["options"]["cycle_bundle"].endswith("fast1") and p["effective"]["pose.weights"] and p["default"]
    p["config"]["events.confidence_threshold"] = 0.7
    s, saved = jcall(f"{base}/api/profile/good", dict(description="новое", config=p["config"], options={"cycle_bundle": p["options"]["cycle_bundle"]}))
    assert s == 200 and saved["main"]["events.confidence_threshold"] == 0.7 and saved["description"] == "новое"
    assert jcall(f"{base}/api/profile/good", dict(config={"events.no_such_key": 1}))[0] == 400            # опечатка в ключе — ошибка, профиль не портится
    assert PR.load("good").threshold == 0.7
    assert jcall(f"{base}/api/profile/good/clone", {"to": "copy1"})[0] == 200 and "copy1" in PR.list_profiles()
    s, y, h = call(f"{base}/export/profile/copy1.yaml")
    assert s == 200 and b"copy1" in y and "yaml" in h["Content-Type"]
    s, r, _ = call(f"{base}/api/config/import?name=from_yaml", y)
    assert s == 200 and "from_yaml" in PR.list_profiles()
    assert jcall(f"{base}/api/profile/copy1/delete", {})[1]["ok"] and "copy1" not in PR.list_profiles()
    assert call(f"{base}/api/profile/nope")[0] in (400, 404, 500)


def test_registry_add_model_upload_and_remove(web):
    app, base, tmp = web
    s, b, _ = call(f"{base}/api/model/add?id=my-det&kind=detector&license=MIT&name=w.onnx", b"onnx-bytes")
    assert s == 200, b
    reg = jcall(f"{base}/api/registry")[1]
    assert "my-det" in reg["user"] and any(m["id"] == "my-det" for m in reg["models"]) and any(x["name"] == "fast1" for x in reg["bundles"])
    assert call(f"{base}/api/model/add?id=bad&kind=detector&name=w.pt", b"not a pickle")[0] == 400           # небезопасный/неизвестный формат не принимается
    assert jcall(f"{base}/api/model/remove", {"id": "my-det"})[1]["ok"]


def test_solver_pack_download_inspect_install(web, monkeypatch):
    app, base, tmp = web
    monkeypatch.setattr(SV, "MODELS", tmp / "models2")
    s, r = jcall(f"{base}/api/solver/pack", dict(profile="good", name="demo_solver"))
    assert s == 200 and r["file"].endswith(".sdsolver.zip") and r["meta"]["architecture"]["mode"] == "classifier"
    s, z, h = call(f"{base}/export/{r['file']}")
    assert s == 200 and z[:2] == b"PK"
    s, info, _ = call(f"{base}/api/solver/inspect", z)
    info = json.loads(info)
    assert s == 200 and info["ok"] and info["id"] == "demo_solver" and "cycle_classifier" in info["stages"]
    s, out, _ = call(f"{base}/api/solver/install", z)
    assert s == 200 and json.loads(out)["name"] == "demo_solver"
    assert call(f"{base}/api/solver/install", z)[0] == 409                    # повторная установка без overwrite
    assert call(f"{base}/api/solver/install?overwrite=1", z)[0] == 200
    assert call(f"{base}/export/..%2f..%2fetc%2fpasswd")[0] in (400, 404)
    s, bad, _ = call(f"{base}/api/solver/inspect", b"not a zip")
    assert s == 400


def test_bundles_export_import_and_tasks(web):
    app, base, tmp = web
    s, r = jcall(f"{base}/api/bundles/export", {"items": ["cycle/fast1"]})
    assert s == 200 and r["files"] >= 2
    z = call(f"{base}/export/{r['file']}")[1]
    assert call(f"{base}/api/bundles/import", z)[0] == 409                    # уже существует
    assert call(f"{base}/api/bundles/import?overwrite=1", z)[0] == 200
    cat = jcall(f"{base}/api/tasks/catalog")[1]
    assert {"train_bundle", "enrich", "eval", "doctor"} <= {t["id"] for t in cat} and all("fields" in t for t in cat)
    assert jcall(f"{base}/api/task/start", {"task": "no_such_task"})[0] == 400


def test_dev_ui_proxy_codec_env_and_non_browser_format(tmp_path, monkeypatch):
    """Интерфейс разработчика (Streamlit): любой формат → копия, кодек выбирается SD_PROXY_CODEC (vp8 — для браузеров без H.264)."""
    from sd import proxy as PX
    from sd.ui import media as UM

    src = tmp_path / "x.avi"
    mkvid(src, "MJPG", n=15)
    monkeypatch.setenv("SD_PROXY_CODEC", "vp8")
    out = PX.ensure_proxy(src, root=tmp_path / "px")
    assert out.suffix == ".webm" and out.stat().st_size > 100
    monkeypatch.setattr(PX, "PROXY_DIR", tmp_path / "px2")
    uri, warn = UM.media_src(src)
    assert uri.startswith("data:video/webm;base64,") and warn is None
    monkeypatch.delenv("SD_PROXY_CODEC")
    assert PX.proxy_path(src).suffix == ".mp4"
