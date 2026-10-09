import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from sd.realtime.store import AlertStore
from sd.web import media
from sd.web.api import WebApp
from sd.web.server import serve


def make_video(path: Path, n: int = 30, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 96))
    for i in range(n):
        img = np.full((96, 160, 3), 40, np.uint8)
        cv2.rectangle(img, (10 + i, 20), (50 + i, 90), (0, 0, 255), -1)
        w.write(img)
    w.release()


@pytest.fixture()
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CACHE", tmp_path / "cache")
    streams, data = tmp_path / "streams", tmp_path / "data"
    make_video(streams / "gatchina-03-20261009T143000" / "part1.mp4")
    make_video(streams / "gatchina-03-20261009T143000" / "part2.mp4")
    make_video(streams / "pavlovsk-01-20261009T100000" / "a.mp4")
    make_video(data / "курение" / "s1.mp4")
    (streams / "cameras.yaml").write_text("gatchina-03: {lat: 59.5762, lon: 30.1291, street: 'ул. Соборная'}\n", encoding="utf-8")
    app = WebApp(AlertStore(tmp_path / "m.db"), streams, data, tmp_path / "out")
    srv = serve(app, port=0, block=False)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield app, base
    srv.shutdown()


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def jget(url):
    s, b, _ = get(url)
    return s, json.loads(b)


def post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_static_index_and_health(web):
    app, base = web
    s, b, h = get(base + "/")
    assert s == 200 and b"<html" in b.lower()
    assert jget(base + "/api/health")[1]["ok"]
    assert get(base + "/..%2f..%2fetc/passwd")[0] in (200, 400, 404)           # выхода из static нет: отдаётся index.html или 404
    assert b"root:" not in get(base + "/..%2f..%2fetc/passwd")[1]


def test_cameras_have_geo_and_overrides(web):
    _, base = web
    cams = {c["camera_id"]: c for c in jget(base + "/api/cameras")[1]}
    assert set(cams) == {"gatchina-03", "pavlovsk-01"}
    assert cams["gatchina-03"]["placed"] and cams["gatchina-03"]["street"] == "ул. Соборная" and abs(cams["gatchina-03"]["lat"] - 59.5762) < 1e-6
    assert not cams["pavlovsk-01"]["placed"] and 59 < cams["pavlovsk-01"]["lat"] < 60
    assert cams["gatchina-03"]["n_chunks"] == 2


def test_camera_chunks_and_video_range_and_transcode(web):
    app, base = web
    cam = jget(base + "/api/camera/gatchina-03")[1]
    assert [c["name"] for c in cam["chunks"]] == ["part1.mp4", "part2.mp4"] and cam["chunks"][0]["duration"] == pytest.approx(3.0, abs=0.2)
    vid = cam["chunks"][0]["id"]
    for _ in range(200):          # mp4v браузер не играет: перекодируется в фоне
        st = jget(base + "/api/media-status?id=" + vid)[1]
        if st["state"] in ("ready", "direct"):
            break
        assert st["state"] in ("preparing",), st
        time.sleep(0.1)
    assert st["state"] == "ready"
    s, b, h = get(base + "/media/video/" + vid)
    assert s == 200 and h["Accept-Ranges"] == "bytes" and len(b) > 500
    s, part, h = get(base + "/media/video/" + vid, {"Range": "bytes=0-99"})
    assert s == 206 and len(part) == 100 and h["Content-Range"].startswith("bytes 0-99/")
    assert get(base + "/media/video/" + vid, {"Range": "bytes=999999999-"})[0] == 416


def test_format_negotiation_vp9_and_direct(web):
    """Клиент без H.264 просит vp9: перекодируется в webm; готовый webm отдаётся как есть, а для h264 перекодируется в mp4."""
    app, base = web
    vid = jget(base + "/api/camera/pavlovsk-01")[1]["chunks"][0]["id"]
    for _ in range(300):
        st = jget(base + f"/api/media-status?id={vid}&fmt=vp9")[1]
        if st["state"] in ("ready", "error"):
            break
        time.sleep(0.1)
    assert st["state"] == "ready", st
    s, b, h = get(base + f"/media/video/{vid}?fmt=vp9")
    assert s == 200 and h["Content-Type"] == "video/webm" and b[:4] == b"\x1a\x45\xdf\xa3"          # заголовок EBML/webm
    assert jget(base + f"/api/media-status?id={vid}&fmt=bogus")[1]["state"] in ("preparing", "ready")        # неизвестный формат → h264


def test_folder_listing_and_path_safety(web):
    app, base = web
    fs = jget(base + "/api/folders")[1]
    kinds = {f["name"]: f["kind"] for f in fs}
    assert kinds["gatchina-03-20261009T143000"] == "camera" and kinds["курение"] == "data"
    folder = next(f for f in fs if f["name"] == "курение")
    vids = jget(base + "/api/videos?folder=" + folder["id"])[1]
    assert [v["name"] for v in vids] == ["s1.mp4"] and vids[0]["analysis"] is None
    import base64
    evil = base64.urlsafe_b64encode(b"streams|../../etc").decode().rstrip("=")
    assert get(base + "/api/videos?folder=" + evil)[0] == 404
    assert get(base + "/media/video/" + evil)[0] == 404
    assert get(base + "/api/videos?folder=%%%")[0] in (400, 404)


def test_alert_review_flow_and_demo(web):
    app, base = web
    assert post(base + "/api/demo", {})[1]["n"] > 0
    al = jget(base + "/api/alerts?status=new&limit=5")[1]
    assert al and all(a["status"] == "new" for a in al)
    a = al[0]
    s, r = post(f"{base}/api/alert/{a['id']}/review", {"status": "confirmed", "reviewer": "оп1", "note": "ок"})
    assert s == 200 and r["status"] == "confirmed" and r["reviewer"] == "оп1" and r["status_ru"] == "подтверждена"
    assert post(f"{base}/api/alert/{a['id']}/review", {"status": "hack"})[0] == 400
    assert post(f"{base}/api/alert/999999/review", {"status": "false"})[0] == 404
    st = jget(base + "/api/stats")[1]
    assert st["confirmed"] == 1 and st["new"] >= 1
    assert get(f"{base}/media/alert/{a['id']}/thumb")[0] in (200, 404)
    s, b, _ = get(base + "/api/journal.csv")
    assert s == 200 and "оп1" in b.decode("utf-8-sig")
    assert post(base + "/api/demo", {"clear": True})[1]["n"] > 0


def test_analyze_job_lifecycle_with_fake_runner(web):
    app, base = web
    from sd import profiles as PR

    def fake_run(j):
        j.state = "running"
        j.alerts.append(dict(t_now=5.0, start=3.0, tid=1, confidence=0.8, explain="x"))
        d = app._result_dir(j.video, j.profile)
        d.mkdir(parents=True, exist_ok=True)
        (d / "trace.json").write_text(json.dumps(dict(frames=[[0.1, [[1, 1, 2, 3, 4]]]], alerts=[], gt=[], frame_hw=[96, 160], fps=10)), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(dict(video=j.video, profile=j.profile, stats={"rt_factor": 2.0})), encoding="utf-8")
        j.progress, j.state = 1.0, "done"

    app.run_job = fake_run
    app.profiles = lambda: [dict(name="p1", kind="profile", ready=True, problems=[], describe={})]
    vid = jget(base + "/api/camera/pavlovsk-01")[1]["chunks"][0]["id"]
    assert post(base + "/api/analyze", {"video": vid, "profile": "nope"})[0] == 404
    s, j = post(base + "/api/analyze", {"video": vid, "profile": "p1"})
    assert s == 202
    for _ in range(100):
        j = jget(f"{base}/api/job/{j['id']}")[1]
        if j["state"] == "done":
            break
        time.sleep(0.05)
    assert j["state"] == "done" and j["alerts"][0]["tid"] == 1
    r = jget(f"{base}/api/analysis?video={vid}")[1]
    assert r["trace"]["frames"][0][1][0][0] == 1 and r["meta"]["stats"]["rt_factor"] == 2.0
    assert jget(f"{base}/api/analysis?video={vid}&profile=other")[1]["trace"] is None
    folder = jget(base + "/api/folder-of?video=" + vid)[1]["folder"]
    summ = jget(base + "/api/videos?folder=" + folder)[1][0]["analysis"]
    assert summ["profile"] == "p1" and summ["rt_factor"] == 2.0
    js = post(base + "/api/analyze", {"folder": folder, "profile": "p1"})[1]
    assert len(js) == 1 and any(x["id"] == js[0]["id"] for x in jget(base + "/api/jobs")[1])


def test_real_job_reports_missing_models_as_error(web):
    """Без классификатора/весов задача завершается понятной ошибкой, а не зависает."""
    app, base = web
    from sd import profiles as PR

    app.profiles = lambda: [dict(name="heur", kind="profile", ready=True, problems=[], describe={})]
    PR.save(PR.heuristic_profile().__class__(name="heur", options=dict(allow_heuristic=True), config={"pose.weights": "no-such-pose-weights"}), Path(app.out) / "pr")
    vid = jget(base + "/api/camera/pavlovsk-01")[1]["chunks"][0]["id"]
    import sd.profiles as profmod
    orig = profmod.load
    profmod.load = lambda n, d=None: orig("heur", Path(app.out) / "pr")
    try:
        s, j = post(base + "/api/analyze", {"video": vid, "profile": "heur"})
        for _ in range(300):
            j = jget(f"{base}/api/job/{j['id']}")[1]
            if j["state"] in ("done", "error"):
                break
            time.sleep(0.1)
    finally:
        profmod.load = orig
    assert j["state"] == "error" and j["error"]


def test_live_analysis_streams_frames_and_alerts(web, monkeypatch):
    """Анализ «в реальном времени»: кадры и тревоги приходят по мере обработки, в конце результат сохраняется."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from rt_helpers import make_pose_fn
    from sd import profiles as PR
    from sd.config import load_config
    from sd.realtime import replay as RP
    from sd.realtime.engine import StreamEngine
    from sd.realtime.sources import Pacer

    app, base = web
    long_video = app.roots["data"] / "курение" / "long.mp4"
    make_video(long_video, n=300, fps=10)
    vid = app.fid(long_video)
    cfg = ["cycles.th_in=0.35", "cycles.th_out=0.55", "events.confidence_threshold=0.5"]
    prof = PR.Profile("t", config={"cycles.th_in": 0.35, "cycles.th_out": 0.55, "events.confidence_threshold": 0.5}, options={"allow_heuristic": True})
    app.profiles = lambda: [dict(name="t", kind="profile", ready=True, problems=[], describe={})]
    app.load_profile = lambda name: prof
    app.engine_factory = lambda p, cid, fps: StreamEngine(load_config(overrides=cfg), cid, make_pose_fn([(5.0, 2.0, 1.5, 1.5), (16.0, 2.0, 1.5, 1.5)]), keep_frames=False)
    monkeypatch.setattr(RP, "Pacer", lambda speed: Pacer(speed * 60))          # «реальное время» ×60, чтобы тест шёл секунды
    s, j = post(base + "/api/analyze", {"video": vid, "profile": "t", "live": True})
    assert s == 202 and j["live"] is True
    since, got, seen_partial = 0, 0, False
    for _ in range(400):
        r = jget(f"{base}/api/job/{j['id']}/stream?since={since}")[1]
        since, got = r["next"], got + len(r["frames"])
        if r["state"] == "running" and 0 < got < 290:
            seen_partial = True
        if r["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert r["state"] == "done", r.get("error")
    assert got == 300 and r["frame_hw"] == [96, 160] and r["horizon"] >= 29 and len(r["alerts"]) == 1
    a = r["alerts"][0]
    assert a["t_open"] > a["end"] and a["tid"] == 1 and 0 < a["confidence"] <= 1 and a["delay"] > 0
    assert seen_partial, "кадры должны приходить по мере обработки, а не одним куском"
    saved = jget(f"{base}/api/analysis?video={vid}")[1]
    assert len(saved["trace"]["frames"]) == 300 and saved["meta"]["n_alerts"] == 1
