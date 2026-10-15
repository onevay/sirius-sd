"""Видео и разбор моделью через API платформы: папки и загрузка, воспроизведение в двух кодеках, очередь разбора и живой вывод, журнал конфигураций, экспорт видео с разметкой, профили разработчика."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient                       # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from rt_helpers import make_bundle                              # noqa: E402
from sd import model_tools as MT                                # noqa: E402
from sd import profiles as PR                                   # noqa: E402
from sd.server import media                                     # noqa: E402
from sd.server.app import create_app                            # noqa: E402
from sd.server.db import Database                               # noqa: E402
from sd.server.services.videos import NotFound, VideoService    # noqa: E402
from sd.server.settings import Settings, sqlite_url             # noqa: E402
from sd.server.store import SqlAlertStore                       # noqa: E402

API = "/api/v1"
FAST = ["hold", "d_min", "mouth_dur", "p_wm_dist"]


def make_video(path: Path, n: int = 30, fps: int = 10, fourcc: str = "mp4v") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (160, 96))
    for i in range(n):
        img = np.full((96, 160, 3), 40, np.uint8)
        cv2.rectangle(img, (10 + i, 20), (50 + i, 90), (0, 0, 255), -1)
        w.write(img)
    w.release()


@pytest.fixture()
def web(tmp_path, monkeypatch):
    """Сервер без входа (локальный режим) с видео в папках камер и данных; профили и реестр моделей — во временных каталогах."""
    import sd.models as M

    monkeypatch.setattr(media, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(PR, "EXPERIMENTS_DIR", tmp_path / "profiles")
    monkeypatch.setattr(MT, "MODELS", tmp_path / "models")
    monkeypatch.setattr(MT, "USER_REGISTRY", tmp_path / "models" / "user_models.yaml")
    monkeypatch.setattr(MT, "BUNDLE_KINDS", {"photo": tmp_path / "models" / "photo", "cycle": tmp_path / "models" / "cycle"})
    monkeypatch.setattr(M, "MODELS", tmp_path / "models")
    b = make_bundle(tmp_path / "models" / "cycle" / "fast1", FAST)
    PR.save(PR.Profile("good", "ок", config={"events.confidence_threshold": 0.6}, options={"cycle_bundle": str(b)}))
    streams, data = tmp_path / "streams", tmp_path / "data"
    make_video(streams / "gatchina-03-20261009T143000" / "part1.mp4")
    make_video(streams / "gatchina-03-20261009T143000" / "part2.mp4")
    make_video(streams / "pavlovsk-01-20261009T100000" / "a.mp4")
    make_video(data / "курение" / "s1.mp4")
    st = Settings(db_url=sqlite_url(tmp_path / "p.db"), secret_key="k" * 40, auth_required=False, allow_fs=True)
    db = Database(st.db_url)
    svc = VideoService(SqlAlertStore(db), streams, data, tmp_path / "out")
    with TestClient(create_app(st, db=db, videos_factory=lambda: svc)) as c:
        yield SimpleNamespace(c=c, svc=svc, tmp=tmp_path, db=db, settings=st)


def folder_of(web, name):
    return next(f for f in web.c.get(f"{API}/sources/folders").json() if f["name"] == name)


def first_video(web, folder_name):
    f = folder_of(web, folder_name)
    return web.c.get(f"{API}/sources/videos", params=dict(folder=f["id"])).json()[0]


def wait_ready(web, query, fmt, tries=400):
    for _ in range(tries):
        st = web.c.get(f"{API}/media/status", params={**query, "fmt": fmt}).json()
        if st["state"] in ("ready", "direct", "error"):
            return st
        time.sleep(0.1)
    raise AssertionError("не дождались перекодирования")


# ============================================================================================ источники и воспроизведение
def test_folder_listing_and_path_safety(web):
    fs = web.c.get(f"{API}/sources/folders").json()
    kinds = {f["name"]: f["kind"] for f in fs}
    assert kinds["gatchina-03-20261009T143000"] == "camera" and kinds["курение"] == "data"
    vids = web.c.get(f"{API}/sources/videos", params=dict(folder=folder_of(web, "курение")["id"])).json()
    assert [v["name"] for v in vids] == ["s1.mp4"] and vids[0]["analysis"] is None
    cam = web.c.get(f"{API}/sources/videos", params=dict(folder=folder_of(web, "gatchina-03-20261009T143000")["id"])).json()
    assert [v["name"] for v in cam] == ["part1.mp4", "part2.mp4"] and cam[0]["duration"] == pytest.approx(3.0, abs=0.2)
    assert web.c.get(f"{API}/sources/folder-of", params=dict(video=cam[0]["id"])).json()["folder"] == folder_of(web, "gatchina-03-20261009T143000")["id"]
    import base64

    evil = base64.urlsafe_b64encode(b"streams|../../etc").decode().rstrip("=")
    assert web.c.get(f"{API}/sources/videos", params=dict(folder=evil)).status_code == 404
    assert web.c.get(f"{API}/media/video/{evil}").status_code == 404
    assert web.c.get(f"{API}/sources/videos", params=dict(folder="%%%")).status_code == 404


def test_video_range_and_transcode_to_browser_format(web):
    vid = first_video(web, "gatchina-03-20261009T143000")["id"]
    assert wait_ready(web, dict(id=vid), "h264")["state"] == "ready"          # mp4v браузер не играет: перекодируется в фоне
    r = web.c.get(f"{API}/media/video/{vid}")
    assert r.status_code == 200 and r.headers["accept-ranges"] == "bytes" and len(r.content) > 500
    part = web.c.get(f"{API}/media/video/{vid}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100 and part.headers["content-range"].startswith("bytes 0-99/")
    assert web.c.get(f"{API}/media/video/{vid}", headers={"Range": "bytes=999999999-"}).status_code == 416


def test_format_negotiation_vp9_and_unknown_format(web):
    """Клиент без H.264 просит vp9: перекодируется в webm; неизвестный формат — как h264."""
    vid = first_video(web, "pavlovsk-01-20261009T100000")["id"]
    assert wait_ready(web, dict(id=vid), "vp9")["state"] == "ready"
    r = web.c.get(f"{API}/media/video/{vid}", params=dict(fmt="vp9"))
    assert r.status_code == 200 and r.headers["content-type"] == "video/webm" and r.content[:4] == b"\x1a\x45\xdf\xa3"          # заголовок EBML/webm
    assert web.c.get(f"{API}/media/video/{vid}", params=dict(fmt="bogus")).status_code == 422                                   # маршрут принимает только h264|vp9
    assert web.svc.media_for(vid, "bogus")["fmt"] == "h264"                                                                       # а сервис не падает и берёт h264


def test_upload_any_format_and_playback_in_both_codecs(web):
    for fourcc, ext in (("MJPG", ".avi"), ("mp4v", ".mp4")):
        src = web.tmp / f"s{ext}"
        make_video(src, 20, fourcc=fourcc)
        r = web.c.post(f"{API}/sources/upload", files={"file": (f"my clip{ext}", src.read_bytes(), "video/mp4")})
        assert r.status_code == 201, r.text
        v = r.json()
        assert v["name"].endswith(ext) and v["duration"] == pytest.approx(2.0, abs=0.3) and v["folder"]
        for fmt in ("h264", "vp9"):
            assert wait_ready(web, dict(id=v["id"]), fmt)["state"] == "ready"
            m = web.c.get(f"{API}/media/video/{v['id']}", params=dict(fmt=fmt))
            assert m.status_code == 200 and len(m.content) > 300 and m.headers["content-type"] == ("video/mp4" if fmt == "h264" else "video/webm")
    assert len(web.c.get(f"{API}/sources/videos", params=dict(folder=v["folder"])).json()) == 2
    assert web.c.post(f"{API}/sources/upload", files={"file": ("evil.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert web.c.post(f"{API}/sources/upload", files={"file": ("a.mp4", b"", "video/mp4")}).status_code == 400


def test_add_folder_and_remove_and_network_mode(web):
    mine = web.tmp / "mine"
    make_video(mine / "sub" / "a.mp4")
    r = web.c.post(f"{API}/sources/roots", json=dict(path=str(mine)))
    assert r.status_code == 201 and r.json()["n"] == 1
    key = r.json()["key"]
    sub = next(f for f in web.c.get(f"{API}/sources/folders").json() if f["name"].endswith("sub"))
    assert sub["removable"] and web.c.get(f"{API}/sources/videos", params=dict(folder=sub["id"])).json()[0]["name"] == "a.mp4"
    assert VideoService(SqlAlertStore(web.db, migrate=False), web.tmp / "streams", web.tmp / "data", web.tmp / "out").roots[key] == mine.resolve()      # переживает перезапуск
    assert web.c.post(f"{API}/sources/roots", json=dict(path=str(web.tmp / "нет такой"))).status_code == 400
    assert web.c.delete(f"{API}/sources/roots/{key}").status_code == 200 and not any(f["name"].endswith("sub") for f in web.c.get(f"{API}/sources/folders").json())
    assert web.c.delete(f"{API}/sources/roots/{key}").status_code == 404
    web.svc.allow_fs = False
    assert web.c.post(f"{API}/sources/roots", json=dict(path=str(web.tmp))).status_code == 403
    assert web.c.post(f"{API}/sources/upload", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code == 403


def test_alert_clip_is_playable_in_any_browser(web):
    assert web.c.post(f"{API}/demo", json=dict(n=3)).json()["n"] > 0
    al = next(a for a in web.c.get(f"{API}/alerts", params=dict(limit=10)).json()["items"] if a["has_clip"])
    for fmt in ("h264", "vp9"):
        assert wait_ready(web, dict(alert=al["id"]), fmt)["state"] in ("ready", "direct")
        r = web.c.get(f"{API}/media/alert/{al['id']}/clip", params=dict(fmt=fmt))
        assert r.status_code == 200 and len(r.content) > 300 and r.headers["content-type"] == ("video/mp4" if fmt == "h264" else "video/webm")
    assert web.c.get(f"{API}/media/alert/99999/clip").status_code == 404


# ============================================================================================ разбор видео
def test_analyze_job_lifecycle_with_fake_runner(web):
    svc = web.svc

    def fake_run(j):
        j.state = "running"
        j.alerts.append(dict(t_now=5.0, start=3.0, tid=1, confidence=0.8, explain="x"))
        d = svc._result_dir(j.video, j.profile)
        d.mkdir(parents=True, exist_ok=True)
        (d / "trace.json").write_text(json.dumps(dict(frames=[[0.1, [[1, 1, 2, 3, 4]]]], alerts=[], gt=[], frame_hw=[96, 160], fps=10)), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(dict(video=j.video, profile=j.profile, stats={"rt_factor": 2.0}, n_alerts=1, created="2026-10-09T10:00:00")), encoding="utf-8")
        j.progress, j.state = 1.0, "done"

    svc.run_job = fake_run
    svc.profiles = lambda: [dict(name="p1", kind="profile", ready=True, problems=[], describe={})]
    vid = first_video(web, "pavlovsk-01-20261009T100000")["id"]
    assert web.c.post(f"{API}/analysis/jobs", json=dict(video=vid, profile="nope")).status_code == 404
    r = web.c.post(f"{API}/analysis/jobs", json=dict(video=vid, profile="p1"))
    assert r.status_code == 202
    jid = r.json()["id"]
    for _ in range(100):
        j = web.c.get(f"{API}/analysis/jobs/{jid}").json()
        if j["state"] == "done":
            break
        time.sleep(0.05)
    assert j["state"] == "done" and j["alerts"][0]["tid"] == 1
    res = web.c.get(f"{API}/analysis/result", params=dict(video=vid)).json()
    assert res["trace"]["frames"][0][1][0][0] == 1 and res["meta"]["stats"]["rt_factor"] == 2.0
    assert web.c.get(f"{API}/analysis/result", params=dict(video=vid, profile="other")).json()["trace"] is None
    folder = web.c.get(f"{API}/sources/folder-of", params=dict(video=vid)).json()["folder"]
    summ = web.c.get(f"{API}/sources/videos", params=dict(folder=folder)).json()[0]["analysis"]
    assert summ["profile"] == "p1" and summ["rt_factor"] == 2.0
    js = web.c.post(f"{API}/analysis/jobs", json=dict(folder=folder, profile="p1")).json()
    assert len(js) == 1 and any(x["id"] == js[0]["id"] for x in web.c.get(f"{API}/analysis/jobs").json())


def test_real_job_reports_missing_models_as_error(web):
    """Без классификатора/весов задача завершается понятной ошибкой, а не зависает."""
    import sd.profiles as profmod

    web.svc.profiles = lambda: [dict(name="heur", kind="profile", ready=True, problems=[], describe={})]
    PR.save(PR.heuristic_profile().__class__(name="heur", options=dict(allow_heuristic=True), config={"pose.weights": "no-such-pose-weights"}), web.tmp / "pr")
    vid = first_video(web, "pavlovsk-01-20261009T100000")["id"]
    orig = profmod.load
    profmod.load = lambda n, d=None: orig("heur", web.tmp / "pr")
    try:
        jid = web.c.post(f"{API}/analysis/jobs", json=dict(video=vid, profile="heur")).json()["id"]
        for _ in range(300):
            j = web.c.get(f"{API}/analysis/jobs/{jid}").json()
            if j["state"] in ("done", "error"):
                break
            time.sleep(0.1)
    finally:
        profmod.load = orig
    assert j["state"] == "error" and j["error"]


def test_live_analysis_streams_frames_and_alerts(web, monkeypatch):
    """Анализ «в реальном времени»: кадры и тревоги приходят по мере обработки, в конце результат сохраняется."""
    from rt_helpers import make_pose_fn
    from sd.config import load_config
    from sd.realtime import replay as RP
    from sd.realtime.engine import StreamEngine
    from sd.realtime.sources import Pacer

    svc = web.svc
    long_video = svc.roots["data"] / "курение" / "long.mp4"
    make_video(long_video, n=300, fps=10)
    vid = svc.fid(long_video)
    cfg = ["cycles.th_in=0.35", "cycles.th_out=0.55", "events.confidence_threshold=0.5"]
    prof = PR.Profile("t", config={"cycles.th_in": 0.35, "cycles.th_out": 0.55, "events.confidence_threshold": 0.5}, options={"allow_heuristic": True})
    svc.profiles = lambda: [dict(name="t", kind="profile", ready=True, problems=[], describe={})]
    svc.load_profile = lambda name: prof
    svc.engine_factory = lambda p, cid, fps: StreamEngine(load_config(overrides=cfg), cid, make_pose_fn([(5.0, 2.0, 1.5, 1.5), (16.0, 2.0, 1.5, 1.5)]), keep_frames=False)
    monkeypatch.setattr(RP, "Pacer", lambda speed: Pacer(speed * 60))          # «реальное время» ×60, чтобы тест шёл секунды
    r = web.c.post(f"{API}/analysis/jobs", json=dict(video=vid, profile="t", live=True))
    assert r.status_code == 202 and r.json()["live"] is True
    jid = r.json()["id"]
    since, got, seen_partial = 0, 0, False
    for _ in range(400):
        s = web.c.get(f"{API}/analysis/jobs/{jid}/stream", params=dict(since=since)).json()
        since, got = s["next"], got + len(s["frames"])
        if s["state"] == "running" and 0 < got < 290:
            seen_partial = True
        if s["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["state"] == "done", s.get("error")
    assert got == 300 and s["frame_hw"] == [96, 160] and s["horizon"] >= 29 and len(s["alerts"]) == 1
    a = s["alerts"][0]
    assert a["t_open"] > a["end"] and a["tid"] == 1 and 0 < a["confidence"] <= 1 and a["delay"] > 0
    assert seen_partial, "кадры должны приходить по мере обработки, а не одним куском"
    saved = web.c.get(f"{API}/analysis/result", params=dict(video=vid)).json()
    assert len(saved["trace"]["frames"]) == 300 and saved["meta"]["n_alerts"] == 1
    person = saved["trace"]["frames"][50][1][0]
    assert len(person) == 6 and len(person[5]) == 17 * 3 and all(0 <= person[5][i] <= 100 for i in range(2, 51, 3))      # [tid, x1, y1, x2, y2, [x, y, conf%]×17]


def _journal_run(root: Path, rid: str, profile, f1: float, fingerprint: str, failed: bool = False) -> None:
    d = root / rid
    d.mkdir(parents=True)
    clips = [dict(clip_id="c1", error="boom" if failed else None)]
    (d / "meta.json").write_text(json.dumps(dict(id=rid, name=profile["name"], fingerprint=fingerprint, created="2026-10-09 10:00:00", mode="events", role="validation", profile=profile, clips=clips)), encoding="utf-8")
    (d / "report.json").write_text(json.dumps(dict(mode="events", metrics=dict(f1=f1, precision=0.9, recall=0.7, tp=7, fp=1, fn=3, clips=5, threshold=0.55), ci=dict(f1=[f1 - 0.1, f1 + 0.1]))), encoding="utf-8")


def test_journal_configs_are_selectable_with_quality(web, tmp_path, monkeypatch):
    """Конфигурации, по которым уже считалась оценка, попадают в выбор с качеством; профиль восстанавливается из журнала, а не из файла."""
    from sd import experiments as XP
    from sd import solver as SV

    monkeypatch.setattr(SV, "_device_issues", lambda p: [])          # готовность не должна зависеть от устройств (openvino/CUDA) той машины, где идёт тест
    root = tmp_path / "exp"
    monkeypatch.setattr(XP, "EXP_DIR", root)
    prof = PR.Profile(name="variant", config={"video.process_fps": 7}, options={"allow_heuristic": True})
    _journal_run(root, "20261009_100000_variant", prof.to_dict(), 0.71, prof.fingerprint())
    _journal_run(root, "20261009_090000_dead", PR.Profile(name="dead", options={"allow_heuristic": True}).to_dict(), 0.5, "x", failed=True)
    svc = web.svc
    assert [c["id"] for c in svc.journal_configs()] == ["20261009_100000_variant"]                           # прогон, где упали все клипы, качества не имеет
    p = next(x for x in svc.profiles() if x["name"] == "run:20261009_100000_variant")
    assert p["kind"] == "journal" and p["ready"] and p["quality"]["f1"] == 0.71 and p["quality"]["clips"] == 5 and "variant" in p["label"]
    assert svc.load_profile("run:20261009_100000_variant").cfg()["video"]["process_fps"] == 7
    with pytest.raises(Exception):
        svc.load_profile("run:нет_такого")
    _journal_run(root, "20261009_110000_stale", prof.to_dict(), 0.9, "другой_отпечаток")        # модели изменились после прогона → не выбирается
    assert "run:20261009_110000_stale" not in {x["name"] for x in svc.profiles()}
    assert any(x["name"] == "run:20261009_100000_variant" for x in web.c.get(f"{API}/analysis/profiles").json())


# ============================================================================================ видео с разметкой
def _saved_analysis(web, vid, profile="p1"):
    """Результат разбора, как его оставляет задача: трасса людей с ключевыми точками, тревога и эталон."""
    d = web.svc._result_dir(vid, profile)
    d.mkdir(parents=True, exist_ok=True)
    kp = [60.0, 40.0, 90.0] * 17
    frames = [[round(i / 10, 2), [[1, 10 + i, 20, 50 + i, 90, kp]]] for i in range(30)]
    (d / "trace.json").write_text(json.dumps(dict(frames=frames, alerts=[dict(tid=1, start=0.5, end=2.0, confidence=0.8, explain="2 цикла", t_open=1.0, delay=0.5)], gt=[dict(start=0.0, end=1.0)],
                                                  frame_hw=[96, 160], fps=10)), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(dict(video=vid, profile=profile, created="2026-10-09T10:00:00", n_alerts=1)), encoding="utf-8")


def test_export_video_with_overlay_and_download(web):
    vid = first_video(web, "pavlovsk-01-20261009T100000")["id"]
    assert web.c.post(f"{API}/analysis/export", json=dict(video=vid, profile="p1")).status_code == 404          # разбора ещё нет
    _saved_analysis(web, vid)
    r = web.c.post(f"{API}/analysis/export", json=dict(video=vid, profile="p1"))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["name"].endswith(".mp4") and "path" not in out and out["size_mb"] >= 0                               # путь на диске наружу не уходит
    f = web.c.get(f"{API}/media/export/{out['name']}")
    assert f.status_code == 200 and f.headers["content-type"] == "video/mp4" and b"ftyp" in f.content[:16]
    assert web.c.get(f"{API}/media/export/нет.mp4").status_code == 404
    with pytest.raises(NotFound):
        web.svc.export_path("../x.mp4")
    assert web.c.post(f"{API}/analysis/export", json=dict(video="bad")).status_code == 404


def test_export_requires_operator_and_is_audited(tmp_path, monkeypatch):
    """С входом: наблюдатель экспортировать не может, оператор может, действие попадает в журнал."""
    from sd.server.services import users as US

    monkeypatch.setattr(media, "CACHE", tmp_path / "cache")
    st = Settings(db_url=sqlite_url(tmp_path / "p.db"), secret_key="k" * 40, auth_required=True, allow_fs=False)
    db = Database(st.db_url)
    make_video(tmp_path / "data" / "курение" / "s1.mp4")
    svc = VideoService(SqlAlertStore(db), tmp_path / "streams", tmp_path / "data", tmp_path / "out")
    with TestClient(create_app(st, db=db, videos_factory=lambda: svc)) as c:
        with db.session() as s:
            US.create(s, "view1", "Str0ngPassw0rd", "viewer")
            US.create(s, "oper1", "Str0ngPassw0rd", "operator")
            US.create(s, "admin1", "Str0ngPassw0rd", "admin")

        def hdr(name):
            return {"Authorization": "Bearer " + c.post(f"{API}/auth/login", json=dict(username=name, password="Str0ngPassw0rd")).json()["access_token"]}

        vid = svc.fid(tmp_path / "data" / "курение" / "s1.mp4")
        d = svc._result_dir(vid, "p1")
        d.mkdir(parents=True)
        (d / "trace.json").write_text(json.dumps(dict(frames=[[0.0, [[1, 10, 20, 50, 90]]]], alerts=[], gt=[], frame_hw=[96, 160], fps=10)), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(dict(video=vid, profile="p1")), encoding="utf-8")
        assert c.post(f"{API}/analysis/export", json=dict(video=vid, profile="p1"), headers=hdr("view1")).status_code == 403
        r = c.post(f"{API}/analysis/export", json=dict(video=vid, profile="p1"), headers=hdr("oper1"))
        assert r.status_code == 200, r.text
        assert c.get(f"{API}/media/export/{r.json()['name']}", headers=hdr("view1")).status_code == 200
        assert any(x["action"] == "analysis.export" for x in c.get(f"{API}/audit", headers=hdr("admin1")).json()["items"])


# ============================================================================================ профили разработчика
def test_profiles_catalog_save_clone_yaml_delete(web):
    c = web.c
    cat = c.get(f"{API}/dev/catalog").json()
    assert "torch" in cat["runtimes"] and 960 in cat["imgsz"] and {"cycle", "pose", "photo"} <= set(cat)
    p = c.get(f"{API}/dev/profiles/good").json()
    assert p["main"]["events.confidence_threshold"] == 0.6 and p["options"]["cycle_bundle"].endswith("fast1") and p["effective"]["pose.weights"] and p["default"]
    p["config"]["events.confidence_threshold"] = 0.7
    r = c.put(f"{API}/dev/profiles/good", json=dict(description="новое", config=p["config"], options={"cycle_bundle": p["options"]["cycle_bundle"]}))
    assert r.status_code == 200 and r.json()["main"]["events.confidence_threshold"] == 0.7 and r.json()["description"] == "новое"
    assert c.put(f"{API}/dev/profiles/good", json=dict(config={"events.no_such_key": 1})).status_code in (400, 422)          # опечатка в ключе — ошибка, профиль не портится
    assert PR.load("good").threshold == 0.7
    assert c.post(f"{API}/dev/profiles/good/clone", json={"to": "copy1"}).status_code == 200 and "copy1" in PR.list_profiles()
    y = c.get(f"{API}/dev/profiles/copy1/yaml")
    assert y.status_code == 200 and "copy1" in y.text and "yaml" in y.headers["content-type"]
    assert c.delete(f"{API}/dev/profiles/copy1").status_code == 200 and "copy1" not in PR.list_profiles()
    assert c.delete(f"{API}/dev/profiles/copy1").status_code == 404
    assert c.get(f"{API}/dev/profiles/nope").status_code in (400, 404)


def test_registry_and_task_catalog(web):
    reg = web.c.get(f"{API}/dev/registry").json()
    assert any(x["name"] == "fast1" for x in reg["bundles"]) and "models" in reg and "solvers" in reg
    cat = web.c.get(f"{API}/dev/tasks/catalog").json()
    assert {"train_bundle", "enrich", "eval", "doctor"} <= {t["id"] for t in cat} and all("fields" in t for t in cat)
    assert web.c.post(f"{API}/dev/tasks", json=dict(task="no_such_task", values={})).status_code == 422
