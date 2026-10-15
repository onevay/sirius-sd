"""Бэкенд платформы (`sd.server`): миграции, безопасность, пользователи и роли, API тревог/камер/разбора/разработчика, хранилище для воркера, перенос из прежней базы, команды CLI."""
import json
import sqlite3
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from alembic.autogenerate import compare_metadata               # noqa: E402
from alembic.runtime.migration import MigrationContext          # noqa: E402
from fastapi.testclient import TestClient                       # noqa: E402
from sqlalchemy import create_engine, inspect                   # noqa: E402

from sd import paths as P                                       # noqa: E402
from sd.realtime.engine import AlertUpdate                      # noqa: E402
from sd.server import clock, migrate as M, security             # noqa: E402
from sd.server.app import create_app                            # noqa: E402
from sd.server.db import Base, Database                         # noqa: E402
from sd.server.services import alerts as AS, users as US        # noqa: E402
from sd.server.settings import Settings, sqlite_url             # noqa: E402
from sd.server.store import SqlAlertStore                       # noqa: E402

PW = "Str0ngPassw0rd"


def au(key="c:1:5.00", kind="open", start=5.0, end=20.0, conf=0.8, cam="gatchina-01"):
    return AlertUpdate(kind, key, cam, 1, start, end, (start + end) / 2, conf, "two_cycles", 2, "2 цикла", (1, 2, 3, 4), end + 1)


def meta(t="2026-10-09T10:00:05", district="gatchina"):
    return dict(district=district, cam_index="01", chunk="$ROOT/x.mp4", chunk_offset=0.0, t_abs=t, fingerprint="fp")


@pytest.fixture
def env(tmp_path):
    from sd.server.services.videos import VideoService

    st = Settings(db_url=sqlite_url(tmp_path / "p.db"), secret_key="k" * 40, auth_required=True, allow_fs=False)
    db = Database(st.db_url)
    streams, data = tmp_path / "streams", tmp_path / "data"
    cam = streams / "gatchina-03-20261009T143000"
    cam.mkdir(parents=True)
    (cam / "c1.mp4").write_bytes(b"x")
    data.mkdir()
    app = create_app(st, db=db, videos_factory=lambda: VideoService(SqlAlertStore(db, migrate=False), streams, data, tmp_path / "out"))
    with TestClient(app) as c:
        with db.session() as s:
            for name, role in (("admin1", "admin"), ("dev1", "developer"), ("oper1", "operator"), ("view1", "viewer")):
                US.create(s, name, PW, role)
        yield SimpleNamespace(client=c, app=app, db=db, settings=st, store=SqlAlertStore(db, migrate=False), tmp=tmp_path)


def login(env, name, **kw):
    r = env.client.post("/api/v1/auth/login", json=dict(username=name, password=PW, **kw))
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ============================================================================================ миграции
def test_migrations_up_down_and_no_drift(tmp_path):
    url = sqlite_url(tmp_path / "m.db")
    assert M.current(url) == [] and not M.is_current(url)
    M.upgrade(url)
    assert M.is_current(url) and M.current(url) == M.heads()
    eng = create_engine(url)
    assert {"users", "api_tokens", "audit_log", "cameras", "chunks", "alerts", "alert_reviews"} <= set(inspect(eng).get_table_names())
    with eng.connect() as c:
        assert compare_metadata(MigrationContext.configure(c, opts=dict(compare_type=True)), Base.metadata) == []       # модели и миграции не расходятся
    M.upgrade(url)                                                                                                   # повторный подъём — без изменений
    M.downgrade(url, "0001_identity")
    assert M.current(url) == ["0001_identity"] and "alerts" not in inspect(create_engine(url)).get_table_names()
    M.downgrade(url, "base")
    assert M.current(url) == []
    assert [r["revision"] for r in M.history()] == ["0001_identity", "0002_monitoring"]


def test_check_constraints_and_cascades(tmp_path):
    from sqlalchemy.exc import IntegrityError

    from sd.server.models import Alert, AlertReview, Camera, Chunk

    db = Database(sqlite_url(tmp_path / "c.db"))
    from sd.server.store import ensure_schema

    ensure_schema(db)
    now = clock.now()
    with pytest.raises(IntegrityError), db.session() as s:
        s.add(Alert(key="k", camera_id="c", start_sec=0, end_sec=1, peak_sec=0.5, confidence=0.5, status="bogus", created_at=now, updated_at=now))
    with db.session() as s:
        s.add(Camera(camera_id="c", city="", title="", street="", created_at=now, status="idle", enabled=True, processed_sec=0))
        s.flush()
        s.add(Chunk(camera_id="c", name="a", duration=1, offset_sec=0, done_at=now))
        a = Alert(key="k", camera_id="c", start_sec=0, end_sec=1, peak_sec=0.5, confidence=0.5, created_at=now, updated_at=now)
        s.add(a)
        s.flush()
        s.add(AlertReview(alert_id=a.id, status="confirmed", reviewer_name="x", created_at=now))
    with db.session() as s:
        s.delete(s.get(Camera, "c"))                         # фрагменты камеры удаляются каскадом
        s.delete(s.get(Alert, 1))                            # история решений — вместе с тревогой
    with db.session() as s:
        from sqlalchemy import func, select

        assert s.scalar(select(func.count()).select_from(Chunk)) == 0 and s.scalar(select(func.count()).select_from(AlertReview)) == 0


# ============================================================================================ безопасность
def test_passwords_tokens_roles():
    h = security.hash_password(PW, n=2 ** 10)
    assert security.verify_password(PW, h) and not security.verify_password(PW + "x", h) and not security.verify_password(PW, "garbage")
    assert h != security.hash_password(PW, n=2 ** 10)                      # соль у каждого своя
    assert security.password_problem("short") and security.password_problem("alllowercase1") and security.password_problem(PW) is None
    tok = security.make_access_token("s" * 32, 7, 3)
    assert security.read_access_token("s" * 32, tok, 60) == (7, 3)
    assert security.read_access_token("other" * 8, tok, 60) is None and security.read_access_token("s" * 32, tok + "x", 60) is None
    time.sleep(1.1)
    assert security.read_access_token("s" * 32, tok, 0) is None            # срок истёк
    t, prefix, hh = security.new_api_token()
    assert t.startswith("sdt_") and t.startswith(prefix) and security.hash_api_token(t) == hh and security.is_api_token(t)
    assert security.has_role("admin", "viewer") and security.has_role("operator", "operator") and not security.has_role("viewer", "operator") and not security.has_role("x", "viewer")


def test_user_rules(tmp_path):
    db = Database(sqlite_url(tmp_path / "u.db"))
    from sd.server.store import ensure_schema

    ensure_schema(db)
    with db.session() as s:
        a = US.create(s, "Boss", PW, "admin")
        assert a.username == "boss"                                         # имя в нижнем регистре
        with pytest.raises(US.UserError):
            US.create(s, "boss", PW)
        with pytest.raises(US.UserError):
            US.create(s, "x", PW)
        with pytest.raises(US.UserError):
            US.create(s, "newuser", "weak")
        with pytest.raises(US.UserError):
            US.create(s, "newuser", PW, "root")
        with pytest.raises(US.UserError):
            US.update(s, a, role="viewer")                                  # последнего администратора не понизить
        with pytest.raises(US.UserError):
            US.update(s, a, is_active=False)
        with pytest.raises(US.UserError):
            US.delete(s, a)
        v0 = a.token_version
        US.update(s, a, password="An0therPassword")
        assert a.token_version == v0 + 1 and US.authenticate(s, "boss", "An0therPassword") and US.authenticate(s, "boss", PW) is None and US.authenticate(s, "nobody", PW) is None
        b = US.create(s, "second", PW, "admin")
        US.update(s, a, role="viewer")                                      # теперь можно: есть второй администратор
        with pytest.raises(US.UserError):
            US.update(s, b, is_active=False)                                # b — теперь единственный администратор
        US.update(s, a, is_active=False)
        assert US.authenticate(s, "boss", "An0therPassword") is None        # заблокированный не входит
        assert US.bootstrap_admin(s, "again", PW) is None                   # пользователи уже есть


# ============================================================================================ API: вход и роли
def test_health_and_auth_required(env):
    c = env.client
    h = c.get("/api/v1/health").json()
    assert h["ok"] and h["db"] and h["schema_current"] and h["auth_required"]
    assert c.get("/api/v1/alerts").status_code == 401
    assert c.get("/api/v1/alerts", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert c.get("/docs").status_code == 200 and "/api/v1/alerts" in c.get("/api/v1/openapi.json").json()["paths"]


def test_login_logout_password_and_token_revocation(env):
    c = env.client
    assert c.post("/api/v1/auth/login", json=dict(username="oper1", password="wrong")).status_code == 401
    h = login(env, "oper1")
    me = c.get("/api/v1/auth/me", headers=h).json()
    assert me["username"] == "oper1" and me["role"] == "operator" and "password" not in json.dumps(me)
    r = c.post("/api/v1/auth/password", headers=h, json=dict(old_password="bad", new_password="N3wPasswordHere"))
    assert r.status_code == 403
    assert c.post("/api/v1/auth/password", headers=h, json=dict(old_password=PW, new_password="weak")).status_code == 422
    assert c.post("/api/v1/auth/password", headers=h, json=dict(old_password=PW, new_password="N3wPasswordHere")).status_code == 200
    assert c.get("/api/v1/auth/me", headers=h).status_code == 401           # токен, выданный до смены пароля, больше не действует
    assert c.post("/api/v1/auth/login", json=dict(username="oper1", password="N3wPasswordHere")).status_code == 200
    h2 = login(env, "view1")
    assert c.post("/api/v1/auth/logout-all", headers=h2).status_code == 200 and c.get("/api/v1/auth/me", headers=h2).status_code == 401


def test_login_lockout_and_failed_login_audited(env):
    c = env.client
    for _ in range(5):
        assert c.post("/api/v1/auth/login", json=dict(username="view1", password="bad")).status_code == 401
    r = c.post("/api/v1/auth/login", json=dict(username="view1", password=PW))          # даже верный пароль — пока блокировка
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    rows = c.get("/api/v1/audit", headers=login(env, "admin1"), params=dict(action="auth.login_failed")).json()
    assert rows["total"] == 5 and rows["items"][0]["entity_id"] == "view1"


def test_cookie_only_for_safe_methods_and_query_token_only_for_media(env):
    c = env.client
    env.store.apply_update(au(), meta())
    tok = c.post("/api/v1/auth/login", json=dict(username="oper1", password=PW, cookie=True)).json()["access_token"]
    assert "sd_token" in c.cookies
    assert c.get("/api/v1/alerts").status_code == 200                                    # cookie годится для чтения
    assert c.post("/api/v1/alerts/1/review", json=dict(status="confirmed")).status_code == 401      # …но не для изменений (защита от CSRF)
    assert c.post("/api/v1/alerts/1/review", json=dict(status="confirmed"), headers={"Authorization": f"Bearer {tok}"}).status_code == 200
    c.cookies.clear()
    assert c.get("/api/v1/alerts", params=dict(access_token=tok)).status_code == 401        # токен в адресе — не для обычных маршрутов
    assert c.get("/api/v1/alerts/1/thumb", params=dict(access_token=tok)).status_code == 404   # а для файлов принимается (файла нет, но вход прошёл)


def test_roles_matrix(env):
    c = env.client
    env.store.apply_update(au(), meta())
    H = {n: login(env, n) for n in ("view1", "oper1", "dev1", "admin1")}
    assert c.get("/api/v1/alerts", headers=H["view1"]).status_code == 200
    assert c.post("/api/v1/alerts/1/review", headers=H["view1"], json=dict(status="confirmed")).status_code == 403
    assert c.post("/api/v1/alerts/1/review", headers=H["oper1"], json=dict(status="confirmed")).status_code == 200
    assert c.get("/api/v1/dev/profiles", headers=H["oper1"]).status_code == 403
    assert c.get("/api/v1/dev/profiles", headers=H["dev1"]).status_code == 200
    assert c.get("/api/v1/users", headers=H["dev1"]).status_code == 403
    assert c.get("/api/v1/users", headers=H["admin1"]).status_code == 200
    assert c.put("/api/v1/cameras/x", headers=H["dev1"], json={}).status_code == 403


def test_api_tokens(env):
    c = env.client
    h = login(env, "dev1")
    r = c.post("/api/v1/auth/tokens", headers=h, json=dict(name="ci"))
    assert r.status_code == 201
    tok, tid = r.json()["token"], r.json()["id"]
    th = {"Authorization": f"Bearer {tok}"}
    assert c.get("/api/v1/auth/me", headers=th).json()["username"] == "dev1" and c.get("/api/v1/dev/profiles", headers=th).status_code == 200
    listed = c.get("/api/v1/auth/tokens", headers=h).json()
    assert listed[0]["prefix"] == tok[:12] and "token" not in listed[0] and "token_hash" not in listed[0]
    assert c.delete(f"/api/v1/auth/tokens/{tid}", headers=login(env, "view1")).status_code == 404      # чужой токен не отозвать
    assert c.delete(f"/api/v1/auth/tokens/{tid}", headers=h).status_code == 200 and c.get("/api/v1/auth/me", headers=th).status_code == 401
    exp = c.post("/api/v1/auth/tokens", headers=h, json=dict(name="old", ttl_days=1)).json()
    with env.db.session() as s:
        from sd.server.models import ApiToken

        s.get(ApiToken, exp["id"]).expires_at = clock.now() - timedelta(minutes=1)
    assert c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {exp['token']}"}).status_code == 401
    US_blocked = login(env, "view1")
    with env.db.session() as s:
        US.update(s, US.get_by_username(s, "view1"), is_active=False)
    assert c.get("/api/v1/auth/me", headers=US_blocked).status_code == 401                             # блокировка отзывает токены


def test_auth_disabled_mode_acts_as_local_admin(tmp_path):
    from dataclasses import replace

    st = replace(Settings(db_url=sqlite_url(tmp_path / "n.db"), secret_key="k" * 40), auth_required=False)
    with TestClient(create_app(st, db=Database(st.db_url), videos_factory=lambda: None)) as c:
        me = c.get("/api/v1/auth/me").json()
        assert me["username"] == "local" and me["role"] == "admin" and c.get("/api/v1/users").status_code == 200
        assert c.put("/api/v1/cameras/a-1", json=dict(city="СПб")).status_code == 200
        assert c.get("/api/v1/audit").json()["items"][0]["username"] == "local"


# ============================================================================================ API: администрирование
def test_users_crud_and_audit(env):
    c, h = env.client, login(env, "admin1")
    r = c.post("/api/v1/users", headers=h, json=dict(username="newbie", password=PW, role="operator", display_name="Новичок"))
    assert r.status_code == 201 and r.json()["display_name"] == "Новичок" and "password" not in r.text
    uid = r.json()["id"]
    assert c.post("/api/v1/users", headers=h, json=dict(username="newbie", password=PW)).status_code == 422
    assert c.patch(f"/api/v1/users/{uid}", headers=h, json=dict(role="developer", password="Another1Password")).json()["role"] == "developer"
    assert c.post("/api/v1/auth/login", json=dict(username="newbie", password="Another1Password")).status_code == 200
    me = c.get("/api/v1/auth/me", headers=h).json()
    assert c.patch(f"/api/v1/users/{me['id']}", headers=h, json=dict(role="viewer")).status_code == 422       # единственного администратора не понизить
    assert c.delete(f"/api/v1/users/{uid}", headers=h).status_code == 200 and c.get(f"/api/v1/users/{uid}", headers=h).status_code == 404
    acts = [x["action"] for x in c.get("/api/v1/audit", headers=h, params=dict(limit=50)).json()["items"]]
    assert {"user.create", "user.update", "user.delete", "auth.login"} <= set(acts)
    audit = c.get("/api/v1/audit", headers=h, params=dict(action="user.update")).json()["items"][0]
    assert audit["details"]["password_changed"] is True and "Another1Password" not in json.dumps(audit)       # пароль в журнал не попадает


# ============================================================================================ API: камеры
def test_cameras_crud_validation_and_metrics(env):
    c, ha, hv = env.client, login(env, "admin1"), login(env, "view1")
    r = c.put("/api/v1/cameras/gatchina-01", headers=ha, json=dict(city="Гатчина", district="gatchina", cam_index="01", lat=59.57, lon=30.13, title="Вокзал"))
    assert r.status_code == 200 and r.json()["city"] == "Гатчина" and r.json()["status"] == "idle"
    assert c.put("/api/v1/cameras/gatchina-01", headers=ha, json=dict(lat=123.0)).status_code == 422
    c.put("/api/v1/cameras/pavlovsk-01", headers=ha, json=dict(city="Павловск", district="pavlovsk"))
    env.store.apply_update(au(), meta())
    env.store.apply_update(au(key="c:2:50.00", start=50, end=60, conf=0.6), meta("2026-10-09T10:00:50"))
    env.store.review(2, "false")
    rows = c.get("/api/v1/cameras", headers=hv, params=dict(city="Гатчина")).json()
    assert len(rows) == 1 and rows[0]["alerts"] == 2 and rows[0]["pending"] == 1 and rows[0]["mean_confidence"] == pytest.approx(0.7)
    assert {d["city"] for d in c.get("/api/v1/districts", headers=hv).json()} == {"Гатчина", "Павловск"}
    assert c.get("/api/v1/cameras/nope", headers=hv).status_code == 404
    assert c.delete("/api/v1/cameras/pavlovsk-01", headers=ha).status_code == 200 and c.delete("/api/v1/cameras/pavlovsk-01", headers=ha).status_code == 404
    c.delete("/api/v1/cameras/gatchina-01", headers=ha)                                                 # тревоги остаются
    assert c.get("/api/v1/alerts", headers=hv).json()["total"] == 2


# ============================================================================================ API: тревоги
def seed(env, n=5):
    for i in range(n):
        env.store.apply_update(au(key=f"c:{i}:{i}", start=i * 30, end=i * 30 + 15, conf=0.5 + i * 0.1, cam="gatchina-01" if i % 2 == 0 else "pavlovsk-02"), meta(f"2026-10-09T10:0{i}:00", "gatchina" if i % 2 == 0 else "pavlovsk"))


def test_alerts_list_filters_and_pagination(env):
    c, h = env.client, login(env, "view1")
    seed(env)
    page = c.get("/api/v1/alerts", headers=h, params=dict(limit=2)).json()
    assert page["total"] == 5 and len(page["items"]) == 2 and page["items"][0]["t_abs"] > page["items"][1]["t_abs"]            # новые сверху
    p2 = c.get("/api/v1/alerts", headers=h, params=dict(limit=2, offset=2)).json()
    assert {a["id"] for a in page["items"]}.isdisjoint({a["id"] for a in p2["items"]})
    assert c.get("/api/v1/alerts", headers=h, params=dict(oldest_first=True, limit=1)).json()["items"][0]["id"] == 1
    assert c.get("/api/v1/alerts", headers=h, params=dict(camera="pavlovsk-02")).json()["total"] == 2
    assert c.get("/api/v1/alerts", headers=h, params=dict(district=["gatchina", "pavlovsk"])).json()["total"] == 5
    assert c.get("/api/v1/alerts", headers=h, params=dict(min_conf=0.75)).json()["total"] == 2
    assert c.get("/api/v1/alerts", headers=h, params=dict(since="2026-10-09T10:02:00", until="2026-10-09T10:03:00")).json()["total"] == 2
    assert c.get("/api/v1/alerts", headers=h, params=dict(status="confirmed")).json()["total"] == 0
    assert c.get("/api/v1/alerts", headers=h, params=dict(status="bogus")).status_code == 422
    assert c.get("/api/v1/alerts", headers=h, params=dict(limit=100000)).status_code == 422
    a = c.get("/api/v1/alerts/3", headers=h).json()
    assert a["box"] == [1, 2, 3, 4] and a["status"] == "new" and a["has_thumb"] is False and c.get("/api/v1/alerts/999", headers=h).status_code == 404


def test_review_history_bulk_and_validation(env):
    c, ho, hv = env.client, login(env, "oper1"), login(env, "view1")
    seed(env, 3)
    r = c.post("/api/v1/alerts/1/review", headers=ho, json=dict(status="false", reason="drink", note="пил воду")).json()
    assert r["status"] == "false" and r["reviewer"] == "oper1" and r["reviewed_at"] and r["note"] == "пил воду"
    c.post("/api/v1/alerts/1/review", headers=ho, json=dict(status="confirmed", note="передумал"))
    hist = c.get("/api/v1/alerts/1/reviews", headers=hv).json()
    assert [x["status"] for x in hist] == ["false", "confirmed"] and hist[0]["reason"] == "drink" and hist[0]["reviewer_name"] == "oper1"      # прежнее решение не стёрто
    assert c.post("/api/v1/alerts/1/review", headers=ho, json=dict(status="new")).json()["reviewed_at"] is None                              # возврат в очередь
    assert c.post("/api/v1/alerts/1/review", headers=ho, json=dict(status="maybe")).status_code == 422
    assert c.post("/api/v1/alerts/1/review", headers=ho, json=dict(status="false", reason="alien")).status_code == 422
    assert c.post("/api/v1/alerts/99/review", headers=ho, json=dict(status="false")).status_code == 404
    b = c.post("/api/v1/alerts/review", headers=ho, json=dict(ids=[2, 3, 3, 77], status="unsure")).json()
    assert b == dict(done=[2, 3], missing=[77])
    st = c.get("/api/v1/stats", headers=hv).json()
    assert st["unsure"] == 2 and st["new"] == 1 and st["total"] == 3 and st["max_id"] == 3 and st["cameras"] == 0
    assert "drink" in c.get("/api/v1/review-reasons", headers=hv).json()
    # решение оператора не сбрасывается продлением события воркером
    env.store.apply_update(au(key="c:1:1", kind="update", start=30, end=60, conf=0.99, cam="pavlovsk-02"), meta())
    assert c.get("/api/v1/alerts/2", headers=hv).json()["status"] == "unsure" and c.get("/api/v1/alerts/2", headers=hv).json()["end_sec"] == 60


def test_journal_csv_and_stats_day(env):
    c, ho = env.client, login(env, "oper1")
    seed(env, 2)
    c.post("/api/v1/alerts/1/review", headers=ho, json=dict(status="confirmed"))
    r = c.get("/api/v1/journal.csv", headers=ho)
    lines = r.text.lstrip("﻿").splitlines()
    assert r.headers["content-type"].startswith("text/csv") and lines[0].startswith("id,t_abs") and len(lines) == 2 and "confirmed" in lines[1]
    d = c.get("/api/v1/stats", headers=ho, params=dict(day="2026-10-09")).json()
    assert d["today"] == {"confirmed": 1, "new": 1} and c.get("/api/v1/stats", headers=ho, params=dict(day="09.10.2026")).status_code == 422


def test_alert_files_are_confined_to_outputs(env, monkeypatch):
    c, h = env.client, login(env, "view1")
    out = env.tmp / "outputs"
    (out / "mon").mkdir(parents=True)
    (out / "mon" / "t.jpg").write_bytes(b"\xff\xd8jpegdata")
    (env.tmp / "secret.txt").write_text("secret")
    monkeypatch.setattr(P, "OUTPUTS", out)
    seed(env, 2)
    env.store.set_artifacts(1, thumb="$OUTPUTS/mon/t.jpg")
    env.store.set_artifacts(2, thumb=str(env.tmp / "secret.txt"))                      # путь вне каталога результатов
    a = c.get("/api/v1/alerts/1", headers=h).json()
    assert a["has_thumb"] is True and a["has_clip"] is False
    r = c.get("/api/v1/alerts/1/thumb", headers=h)
    assert r.status_code == 200 and r.content == b"\xff\xd8jpegdata"
    assert c.get("/api/v1/alerts/1/thumb", headers={**h, "Range": "bytes=0-1"}).status_code == 206
    assert c.get("/api/v1/alerts/2/thumb", headers=h).status_code == 404 and c.get("/api/v1/alerts/2", headers=h).json()["has_thumb"] is False
    assert c.get("/api/v1/alerts/1/clip", headers=h).status_code == 404 and c.get("/api/v1/alerts/1/other", headers=h).status_code == 404


def test_events_stream_reports_new_alerts_and_reviews(env):
    c, h = env.client, login(env, "oper1")
    seed(env, 1)

    def read(**params):
        with c.stream("GET", "/api/v1/events", headers=h, params=dict(timeout=1, interval=0.05, **params)) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            return "".join(r.iter_text())

    body = read(since_alert=0, since_review=0)
    assert "event: ready" in body and "event: alert.new" in body and '"id": 1' in body and '"t_abs": "2026-10-09T10:00:00"' in body          # время — ISO, как во всём API
    assert "alert.new" not in read()                                                  # без курсоров — только новое после подключения
    c.post("/api/v1/alerts/1/review", headers=h, json=dict(status="confirmed", reason=None))
    assert "event: alert.review" in read(since_alert=1, since_review=0)


# ============================================================================================ API: разбор видео и разработчик
def test_analysis_sources_and_error_mapping(env):
    c, h, ho = env.client, login(env, "view1"), login(env, "oper1")
    folders = c.get("/api/v1/sources/folders", headers=h).json()
    cam = next(f for f in folders if f["kind"] == "camera")
    vids = c.get("/api/v1/sources/videos", headers=h, params=dict(folder=cam["id"])).json()
    assert vids and vids[0]["name"] == "c1.mp4"
    assert c.get("/api/v1/sources/videos", headers=h, params=dict(folder="bad")).status_code == 404         # NotFound прежнего модуля → 404
    assert c.get("/api/v1/media/video/bad", headers=h).status_code == 404
    assert c.post("/api/v1/analysis/jobs", headers=h, json=dict(video=vids[0]["id"], profile="mvp")).status_code == 403
    assert c.post("/api/v1/analysis/jobs", headers=ho, json=dict(profile="mvp")).status_code == 422
    assert c.post("/api/v1/sources/roots", headers=login(env, "admin1"), json=dict(path=str(env.tmp))).status_code == 403       # сервер не локальный: папки не подключаются
    assert c.get("/api/v1/analysis/jobs/nope", headers=h).status_code == 404 and c.get("/api/v1/analysis/jobs", headers=h).json() == []
    files = dict(file=("v.mp4", b"x" * 10, "video/mp4"))
    assert c.post("/api/v1/sources/upload", headers=ho, files=files).status_code == 403


def test_analysis_submit_and_job_state(env):
    c, ho = env.client, login(env, "oper1")
    videos = env.app.state.ctx.videos
    videos.run_job = lambda j: None                                                   # без распознавания: проверяется только очередь
    folders = c.get("/api/v1/sources/folders", headers=ho).json()
    vid = c.get("/api/v1/sources/videos", headers=ho, params=dict(folder=folders[0]["id"])).json()[0]["id"]
    r = c.post("/api/v1/analysis/jobs", headers=ho, json=dict(video=vid, profile="mvp"))
    assert r.status_code == 202 and r.json()["video"]
    jid = r.json()["id"]
    assert c.get(f"/api/v1/analysis/jobs/{jid}", headers=ho).json()["id"] == jid
    assert any(x["action"] == "analysis.submit" for x in c.get("/api/v1/audit", headers=login(env, "admin1")).json()["items"])


def test_dev_endpoints(env, monkeypatch):
    from sd import experiments as XP

    monkeypatch.setattr(XP, "EXP_DIR", env.tmp / "experiments")
    c, hd = env.client, login(env, "dev1")
    assert "mvp" in c.get("/api/v1/dev/profiles", headers=hd).json()
    assert c.get("/api/v1/dev/runs", headers=hd).json() == []
    assert c.get("/api/v1/dev/runs/nope", headers=hd).status_code == 404 and c.patch("/api/v1/dev/runs/nope/notes", headers=hd, json=dict(tag="x")).status_code == 404
    d = env.tmp / "experiments" / "r1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(dict(id="r1", name="t")), encoding="utf-8")
    assert c.patch("/api/v1/dev/runs/r1/notes", headers=hd, json=dict(tag="5 к/с", pinned=True)).json() == {"tag": "5 к/с", "pinned": True}
    assert c.get("/api/v1/dev/runs/r1", headers=hd).json()["notes"]["tag"] == "5 к/с"
    assert c.get("/api/v1/dev/runs/..%2F..%2Fetc", headers=hd).status_code == 404
    cat = c.get("/api/v1/dev/tasks/catalog", headers=hd).json()
    assert any(t["id"] == "eval" for t in cat)
    assert c.post("/api/v1/dev/tasks", headers=hd, json=dict(task="rm_rf", values={})).status_code == 422                 # только белый список
    assert c.get("/api/v1/dev/labels/bogus/versions", headers=hd).status_code == 404
    assert isinstance(c.get("/api/v1/dev/labels/events/versions", headers=hd).json(), list)
    assert c.get("/api/v1/dev/profiles/no_such_profile", headers=hd).status_code in (400, 404)
    assert c.get("/api/v1/dev/classifiers/specs/no_such", headers=hd).status_code == 404
    assert isinstance(c.get("/api/v1/dev/releases", headers=hd).json(), list)


def test_profile_save_is_validated_and_audited(env, monkeypatch):
    from sd import profiles as PR

    monkeypatch.setattr(PR, "EXPERIMENTS_DIR", env.tmp / "profiles")
    c, hd = env.client, login(env, "dev1")
    bad = c.put("/api/v1/dev/profiles/t1", headers=hd, json=dict(config={"cycles.no_such_key": 1}))
    assert bad.status_code in (400, 422)                                                                       # опечатка в ключе — ошибка, а не молчание
    assert not (env.tmp / "profiles" / "t1.yaml").exists()


def test_demo_seed_and_clear(env):
    c, ha = env.client, login(env, "admin1")
    r = c.post("/api/v1/demo", headers=ha, json=dict(n=3))
    assert r.status_code == 200 and r.json()["n"] == 3
    assert c.get("/api/v1/alerts", headers=ha, params=dict(demo=True)).json()["total"] == 3
    assert c.post("/api/v1/demo", headers=login(env, "oper1"), json=dict(n=3)).status_code == 403
    assert c.post("/api/v1/demo", headers=ha, json=dict(clear=True)).json()["n"] == 3
    assert c.get("/api/v1/alerts", headers=ha).json()["total"] == 0


# ============================================================================================ хранилище для воркера и перенос
def test_sql_store_lifecycle(tmp_path):
    import pytest as _p

    st = SqlAlertStore(tmp_path / "m.db")
    st.upsert_camera("c", "gatchina", "01", "$ROOT/streams/c", "2026-10-09T10:00:00")
    m = meta()
    aid, k = st.apply_update(au(key="c:1:5.00", cam="c"), m)
    assert k == "open" and aid > 0 and st.max_id() == aid
    st.review(aid, "confirmed", "оператор", "ок")
    aid2, k2 = st.apply_update(au(key="c:1:5.00", kind="update", end=30.0, conf=0.9, cam="c"), m)
    a = st.get(aid)
    assert aid2 == aid and k2 == "update" and a["end_sec"] == 30.0 and a["status"] == "confirmed" and a["confidence"] == 0.9 and a["box"] == [1, 2, 3, 4]
    assert st.apply_update(au(key="zzz", kind="close", cam="c"), m) == (-1, "close")
    st.apply_update(au(key="c:1:5.00", kind="close", cam="c"), m)
    assert st.get(aid)["state"] == "closed" and isinstance(a["t_abs"], str) and a["t_abs"] == "2026-10-09T10:00:05"
    st.apply_update(au(key="c:1:50.00", start=50.0, end=60.0, conf=0.6, cam="c"), {**m, "t_abs": "2026-10-09T10:00:50"})
    s = st.stats("2026-10-09")
    assert (s["new"], s["confirmed"], s["total"], s["precision"]) == (1, 1, 2, 1.0) and s["today"]["new"] == 1 and s["median_review_sec"] is not None
    df = st.list_alerts(status=["new"], districts=["gatchina"], min_conf=0.5)
    assert len(df) == 1 and df.iloc[0]["key"] == "c:1:50.00"
    assert st.list_alerts(since="2026-10-09T10:00:30").shape[0] == 1 and st.list_alerts(newest_first=False).iloc[0]["id"] == aid
    cams = st.cameras()
    assert cams.iloc[0]["alerts"] == 2 and cams.iloc[0]["pending"] == 1 and cams.iloc[0]["start_at"] == "2026-10-09T10:00:00"
    st.mark_chunk("c", "a.mp4", 30.0, 0.0)
    st.mark_chunk("c", "a.mp4", 31.0, 0.0)
    assert st.chunks_done("c") == {"a.mp4": (31.0, 0.0)}
    st.touch_camera("c", "processing", 60.0)
    assert st.cameras().iloc[0]["status"] == "processing" and st.cameras().iloc[0]["processed_sec"] == 60.0
    with _p.raises(ValueError):
        st.review(aid, "maybe")
    with st.db.session() as s2:
        assert len(AS.history(s2, aid)) == 1                                           # решение оператора попало и в историю


def test_store_uses_database_from_env(tmp_path, monkeypatch):
    from sd.server.settings import default_db_url

    monkeypatch.delenv("SD_DB_URL", raising=False)
    assert default_db_url().endswith("platform.db") and "platform" in default_db_url()          # воркер, API и Dev Studio по умолчанию смотрят в один файл
    monkeypatch.setenv("SD_DB_URL", sqlite_url(tmp_path / "b.db"))
    assert default_db_url() == sqlite_url(tmp_path / "b.db") and SqlAlertStore().path == tmp_path / "b.db"
    assert SqlAlertStore(sqlite_url(tmp_path / "c.db")).path == tmp_path / "c.db" and SqlAlertStore(tmp_path / "d.db").path == tmp_path / "d.db"


LEGACY_SCHEMA = """
CREATE TABLE cameras(camera_id TEXT PRIMARY KEY, district TEXT, cam_index TEXT, folder TEXT, start_at TEXT, status TEXT, last_seen TEXT, processed_sec REAL DEFAULT 0);
CREATE TABLE chunks(camera_id TEXT, name TEXT, duration REAL, offset_sec REAL, done_at TEXT, PRIMARY KEY(camera_id, name));
CREATE TABLE alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, camera_id TEXT, district TEXT, cam_index TEXT, chunk TEXT, chunk_offset REAL,
  start_sec REAL, end_sec REAL, peak_sec REAL, t_abs TEXT, t_raised TEXT, confidence REAL, tid INTEGER, box TEXT, rule TEXT, explain TEXT, n_cycles INTEGER,
  state TEXT DEFAULT 'open', status TEXT DEFAULT 'new', reviewer TEXT, reviewed_at TEXT, note TEXT, thumb TEXT, clip TEXT, demo INTEGER DEFAULT 0, fingerprint TEXT,
  created_at TEXT, updated_at TEXT);
"""


def _legacy_db(path):
    """База прежнего формата (`outputs/monitor/monitor.db` до объединения хранилищ): две тревоги, одна уже решена оператором."""
    con = sqlite3.connect(path)
    con.executescript(LEGACY_SCHEMA)
    con.execute("INSERT INTO cameras(camera_id, district, cam_index, folder, start_at, status, processed_sec) VALUES('gatchina-01','gatchina','01','$ROOT/streams/x','2026-10-09T10:00:00','idle',60)")
    con.execute("INSERT INTO chunks VALUES('gatchina-01','a.mp4',30,0,'2026-10-09T10:01:00')")
    con.execute("INSERT INTO chunks VALUES('ghost','b.mp4',30,0,'2026-10-09T10:01:00')")                  # фрагмент камеры без записи о камере — пропускается
    row = ("gatchina-01", "gatchina", "01", "$ROOT/x.mp4", 0.0, 5.0, 20.0, 12.0, "2026-10-09T10:00:05", "2026-10-09T10:00:21", 0.8, 1, "[1, 2, 3, 4]", "two_cycles", "2 цикла", 2, "closed")
    sql = ("INSERT INTO alerts(id, key, camera_id, district, cam_index, chunk, chunk_offset, start_sec, end_sec, peak_sec, t_abs, t_raised, confidence, tid, box, rule, explain, n_cycles, state, "
           "status, reviewer, reviewed_at, note, thumb, created_at, updated_at) VALUES(?, ?, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, ?, ?, ?, ?, ?, ?, ?)")
    con.execute(sql, (1, "k1", *row, "confirmed", "иван", "2026-10-09T10:05:00", "ок", None, "2026-10-09T10:00:21", "2026-10-09T10:05:00"))
    con.execute(sql, (2, "k2", *row, "new", None, None, None, "$OUTPUTS/x/t.jpg", "2026-10-09T10:00:51", "2026-10-09T10:00:51"))
    con.commit()
    con.close()


def test_import_legacy_database(tmp_path):
    from sd.server.legacy import import_legacy
    from sd.server.store import ensure_schema

    _legacy_db(tmp_path / "monitor.db")
    db = Database(sqlite_url(tmp_path / "new.db"))
    ensure_schema(db)
    r = import_legacy(db, tmp_path / "monitor.db")
    assert (r["cameras"], r["chunks"], r["alerts"], r["reviews"]) == (1, 1, 2, 1)
    st = SqlAlertStore(db, migrate=False)
    assert st.get(1)["status"] == "confirmed" and st.get(1)["reviewer"] == "иван" and st.get(2)["thumb"] == "$OUTPUTS/x/t.jpg" and st.get(2)["box"] == [1, 2, 3, 4]
    assert import_legacy(db, tmp_path / "monitor.db")["alerts"] == 0                    # повторный перенос ничего не дублирует
    with db.session() as s:
        assert [x.reviewer_name for x in AS.history(s, 1)] == ["иван"]
    with pytest.raises(FileNotFoundError):
        import_legacy(db, tmp_path / "none.db")
    con = sqlite3.connect(tmp_path / "monitor.db")
    assert con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 2                 # исходная база не изменена
    con.close()


# ============================================================================================ CLI
def test_cli_db_and_user_commands(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from sd.cli import app

    monkeypatch.setenv("SD_DB_URL", sqlite_url(tmp_path / "cli.db"))
    monkeypatch.setenv("SD_SECRET_KEY", "k" * 40)
    run = CliRunner().invoke
    r = run(app, ["db", "current"])
    assert r.exit_code == 0 and "миграции не применялись" in r.output
    assert run(app, ["db", "upgrade"]).exit_code == 0 and "актуальна" in run(app, ["db", "current"]).output
    assert "0002_monitoring" in run(app, ["db", "history"]).output
    r = run(app, ["user", "create", "ivan", "--role", "operator", "--password", PW])
    assert r.exit_code == 0 and "создан: ivan" in r.output
    assert run(app, ["user", "create", "ivan", "--password", PW]).exit_code == 2
    assert run(app, ["user", "create", "weakone", "--password", "weak"]).exit_code == 2
    assert "ivan" in run(app, ["user", "list"]).output
    assert run(app, ["user", "set", "ivan", "--role", "developer"]).exit_code == 0 and run(app, ["user", "set", "nobody", "--role", "admin"]).exit_code == 2
    tok = run(app, ["user", "token", "ivan"]).output.strip()
    assert tok.startswith("sdt_")
    assert run(app, ["db", "downgrade", "0001_identity", "--yes"]).exit_code == 0
    assert run(app, ["db", "import-legacy", str(tmp_path / "none.db")]).exit_code == 2
    assert run(app, ["serve", "--host", "0.0.0.0", "--no-auth"]).exit_code == 2         # вход нельзя отключить при доступе из сети
    monkeypatch.setenv("SD_AUTH", "off")
    assert run(app, ["serve", "--host", "0.0.0.0"]).exit_code == 2                      # …в том числе переменной окружения


def test_cli_app_and_serve_modes(tmp_path, monkeypatch):
    """`sd app` — приложение «для себя» (порт 8502, вход отключён, файлы разрешены), на сетевом адресе вход включается сам; `sd serve` — вход всегда."""
    import uvicorn
    from typer.testing import CliRunner

    from sd.cli import app

    monkeypatch.setenv("SD_DB_URL", sqlite_url(tmp_path / "a.db"))
    monkeypatch.setenv("SD_SECRET_KEY", "k" * 40)
    monkeypatch.delenv("SD_AUTH", raising=False)
    monkeypatch.delenv("SD_UI", raising=False)
    seen: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda application, **kw: seen.update(app=application, **kw))

    def launch(*args):
        seen.clear()
        r = CliRunner().invoke(app, list(args))
        return r, (seen["app"].state.ctx.settings if "app" in seen else None)

    r, st = launch("app")
    assert r.exit_code == 0 and seen["port"] == 8502 and seen["host"] == "127.0.0.1" and "/app/" in r.output
    assert st.auth_required is False and st.allow_fs is True and st.serve_ui is True
    r, st = launch("app", "--login")
    assert r.exit_code == 0 and st.auth_required is True and st.allow_fs is True
    r, st = launch("app", "--host", "0.0.0.0", "--port", "9000")
    assert r.exit_code == 0 and seen["port"] == 9000 and st.auth_required is True and st.allow_fs is False       # из сети: вход включён, загрузка файлов отключена
    r, st = launch("serve")
    assert r.exit_code == 0 and seen["port"] == 8503 and st.auth_required is True
    r, st = launch("serve", "--no-auth")
    assert r.exit_code == 0 and st.auth_required is False
    monkeypatch.setenv("SD_UI", "0")
    r, st = launch("app")
    assert r.exit_code == 2 and st is None and "SD_UI" in r.output                                           # приложение без интерфейса бессмысленно: подсказывает `sd serve`
    r, st = launch("serve")
    assert r.exit_code == 0 and st.serve_ui is False


def test_migrations_render_for_postgresql():
    """Те же миграции описываются корректным SQL для PostgreSQL (без драйвера и сервера: автономный режим Alembic)."""
    import io

    from alembic import command

    cfg = M.config("postgresql://u:p@localhost/db")
    cfg.output_buffer = io.StringIO()
    command.upgrade(cfg, "head", sql=True)
    sql = cfg.output_buffer.getvalue()
    assert "CREATE TABLE alerts" in sql and "CREATE TABLE alert_reviews" in sql and "CONSTRAINT ck_alerts_status CHECK" in sql and "INSERT INTO alembic_version" in sql
    cfg2 = M.config("postgresql://u:p@localhost/db")
    cfg2.output_buffer = io.StringIO()
    command.downgrade(cfg2, "0002_monitoring:base", sql=True)
    assert "DROP TABLE alerts" in cfg2.output_buffer.getvalue()


def test_worker_writes_to_platform_db_and_api_sees_it(env):
    """Воркер (`realtime.worker`) работает с `SqlAlertStore` без правок; то, что он записал, видно через API."""
    pytest.importorskip("cv2")
    from test_realtime_pipeline import G2, PROF, fake_engine_factory, make_stream, synthetic_frames

    from sd.realtime import sources as S
    from sd.realtime import worker as W

    streams = make_stream(env.tmp / "w").parent
    stream = S.scan_streams(streams)[0]
    eng = fake_engine_factory(G2)(PROF, stream.camera_id, 10.0)
    r = W.process_stream(stream, env.store, eng, PROF, frames_fn=lambda c, o, f: synthetic_frames(c, o, f), probe_fn=lambda c: SimpleNamespace(duration=30.0),
                         artifacts_dir=env.tmp / "art", log=lambda *a: None)
    assert r["chunks"] == 2 and r["alerts"] == 1
    h = login(env, "view1")
    al = env.client.get("/api/v1/alerts", headers=h).json()
    assert al["total"] == 1 and al["items"][0]["camera_id"] == "gatchina-03" and al["items"][0]["state"] == "closed" and al["items"][0]["fingerprint"] == PROF.fingerprint()
    cam = env.client.get("/api/v1/cameras", headers=h).json()[0]
    assert cam["camera"]["camera_id"] == "gatchina-03" and cam["camera"]["processed_sec"] == 60.0 and cam["camera"]["status"] == "idle" and cam["alerts"] == 1
    assert env.store.chunks_done("gatchina-03").keys() == {"c1.mp4", "c2.mp4"}
