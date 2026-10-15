"""Веб-интерфейс платформы (`sd/server/static`): раздача и заголовки, целостность модулей, соответствие вызовов API реальным маршрутам, безопасность кода, сводка отчётов."""
import re
import shutil
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient                       # noqa: E402

from sd.server import clock                                      # noqa: E402
from sd.server.app import STATIC_DIR, create_app                 # noqa: E402
from sd.server.db import Database                                # noqa: E402
from sd.server.services import alerts as AS, reports as RS, users as US     # noqa: E402
from sd.server.settings import Settings, sqlite_url              # noqa: E402
from sd.server.store import SqlAlertStore                        # noqa: E402

from tests.test_server import PW, au, login, meta               # noqa: E402,F401  (фикстура env подхватывается ниже)
from tests.test_server import env                                # noqa: E402,F401

JS = STATIC_DIR / "js"
SOURCES = sorted(JS.rglob("*.js"))
NODE = shutil.which("node")


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ============================================================================================ раздача
def test_ui_is_served_with_right_types_and_headers(env):
    c = env.client
    r = c.get("/app/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in r.headers["content-security-policy"] and "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-cache" and r.headers["x-content-type-options"] == "nosniff"
    assert 'src="js/main.js"' in r.text and 'href="css/app.css"' in r.text

    js = c.get("/app/js/main.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]       # на Windows реестр мог бы отдать text/plain — браузер не выполнит модуль
    assert c.get("/app/css/app.css").headers["content-type"].startswith("text/css")
    font = c.get("/app/fonts/Geologica-300-700-cyrillic.woff2")
    assert font.status_code == 200 and font.headers["content-type"] == "font/woff2" and font.content[:4] == b"wOF2"
    assert "max-age" in font.headers["cache-control"]
    assert c.get("/app/js/нет_такого.js").status_code == 404
    assert c.get("/app/%2e%2e/%2e%2e/pyproject.toml").status_code in (400, 404)     # выход из каталога
    assert c.get("/api/v1/health").headers["cache-control"] == "no-store"            # политика интерфейса не задевает API


def test_root_redirects_to_ui_and_ui_can_be_disabled(env, tmp_path):
    r = env.client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/app/"
    st = Settings(db_url=sqlite_url(tmp_path / "x.db"), secret_key="k" * 40, serve_ui=False)
    with TestClient(create_app(st, db=Database(st.db_url))) as c:
        assert c.get("/app/").status_code == 404
        assert c.get("/", follow_redirects=False).headers["location"] == "/docs"


def test_ui_files_are_in_package_data():
    cfg = read(Path(__file__).resolve().parents[1] / "pyproject.toml")
    assert "server/static/**/*" in cfg


# ============================================================================================ целостность модулей
IMPORT_RE = re.compile(r"""(?:from\s+|import\s*\(\s*|import\s+)["'](\.{1,2}/[^"']+)["']""")


def test_every_import_resolves_and_screens_export_default():
    assert len(SOURCES) >= 25
    for f in SOURCES:
        for rel in IMPORT_RE.findall(read(f)):
            assert (f.parent / rel).resolve().is_file(), f"{f.relative_to(STATIC_DIR)}: нет файла {rel}"
    reg = read(JS / "registry.js")
    screens = re.findall(r'import\("\./(screens/[\w]+\.js)"\)', reg)
    assert len(screens) == len(set(screens)) >= 11
    for s in screens:
        assert re.search(r"export default (async )?function", read(JS / s)), f"{s}: нет экспорта по умолчанию"


def test_registry_paths_and_roles_are_consistent():
    reg = read(JS / "registry.js")
    ids = re.findall(r'\{ id: "(\w+)", path: "([^"]+)", title: "[^"]+", icon: "(\w+)", role: "(\w+)"', reg)
    assert len(ids) == 11 and len({i[0] for i in ids}) == 11 and len({i[1] for i in ids}) == 11
    assert {i[3] for i in ids} <= {"viewer", "operator", "developer", "admin"}
    icons = set(re.findall(r"^  (\w+): \"", read(JS / "dom.js"), re.M))
    assert {i[2] for i in ids} <= icons, "значок экрана не определён в dom.js"


def test_all_used_icons_exist():
    icons = set(re.findall(r"^  (\w+): \"", read(JS / "dom.js"), re.M))
    assert len(icons) >= 35
    used = set()
    for f in SOURCES:
        txt = read(f)
        used |= set(re.findall(r"""\bicon\(\s*["'](\w+)["']""", txt)) | set(re.findall(r"""\bic:\s*["'](\w+)["']""", txt)) | set(re.findall(r"""\bnavIcon:\s*["'](\w+)["']""", txt))
        used |= set(re.findall(r"""\bicon\s*:\s*["'](\w+)["']""", txt))
    missing = used - icons
    assert not missing, f"значки без определения: {sorted(missing)}"


def test_no_external_requests_and_no_unsafe_dom_apis():
    allowed = ("http://www.w3.org/2000/svg",)
    for f in [*SOURCES, STATIC_DIR / "index.html", *STATIC_DIR.rglob("*.css")]:
        txt = read(f)
        for url in re.findall(r"https?://[^\s\"'`)<>]+", txt):
            assert url.startswith(allowed), f"{f.relative_to(STATIC_DIR)}: внешний адрес {url}"
    forbidden = re.compile(r"\b(innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\s*\(|new Function\s*\()|dangerouslySetInnerHTML")
    for f in SOURCES:
        m = forbidden.search(re.sub(r"(?<![:\"'])//.*", "", read(f)))        # комментарии не считаются
        assert not m, f"{f.relative_to(STATIC_DIR)}: небезопасный вызов {m.group(0)}"
    css = read(STATIC_DIR / "css" / "fonts.css")
    for name in re.findall(r'url\("\.\./fonts/([^"]+)"\)', css):
        assert (STATIC_DIR / "fonts" / name).is_file(), f"нет шрифта {name}"
    assert (STATIC_DIR / "fonts" / "README.txt").exists()                  # лицензии шрифтов лежат рядом с ними


def _top_level_args(src: str, open_idx: int) -> list[str]:
    """Аргументы вызова верхнего уровня: open_idx — позиция «(» (строки, шаблоны и скобки внутри пропускаются)."""
    args, depth, cur, i, n = [], 0, [], open_idx, len(src)
    while i < n:
        ch = src[i]
        if ch in "\"'`":
            q, j = ch, i + 1
            while j < n and src[j] != q:
                if src[j] == "\\":
                    j += 1
                elif q == "`" and src[j:j + 2] == "${":
                    d, j = 1, j + 2
                    while j < n and d:
                        d += {"{": 1, "}": -1}.get(src[j], 0)
                        j += 1
                    continue
                j += 1
            cur.append(src[i:j + 1])
            i = j + 1
            continue
        if ch in "([{":
            depth += 1
            if depth > 1:
                cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip())
                return args
            cur.append(ch)
        elif ch == "," and depth == 1:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    return args


def _flatten(arg: str) -> str:
    """Выражение без содержимого скобок и строк: остаётся только верхний уровень («cond ? () : null»)."""
    out, depth, i, n = [], 0, 0, len(arg)
    while i < n:
        ch = arg[i]
        if ch in "\"'`":
            j = i + 1
            while j < n and arg[j] != ch:
                j += 2 if arg[j] == "\\" else 1
            i = j + 1
            if depth == 0:
                out.append("''")
            continue
        if ch in "([{":
            depth += 1
            if depth == 1:
                out.append("()")
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            out.append(ch)
        i += 1
    return " ".join("".join(out).split())


def test_no_native_append_receives_null():
    """Node.append(null) вставляет на страницу текст «null»: в прямых вызовах .append допустимы только узлы. Условные аргументы допустимы внутри h()/s()."""
    cond_null = re.compile(r"\?.*:\s*(null|undefined|false)$|^(null|undefined|false)$|&&")
    checked = 0
    for f in SOURCES:
        if f.name == "dom.js":
            continue
        src = read(f)
        for m in re.finditer(r"\.append\(", src):
            line = src.count("\n", 0, m.start()) + 1
            for arg in _top_level_args(src, m.end() - 1):
                checked += 1
                assert not cond_null.search(_flatten(arg)), f"{f.relative_to(STATIC_DIR)}:{line}: append с условным null/false: {arg[:90]}"
    assert checked > 150


# ============================================================================================ синтаксис и чистая логика (нужен Node)
@pytest.mark.skipif(NODE is None, reason="Node.js не установлен")
def test_all_modules_parse():
    with tempfile.TemporaryDirectory() as d:
        for f in SOURCES:
            tmp = Path(d) / (f.relative_to(JS).as_posix().replace("/", "_") + ".mjs")
            tmp.write_text(read(f), encoding="utf-8")
            r = subprocess.run([NODE, "--check", str(tmp)], capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, f"{f.relative_to(STATIC_DIR)}:\n{r.stderr[:600]}"


@pytest.mark.skipif(NODE is None, reason="Node.js не установлен")
def test_pure_helpers_in_node():
    """Форматирование, склонения, CSV и маршрутизатор — без браузера: модули копируются как .mjs с исправленными путями."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name in ("dom", "bus", "files", "router"):
            src = read(JS / f"{name}.js")
            (d / f"{name}.mjs").write_text(re.sub(r'from "\./(\w+)\.js"', r'from "./\1.mjs"', src), encoding="utf-8")
        (d / "t.mjs").write_text('''
import { plural, pct, num, dur, clock, initials, fmtTime, ago } from "./dom.mjs";
import { csvCell, toCsv } from "./files.mjs";
import { match, parse, href } from "./router.mjs";
const eq = (a, b, m) => { if (JSON.stringify(a) !== JSON.stringify(b)) { console.error("FAIL", m, JSON.stringify(a), "!=", JSON.stringify(b)); process.exit(1); } };
eq([plural(1, "тревога", "тревоги", "тревог"), plural(2, "тревога", "тревоги", "тревог"), plural(5, "тревога", "тревоги", "тревог"), plural(11, "тревога", "тревоги", "тревог"), plural(21, "тревога", "тревоги", "тревог"), plural(112, "тревога", "тревоги", "тревог")],
   ["тревога", "тревоги", "тревог", "тревог", "тревога", "тревог"], "plural");
eq([pct(0.456), pct(null), pct(1), pct(0.456, 1)], ["46%", "—", "100%", "45.6%"], "pct");
eq([num(0.5), num(null), num(1.234, 1)], ["0,50", "—", "1,2"], "num");
eq([dur(30), dur(120), dur(7200), dur(null)], ["30 с", "2 мин", "2,0 ч", "—"], "dur");
eq([clock(0), clock(65), clock(3725)], ["00:00", "01:05", "62:05"], "clock");
eq([initials("Иван Петров"), initials(""), initials("admin")], ["ИП", "?", "A"], "initials");
eq(fmtTime("2026-10-09T10:05:30"), "10:05", "fmtTime");
eq(ago("2026-10-09T10:00:00", new Date("2026-10-09T10:00:20")), "только что", "ago1");
eq(ago("2026-10-09T10:00:00", new Date("2026-10-09T10:07:00")), "7 мин назад", "ago2");
eq(csvCell('a,b'), '"a,b"', "csv comma"); eq(csvCell('say "hi"'), '"say ""hi"""', "csv quote");
eq(csvCell("=SUM(A1)"), "'=SUM(A1)", "csv formula"); eq(csvCell("-5"), "-5", "csv negative number"); eq(csvCell(null), "", "csv null");
eq(toCsv([["a", 1], ["б", "x;y"]]), "\\uFEFFa,1\\r\\n" + 'б,"x;y"' + "\\r\\n", "toCsv");
const screens = [{ path: "/", id: "o" }, { path: "/review/:id?", id: "r" }, { path: "/map", id: "m" }];
eq(match(screens, { path: "/", query: {} }).screen.id, "o", "route root");
eq(match(screens, { path: "/review", query: {} }).params, { id: undefined }, "route opt none");
eq(match(screens, { path: "/review/42", query: { a: "1" } }).params, { id: "42" }, "route param");
eq(match(screens, { path: "/review/42/", query: {} }).screen.id, "r", "trailing slash");
eq(match(screens, { path: "/nope", query: {} }), null, "route miss");
eq(parse("#/review/7?x=1&y=%D0%B0"), { path: "/review/7", query: { x: "1", y: "а" } }, "parse");
eq(href("/alerts", { camera: "c1", min: "", z: undefined }), "#/alerts?camera=c1", "href skips empty");
console.log("ok");
''', encoding="utf-8")
        r = subprocess.run([NODE, str(d / "t.mjs")], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr or r.stdout


# ============================================================================================ контракт с API
API_CALL = re.compile(r"""\bapi\.(get|post|put|patch|del|upload)\(\s*([`"'])(/[^`"']*)\2""")
METHOD = {"get": "get", "post": "post", "put": "put", "patch": "patch", "del": "delete", "upload": "post"}


def _norm(path: str) -> str:
    path = re.sub(r"\$\{[^}]*\}", "{}", path)
    return re.sub(r"\{[^}]*\}", "{}", path)


def test_every_api_call_in_ui_matches_a_real_route(env):
    spec = env.client.get("/api/v1/openapi.json").json()["paths"]
    routes = {}
    for p, ops in spec.items():
        routes.setdefault(_norm(p.removeprefix("/api/v1") if p.startswith("/api/v1") else p), set()).update(ops)
    calls = []
    for f in SOURCES:
        for method, _q, path in API_CALL.findall(read(f)):
            calls.append((f.name, method, _norm(path)))
    assert len(calls) > 60
    missing = [(f, m, p) for f, m, p in calls if p not in routes or METHOD[m] not in routes[p]]
    assert not missing, f"вызовы без маршрута: {missing}"
    # SSE и медиа адресуются строками, а не через api.*: проверим, что маршруты на месте
    for p in ("/events", "/media/status", "/media/alert/{}/clip", "/media/video/{}", "/alerts/{}/{}"):
        assert p in routes or any(_norm(k) == p for k in routes), p


# ============================================================================================ сводка отчётов
def _seed(env, now):
    for i, (cam, hours_ago, conf) in enumerate([("gatchina-01", 1, 0.9), ("gatchina-01", 2, 0.6), ("gatchina-02", 30, 0.8), ("gatchina-02", 55, 0.5), ("pavlovsk-01", 100, 0.7)]):
        t = (now - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
        env.store.apply_update(au(key=f"c:{i}:5.00", conf=conf, cam=cam), meta(t, district=cam.split("-")[0]))


def test_reports_summary_counts_reasons_cameras_and_hours(env):
    _seed(env, clock.now() - timedelta(minutes=5))
    h = login(env, "oper1")
    items = env.client.get("/api/v1/alerts?limit=50", headers=h).json()["items"]
    assert len(items) == 5
    by_conf = sorted(items, key=lambda a: -a["confidence"])
    for a, body in zip(by_conf, (dict(status="confirmed"), dict(status="false", reason="drink"), dict(status="false", reason="phone"), dict(status="unsure"))):
        assert env.client.post(f"/api/v1/alerts/{a['id']}/review", json=body, headers=h).status_code == 200

    r = env.client.get("/api/v1/reports/summary?days=7", headers=login(env, "view1"))        # смотреть отчёты может любой вошедший
    assert r.status_code == 200
    d = r.json()
    t = d["totals"]
    assert t["total"] == 5 and (t["confirmed"], t["false"], t["unsure"], t["new"]) == (1, 2, 1, 1)
    assert t["precision"] == pytest.approx(1 / 3) and t["mean_confidence"] == pytest.approx((0.9 + 0.6 + 0.8 + 0.5 + 0.7) / 5)
    assert len(d["series"]) == 7 and sum(x["total"] for x in d["series"]) == 5 and d["bucket"] == "day" and not d["truncated"]
    assert {x["reason"]: x["n"] for x in d["reasons"]} == {"drink": 1, "phone": 1}
    cams = {c["camera_id"]: c for c in d["cameras"]}
    assert cams["gatchina-01"]["total"] == 2 and cams["gatchina-02"]["total"] == 2 and cams["pavlovsk-01"]["total"] == 1
    assert [c["total"] for c in d["cameras"]] == sorted((c["total"] for c in d["cameras"]), reverse=True)
    assert len(d["hours"]) == 24 and sum(d["hours"]) == 5

    one = env.client.get("/api/v1/reports/summary?days=7&camera=gatchina-02", headers=h).json()
    assert one["totals"]["total"] == 2 and [c["camera_id"] for c in one["cameras"]] == ["gatchina-02"]
    assert env.client.get("/api/v1/reports/summary?days=7&district=pavlovsk", headers=h).json()["totals"]["total"] == 1
    hourly = env.client.get("/api/v1/reports/summary?days=1&bucket=hour", headers=h).json()
    assert hourly["bucket"] == "hour" and 1 <= len(hourly["series"]) <= 25 and all(len(x["t"]) == 16 for x in hourly["series"])


def test_reports_windows_by_day_and_hour_are_exact(env):
    now = clock.parse("2026-10-14T12:00:00")
    _seed(env, now)                                       # 11:00 и 10:00 14-го, 06:00 13-го, 05:00 12-го, 08:00 10-го
    with env.db.session() as s:
        day = RS.summary(s, days=2, now=now)
        hour = RS.summary(s, days=2, bucket="hour", now=now)
        today = RS.summary(s, days=1, now=now)
    assert [x["t"] for x in day["series"]] == ["2026-10-13", "2026-10-14"] and [x["total"] for x in day["series"]] == [1, 2]
    assert day["totals"]["total"] == 3 and today["totals"]["total"] == 2
    assert len(hour["series"]) == 37 and sum(x["total"] for x in hour["series"]) == 3 and hour["series"][0]["t"] == "2026-10-13T00:00"
    assert day["since"] == "2026-10-13T00:00:00" and day["until"] == "2026-10-14T12:00:00"


def test_reports_series_keeps_alerts_from_a_fast_camera_clock(env):
    now = clock.parse("2026-10-14T12:00:00")
    env.store.apply_update(au(key="c:f:1", cam="gatchina-01"), meta("2026-10-14T13:30:00"))        # часы камеры на полтора часа впереди
    with env.db.session() as s:
        d = RS.summary(s, days=1, bucket="hour", now=now)
    assert d["totals"]["total"] == 1 and sum(x["total"] for x in d["series"]) == 1 and d["series"][-1]["t"] == "2026-10-14T13:00"


def test_reports_summary_validation_and_access(env):
    h = login(env, "view1")
    for q in ("days=0", "days=400", "bucket=week", "days=30&bucket=hour"):
        assert env.client.get(f"/api/v1/reports/summary?{q}", headers=h).status_code in (400, 422), q
    assert env.client.get("/api/v1/reports/summary").status_code == 401
    empty = env.client.get("/api/v1/reports/summary?days=3", headers=h).json()
    assert empty["totals"]["total"] == 0 and empty["totals"]["precision"] is None and empty["cameras"] == [] and empty["reasons"] == [] and len(empty["series"]) == 3


def test_reports_service_truncates_instead_of_failing(env, monkeypatch):
    _seed(env, clock.now() - timedelta(minutes=5))
    monkeypatch.setattr(RS, "MAX_ROWS", 3)
    with env.db.session() as s:
        d = RS.summary(s, days=7)
    assert d["truncated"] is True and d["totals"]["total"] == 3
