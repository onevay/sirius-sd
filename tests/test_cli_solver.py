import json

import pytest
from typer.testing import CliRunner

from rt_helpers import make_bundle
from sd import cli
from sd import profiles as PR
from sd import solver as SV

FAST = ["hold", "d_min", "mouth_dur", "p_wm_dist"]
runner = CliRunner()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    pdir = tmp_path / "profiles"
    monkeypatch.setattr(PR, "EXPERIMENTS_DIR", pdir)
    b = make_bundle(tmp_path / "cycle" / "fast1", FAST)
    PR.save(PR.Profile("good", "ок", config={"events.confidence_threshold": 0.6}, options={"cycle_bundle": str(b)}))
    PR.save(PR.Profile("nocls", "без классификатора"))
    PR.save(PR.Profile("heavy", options={"cycle_bundle": str(make_bundle(tmp_path / "cycle" / "heavy", FAST + ["obj_d_max_conf"], seed=2)), "objects": ["d"]}))
    return tmp_path


def run(*args):
    r = runner.invoke(cli.app, list(args), catch_exceptions=False)
    return r.exit_code, r.output


def test_check_command_reports_by_mode(home):
    assert run("solver", "check", "-p", "good")[0] == 0
    code, out = run("solver", "check", "-p", "nocls")
    assert code == 1 and "обязателен" in out
    assert run("solver", "check", "-p", "heavy")[0] == 0 and run("solver", "check", "-p", "heavy", "--mode", "replay")[0] == 0
    code, out = run("solver", "check", "-p", "heavy", "--mode", "live")
    assert code == 1 and "настоящем потоке" in out


def test_pack_inspect_json_and_refusal(home):
    out_zip = home / "o" / "good.sdsolver.zip"
    code, out = run("solver", "pack", "-p", "good", "-o", str(out_zip))
    assert code == 0 and out_zip.exists() and "classifier" in out
    code, out = run("solver", "pack", "-p", "nocls", "-o", str(home / "o" / "x.sdsolver.zip"))
    assert code == 1 and not (home / "o" / "x.sdsolver.zip").exists()
    code, out = run("solver", "inspect", str(out_zip), "--json")
    assert code == 0 and json.loads(out)["architecture"]["mode"] == "classifier"
    code, out = run("solver", "inspect", str(out_zip))
    assert code == 0 and "cycle_classifier" in out and "ОК" in out


def test_pack_can_override_classifier(home):
    other = make_bundle(home / "cycle" / "fast9", FAST, seed=9)
    out_zip = home / "o" / "n.sdsolver.zip"
    assert run("solver", "pack", "-p", "nocls", "--classifier", str(other), "-o", str(out_zip))[0] == 0
    assert SV.inspect(out_zip)["meta"]["profile"]["options"]["cycle_bundle"] == "models/cycle/fast9"


def test_eval_refuses_without_classifier_and_accepts_override(home, monkeypatch):
    monkeypatch.setattr(SV, "_device_issues", lambda p: [])          # устройства тестовой машины не важны
    d = home / "курение"
    d.mkdir()
    code, out = run("eval", "-d", str(d), "-p", "nocls")
    assert code == 1 and "обязателен" in out
    code, out = run("eval", "-d", str(d), "-p", "good")           # классификатор есть, но нет видео — другая, содержательная ошибка
    assert code == 1 and "нет видео" in out


def test_run_command_refuses_without_classifier(home, capsys):
    from sd import run as RUN

    d = home / "clips"
    d.mkdir()
    (d / "a.mp4").write_bytes(b"x")
    assert RUN.main(["--input", str(d), "--out", str(home / "p.csv"), "--profile", "nocls"]) == 2
    assert "обязателен" in capsys.readouterr().err


def test_monitor_refuses_heavy_classifier_in_live_mode(home):
    code, out = run("monitor", "--streams", str(home / "s"), "-p", "heavy")
    assert code == 1 and "настоящем потоке" in out
    code, out = run("monitor", "--streams", str(home / "s"), "-p", "nocls")
    assert code == 1 and "обязателен" in out
