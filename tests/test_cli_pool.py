"""Команды работы с пулом кандидатов: понятная ошибка вместо трассировки, если предыдущий шаг не выполнен (порядок: start-eval --rebuild → pool-features → compare-sets/train-bundle --pool)."""
import pytest
from typer.testing import CliRunner

from sd import cli
from sd import pool_features as PFt
from sd import start_eval as SE

runner = CliRunner()


def _plain(res) -> str:
    return " ".join(res.output.split())            # rich переносит строки по ширине терминала


def test_pool_features_asks_for_pool_first(tmp_path, monkeypatch):
    monkeypatch.setattr(SE, "POOL", tmp_path / "gesture_pool.parquet")
    res = runner.invoke(cli.app, ["pool-features", "--objects", ""])
    assert res.exit_code == 1
    assert "sd start-eval --rebuild" in _plain(res)


@pytest.mark.parametrize("args", [["compare-sets", "--pool"], ["train-bundle", "tmp_bundle_x", "--pool"]])
def test_pool_modes_ask_for_pool_features_first(tmp_path, monkeypatch, args):
    monkeypatch.setattr(PFt, "POOL_AN", tmp_path / "analysis_pool")                  # таблиц признаков пула нет
    monkeypatch.setattr(cli, "ROOT", tmp_path)                                       # относительный путь в сообщении
    res = runner.invoke(cli.app, args)
    assert res.exit_code == 1
    assert "sd pool-features" in _plain(res)
