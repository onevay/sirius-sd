"""Страница «Анализ»: запись правок меток жестов — копия прежнего файла, атомарность, сохранение столбцов."""
import pandas as pd

from sd import start_eval as SE
from sd.ui import page_analysis as PA


def test_save_gestures_keeps_backup_and_columns(tmp_path, monkeypatch):
    p = tmp_path / "gesture_gt.csv"
    monkeypatch.setattr(SE, "GT_CSV", p)
    monkeypatch.setattr(PA, "OUTPUTS", tmp_path / "out")
    monkeypatch.setattr(PA, "ROOT", tmp_path)
    old = pd.DataFrame([dict(video="v", tid=1, peak_t=2.0, start=1.0, label="unsure", start_quality=None, note="", source="pool", reviewer="assistant_visual")])
    old.to_csv(p, index=False, encoding="utf-8-sig")
    new = old.assign(label="smoke", note="исправил вручную")
    b = PA._save_gestures(new)
    got = pd.read_csv(p, encoding="utf-8-sig")
    assert list(got.columns) == PA.GT_COLS and got.label.iloc[0] == "smoke"
    assert b is not None and pd.read_csv(b, encoding="utf-8-sig").label.iloc[0] == "unsure"        # копия содержит прежнее значение
    assert not p.with_suffix(".tmp").exists()
