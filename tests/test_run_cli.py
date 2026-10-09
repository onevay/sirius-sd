import json
from types import SimpleNamespace

import pandas as pd

from sd import profiles as PR
from sd import run as RUN
from sd import runner as R


def test_sd_run_writes_preds_above_threshold_and_manifest(tmp_path, monkeypatch):
    d = tmp_path / "cam1"
    d.mkdir()
    (d / "a.mp4").write_bytes(b"x")
    (d / "b.mp4").write_bytes(b"x")

    def fake(video, start, end, profile):
        n = video.stem
        rows = [dict(camera_id="c", clip_id=n, event_id="e", start_sec=1, end_sec=9, confidence=0.9 if n == "a" else 0.3, label="smoking_like", person_track_id=1, peak_sec=5,
                     x1=0, y1=0, x2=10, y2=20)]
        return SimpleNamespace(events=pd.DataFrame(rows, columns=R.EVENT_COLUMNS), meta=dict(run_dir=None, out_dir=None, counts=dict(cycles=1, people=1), warnings=[]))

    monkeypatch.setattr(R, "_default_recognize", fake)
    monkeypatch.setattr(R, "_default_probe", lambda v: SimpleNamespace(duration=30.0))
    monkeypatch.setattr(R, "CACHE_DIR", tmp_path / "cache")
    out = tmp_path / "o" / "preds.csv"
    assert RUN.main(["--input", str(d), "--out", str(out), "--threshold", "0.5"]) == 2          # без классификатора цикла запуск отклоняется
    pf = PR.save(PR.heuristic_profile(), tmp_path / "prof")
    assert RUN.main(["--input", str(d), "--out", str(out), "--threshold", "0.5", "--profile", str(pf)]) == 0
    df = pd.read_csv(out)
    assert df.clip_id.tolist() == ["a"] and df.camera_id.tolist() == ["cam1"] and list(df.columns) == R.EVENT_COLUMNS
    assert len(pd.read_csv(out.with_suffix(".all.csv"))) == 2
    man = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert man["confidence_threshold"] == 0.5 and man["events_written"] == 1 and man["fingerprint"] == PR.heuristic_profile().fingerprint() and not man["errors"]
    assert RUN.main(["--input", str(tmp_path / "пусто"), "--out", str(out)]) == 2
