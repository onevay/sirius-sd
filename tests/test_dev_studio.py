"""Бэкенд Dev Studio: версии разметки, метки жестов, защита от утечки обучения, анализ кэша, PDF-отчёт, ворота выпуска, каталог данных, переносимые пути."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sd import bundle as B
from sd import cache_analysis as CA
from sd import data_catalog as DC
from sd import experiments as XP
from sd import gestures as G
from sd import gt as GT
from sd import labels_history as LH
from sd import profiles as PR
from sd import runner as R
from sd.paths import from_portable, portable, video_id
from test_runner import Fake, _gt, make_world, probe


# ---------------------------------------------------------------------------------------------- пути
def test_portable_is_idempotent_and_old_double_token_resolves():
    s = portable(Path.cwd() / "outputs" / "x")
    assert portable(s) == s                                         # повторный вызов не даёт «$ROOT/$OUTPUTS/…»
    assert from_portable("$ROOT/$OUTPUTS/recognize/a") == from_portable("$OUTPUTS/recognize/a")


def test_load_named_rejects_paths(tmp_path):
    PR.save(PR.Profile("p1", config={"video.process_fps": 5}), tmp_path)
    assert PR.load_named("p1", tmp_path).config["video.process_fps"] == 5
    for bad in ("p1.yaml", "..\\x", "../x", "x.yml", "a.b"):
        with pytest.raises((ValueError, FileNotFoundError)):
            PR.load_named(bad, tmp_path)
    with pytest.raises(FileNotFoundError):
        PR.load_named("nope", tmp_path)


# ---------------------------------------------------------------------------------------------- версии разметки
def test_snapshot_dedup_restore_and_diff(tmp_path):
    f, hist = tmp_path / "events_gt.csv", tmp_path / "hist"
    GT.add("c1", 1, 5, "POSITIVE", person=1, box=(0, 0, 10, 20), path=f)
    r1 = LH.snapshot(f, "ann", "первая", "manual", root=hist)
    assert r1 and r1["rows"] == 1 and (hist / r1["snapshot"]).exists()
    assert LH.snapshot(f, "ann", "та же", "manual", root=hist) is None          # содержимое не менялось
    GT.add("c2", 2, 6, "NEGATIVE", path=f)
    r2 = LH.snapshot(f, "ann", "вторая", "manual", root=hist)
    v = LH.versions(f, hist)
    assert list(v.snapshot) == [r2["snapshot"], r1["snapshot"]]
    d = LH.diff("events", LH.read_version("events", r1["snapshot"], hist), LH.read_version("events", r2["snapshot"], hist))
    assert len(d["added"]) == 1 and len(d["removed"]) == 0
    assert LH.snapshot(f, "ann", "авто", "auto", root=hist) is None             # авто: не чаще AUTO_EVERY_SEC и только при изменении


def test_restore_keeps_current_as_version(tmp_path, monkeypatch):
    f, hist = tmp_path / "events_gt.csv", tmp_path / "hist"
    monkeypatch.setattr(GT, "default_path", lambda: f)
    GT.add("c1", 1, 5, "POSITIVE", path=f)
    r1 = LH.snapshot("events", "a", "v1", root=hist)
    GT.add("c2", 2, 6, "POSITIVE", path=f)
    out = LH.restore("events", r1["snapshot"], "a", root=hist)
    assert len(GT.load(f)) == 1 and out["before"] is not None                  # прежнее содержимое (2 строки) сохранено версией
    assert len(LH.read_version("events", out["before"]["snapshot"], hist)) == 2


# ---------------------------------------------------------------------------------------------- метки жестов
def test_gestures_validate_and_save_with_version(tmp_path, monkeypatch):
    from sd import start_eval as SE

    p = tmp_path / "gesture_gt.csv"
    monkeypatch.setattr(SE, "GT_CSV", p)
    monkeypatch.setattr(LH, "HIST", tmp_path / "hist")
    old = pd.DataFrame([dict(video="v", tid=1, peak_t=2.0, start=1.0, label="unsure", start_quality=None, note="", source="pool", reviewer="a")])
    old.to_csv(p, index=False, encoding="utf-8-sig")
    bad = old.assign(label="курит")
    assert any(l == "error" for l, _ in G.validate(bad))
    with pytest.raises(ValueError):
        G.save(bad)
    assert pd.read_csv(p, encoding="utf-8-sig").label.iloc[0] == "unsure"     # при ошибке файл не тронут
    snap = G.save(old.assign(label="smoke", note="исправил"), author="ann")
    assert pd.read_csv(p, encoding="utf-8-sig").label.iloc[0] == "smoke" and list(pd.read_csv(p, encoding="utf-8-sig").columns) == G.COLS
    assert snap and pd.read_csv(tmp_path / "hist" / snap["snapshot"], encoding="utf-8-sig").label.iloc[0] == "unsure"
    dup = pd.concat([old, old.assign(peak_t=2.1)], ignore_index=True)
    assert any("дубль" in t for _, t in G.validate(dup))


# ---------------------------------------------------------------------------------------------- утечка обучения
def _bundle(tmp_path, videos, by_scene=False, in_manifest=True):
    d = tmp_path / "cycle_x"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(dict(kind="cycle_classifier", by_scene=by_scene, **({"videos": videos} if in_manifest else {}))), encoding="utf-8")
    pd.DataFrame(dict(y=[1] * len(videos), group=videos)).to_csv(d / "oof.csv", index=False)
    return d


def test_training_videos_and_leak(tmp_path):
    d = _bundle(tmp_path, ["a__1", "a__2"])
    assert B.training_videos(d) == ({"a__1", "a__2"}, "manifest")
    assert B.leak(d, ["a__1", "b__9"])["clips"] == ["a__1"]
    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "manifest.json").write_text(json.dumps(dict(kind="cycle_classifier")), encoding="utf-8")
    pd.DataFrame(dict(y=[1], group=["a__3"])).to_csv(d2 / "oof.csv", index=False)
    assert B.training_videos(d2) == ({"a__3"}, "oof")                          # старые пакеты: из групп oof.csv


def test_evaluate_dirs_flags_leak_and_blocks_hidden(tmp_path):
    d1, d2 = make_world(tmp_path)
    gt = _gt(tmp_path, d1, d2)
    b = _bundle(tmp_path, [video_id(d1 / "s1.mp4")])
    prof = PR.Profile("h", options={"allow_heuristic": True, "cycle_bundle": str(b)})
    fake = Fake({"s1": [(10, 20, 0.9)]})
    kw = dict(gt_df=gt, recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c", exp_root=tmp_path / "exp", n_boot=10)
    out = R.evaluate_dirs([d1, d2], prof, policy="fixed", **kw)
    assert out.meta["leak"]["clips"] == [video_id(d1 / "s1.mp4")] and any("оптимистична" in n for n in out.report.notes)
    with pytest.raises(ValueError, match="скрытого"):
        R.evaluate_dirs([d1, d2], prof, role="hidden", policy="fixed", **kw)


# ---------------------------------------------------------------------------------------------- анализ кэша и PDF
@pytest.fixture()
def run(tmp_path):
    d1, d2 = make_world(tmp_path)
    gt = _gt(tmp_path, d1, d2)
    fake = Fake({"s1": [(10, 20, 0.9)], "s2": [(31, 44, 0.3)], "f1": [(5, 12, 0.8)]})
    out = R.evaluate_dirs([d1, d2], PR.heuristic_profile(), policy="fixed", gt_df=gt, recognize_fn=fake, probe_fn=probe, cache_root=tmp_path / "c", exp_root=tmp_path / "exp",
                          n_boot=10)
    return out.run_id, tmp_path / "exp"


def test_cache_table_and_weak_spots(run):
    rid, root = run
    tab = CA.run_table(rid, root)
    assert len(tab) == 3 and {"f1", "tp", "fp", "fn", "rt_factor", "leak"} <= set(tab.columns)
    assert (root / rid / CA.DETAILS).exists()
    rep, _, _, _ = XP.load_run(rid, root)
    gest = pd.DataFrame([dict(video=tab[tab.fp > 0].clip_id.iloc[0], peak_t=8.5, label="drink")])
    spots = CA.weak_spots(tab, rep.errors, gest)
    assert "below_threshold" in set(spots.rule) and "similar_gesture" in set(spots.rule)       # FP рядом с размеченным питьём
    s = CA.summary(spots)
    assert set(s.правило) == set(spots.rule)
    vals, good = CA.matrix(tab)
    assert vals.shape == good.shape and good.abs().max().max() <= 1


def test_pdf_report_builds(run, tmp_path):
    rid, root = run
    pdf = __import__("sd.reports", fromlist=["x"]).build(rid, out=tmp_path / "r.pdf", root=root, author="тест")
    data = pdf.read_bytes()
    assert data[:4] == b"%PDF" and len(data) > 5000 and data.count(b"/Type /Page") >= 3


def test_reproduce_command_has_overrides():
    from sd.reports import reproduce_command

    cmd = reproduce_command(dict(dirs=["data/a"], profile=dict(name="mvp", config={"video.process_fps": 10}, options={"objects": ["d1"]}), policy="fixed", role="validation"))
    assert "-p mvp" in cmd and "video.process_fps=10" in cmd and 'options.objects=["d1"]' in cmd


# ---------------------------------------------------------------------------------------------- ворота выпуска
def test_release_gates(run, tmp_path, monkeypatch):
    from sd import release as RL

    rid, root = run
    pdir = tmp_path / "profiles"
    monkeypatch.setattr(PR, "EXPERIMENTS_DIR", pdir)
    _, meta, _, _ = XP.load_run(rid, root)
    hp = PR.heuristic_profile()
    PR.save(PR.Profile("h1", config=dict(hp.config), options=dict(hp.options)), pdir)
    gs = {g["id"]: g for g in RL.gates("h1", rid, 0.1, exp_root=root)}
    assert gs["run_ok"]["ok"] is True and gs["f1"]["ok"] is True
    assert gs["same_profile"]["ok"] is (PR.load_named("h1", pdir).fingerprint() == meta["fingerprint"])
    assert RL.gates("нет_такого", rid)[0]["ok"] is False
    assert RL.passed([dict(ok=True), dict(ok=None)]) and not RL.passed([dict(ok=False)])


# ---------------------------------------------------------------------------------------------- каталог данных
def test_catalog_issues(tmp_path):
    d_s, d_f = tmp_path / "курение", tmp_path / "лжекурение"
    d_s.mkdir()
    d_f.mkdir()
    (d_s / "a.mp4").write_bytes(b"1" * 100)
    (d_f / "b.mp4").write_bytes(b"2" * 100)
    (d_f / "c.mp4").write_bytes(b"1" * 100)              # копия a.mp4 под другим именем
    gt = GT.empty()
    f = tmp_path / "gt.csv"
    GT.add(video_id(d_f / "b.mp4"), 1, 9, "POSITIVE", path=f)
    gest = pd.DataFrame([dict(video=video_id(d_s / "a.mp4"), peak_t=3.0, label="smoke")])
    tab = DC.videos_table([d_s, d_f], gestures=gest, gt=GT.load(f))
    iss = DC.issues(tab)
    codes = set(iss.code)
    assert {"weak_conflict", "gestures_no_gt", "duplicate", "unlabeled"} <= codes
    assert set(iss[iss.code == "duplicate"].clip_id) == {video_id(d_s / "a.mp4"), video_id(d_f / "c.mp4")}
    cov = DC.coverage(tab)
    assert int(cov.видео.sum()) == 3


def test_runs_with_tables_and_collect_from_run_errors(run):
    from sd import oof_eval as O

    rid, root = run
    assert O.runs_with_tables(root) == []                       # подставной распознаватель не оставил запусков на диске
    with pytest.raises(ValueError, match="нет полных запусков"):
        O.collect_from_run(rid, exp_root=root)


# ---------------------------------------------------------------------------------------------- аналитика подходов и конструктор
@pytest.fixture()
def two_runs(tmp_path):
    d1, d2 = make_world(tmp_path)
    gt = _gt(tmp_path, d1, d2)
    kw = dict(policy="fixed", gt_df=gt, probe_fn=probe, cache_root=tmp_path / "c", exp_root=tmp_path / "exp", n_boot=10)
    a = R.evaluate_dirs([d1, d2], PR.heuristic_profile(), recognize_fn=Fake({"s1": [(10, 20, 0.9)], "f1": [(5, 12, 0.8)]}), **kw)
    hp = PR.heuristic_profile()
    b = R.evaluate_dirs([d1, d2], PR.Profile("v2", config={**hp.config, "events.cycle_th": 0.3}, options=dict(hp.options)),
                        recognize_fn=Fake({"s1": [(10, 20, 0.9)], "s2": [(31, 44, 0.9)]}), **kw)
    return a.run_id, b.run_id, tmp_path / "exp"


def test_notes_and_catalog(two_runs):
    from sd import compare as CP

    a, b, root = two_runs
    XP.set_notes(a, root, tag="база", note="первый", pinned=True)
    runs = XP.list_runs(root)
    assert runs.set_index("id").loc[a, "tag"] == "база" and bool(runs.set_index("id").loc[a, "pinned"])
    XP.set_notes(a, root, tag="", pinned=False)                     # пустые значения удаляют поле
    assert XP.read_notes(a, root) == {"note": "первый"}
    cat = CP.catalog(root)
    assert "cfg.events.cycle_th" in cat and "cfg.events.cycle_th" in CP.varying(cat)
    assert cat.set_index("id").loc[b, "cfg.events.cycle_th"] == 0.3


def test_compare_matrices(two_runs):
    from sd import compare as CP

    a, b, root = two_runs
    f1 = CP.per_clip([a, b], root)
    out = CP.outcome_matrix([a, b], root)
    assert set(f1.columns) == {a, b} and (out.nunique(axis=1) > 1).any()     # клип s2 найден только во втором подходе
    assert CP.disagreement(f1).max() > 0
    assert set(CP.error_causes([a, b], root).id) <= {a, b}
    assert set(CP.curves([a, b], root).id) == {a, b}
    cat = CP.catalog(root)
    eff = CP.effect(cat, "cfg.events.cycle_th")
    assert set(eff["значение"]) == {"0.3", str(PR.heuristic_profile().cfg()["events"]["cycle_th"])}
    pairs = CP.controlled_pairs(cat, "cfg.events.cycle_th")
    assert len(pairs) == 1                                           # отличаются только этим параметром
    m = CP.pairwise([a, b], n=50, root=root)
    assert m.loc[a, a] == "—" and m.loc[a, b] != ""


def test_arch_schema_impact_and_graph():
    from sd import arch as A

    df = A.parse_defaults()
    assert len(df) > 80 and (df.comment == "").sum() == 0             # у каждого параметра default.yaml есть описание
    assert A.stage_of("pose.imgsz") == "pose" and A.stage_of("options.objects") == "evidence" and A.stage_of("events.cycle_th") == "events"
    assert A.impact(["events.cycle_th"])["level"] == 0 and A.impact(["video.process_fps"])["level"] == 4
    assert "series" in A.impact(["cycles.th_in"])["recompute"] or A.impact(["cycles.th_in"])["recompute"][0] == "cycles"
    base = PR.heuristic_profile()
    new = PR.with_overrides(base, ["video.process_fps=7", "options.objects=[smoking_yolo11m_beehzod]"])
    d = A.diff(base, new)
    assert set(d.key) == {"video.process_fps", "options.objects"}
    dot = A.dot(new, {"pose"}, {"series"})
    assert dot.startswith("digraph") and "fde7c7" in dot and "smoking_yolo11m_beehzod" in dot
