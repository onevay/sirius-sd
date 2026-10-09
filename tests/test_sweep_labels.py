import pandas as pd
import pytest

from sd import evaluation as EV
from sd import gt as GT
from sd import labels_from_gt as LG
from sd import profiles as PR
from sd import sweep as SW
from sd.evaluate import GT as G
from sd.evaluate import Pred

B = (0.0, 0.0, 100.0, 200.0)


def test_variants_combine_params_and_validate_keys():
    base = PR.Profile("base", config={"pose.imgsz": 960})
    vs = SW.variants_from_rows(base, [dict(name="small", param="pose.imgsz", value="640"), dict(name="vlm", param="vlm_model", value="m1"), dict(name="vlm", param="vlm_mode", value="grey"),
                                      dict(name="", param="x", value="1"), dict(name="skip", param="", value="1")])
    assert [v.name for v in vs] == ["base", "base+small", "base+vlm"]
    assert vs[0].config["pose.imgsz"] == 960 and vs[1].config["pose.imgsz"] == 640 and vs[2].options == dict(vlm_model="m1", vlm_mode="grey")
    assert [v.name for v in SW.variants_from_rows(base, [dict(name="a", param="events.cycle_th", value="0.6")], include_base=False)] == ["base+a"] and vs[0].config is not base.config
    with pytest.raises(KeyError):
        SW.variants_from_rows(base, [dict(name="bad", param="pose.imgszz", value="1")])
    assert SW.variants_from_rows(base, [dict(name="l", param="objects", value="[a, b]")])[1].options["objects"] == ["a", "b"]


def _scene():
    durations = {f"c{i}": 100.0 for i in range(8)}
    gts = [G(c, "1", 10, 20, "POSITIVE", box=B) for c in durations]
    good = [Pred(c, 11, 19, 0.9, 15, B) for c in durations]                       # профиль A находит все
    weak = [Pred(c, 11, 19, 0.9, 15, B) for c in list(durations)[:4]]             # профиль B — половину
    return durations, gts, good, weak


def test_paired_bootstrap_detects_clear_difference_and_is_deterministic():
    d, g, a, b = _scene()
    st = EV.Settings(threshold=0.5)
    r = EV.paired_bootstrap(b, a, g, d, st, n=80, seed=3)                         # B=weak, «B лучше A»? A=weak, B=good
    assert r["diff"] > 0.2 and r["p_better"] > 0.95 and r["lo"] > 0 and r == EV.paired_bootstrap(b, a, g, d, st, n=80, seed=3)
    same = EV.paired_bootstrap(a, a, g, d, st, n=80)
    assert same["diff"] == 0.0 and same["p_better"] == 0.0
    assert EV.paired_bootstrap(a, b, g, {"c0": 1.0}, st)["diff"] is None


def test_compare_saved_runs(tmp_path):
    d, gts_, good, weak = _scene()
    st = EV.Settings(threshold=0.5)
    gt_rows = pd.DataFrame([dict(id=str(i), clip_id=c, camera_id="", person_gt_id="1", start_sec=10.0, end_sec=20.0, label="POSITIVE", peak_sec=15.0, x1=0.0, y1=0.0, x2=100.0, y2=200.0,
                                 note="", labeler="", ts="") for i, c in enumerate(d)])
    from sd import experiments as XP

    def save(preds, name):
        rep = EV.build_report(preds, GT.to_eval_gts(gt_rows), d, st, n_boot=20)
        ev = pd.DataFrame([dict(camera_id="c", clip_id=p.clip_id, event_id=p.event_id or "e", start_sec=p.start, end_sec=p.end, confidence=p.confidence, label="x", person_track_id=1,
                                peak_sec=p.peak, x1=p.box[0], y1=p.box[1], x2=p.box[2], y2=p.box[3]) for p in preds])
        meta = dict(name=name, clips=[dict(clip_id=c, duration=v) for c, v in d.items()], gt_fingerprint="g", describe={})
        return XP.save_run(rep, meta, ev, gt_rows, run_id=name, root=tmp_path).name

    a, b = save(weak, "a"), save(good, "b")
    r = SW.compare_saved(a, b, n=60, root=tmp_path)
    assert r["diff"] > 0.2 and r["clips"] == 8 and r["same_gt"] is True


def test_cycle_labels_from_gt_rules():
    gt = pd.DataFrame([dict(clip_id="v", label="POSITIVE", start_sec=10, end_sec=30, x1=0.0, y1=0.0, x2=100.0, y2=200.0),
                       dict(clip_id="v", label="IGNORE", start_sec=50, end_sec=60, x1=float("nan"), y1=float("nan"), x2=float("nan"), y2=float("nan")),
                       dict(clip_id="v", label="NEGATIVE", start_sec=70, end_sec=80, x1=float("nan"), y1=float("nan"), x2=float("nan"), y2=float("nan"))])
    cyc = pd.DataFrame([dict(video="v", tid=1, peak_t=12.0, cx=1.0, cy=1.0), dict(video="v", tid=2, peak_t=14.0, cx=1.0, cy=1.0), dict(video="v", tid=1, peak_t=33.0, cx=1.0, cy=1.0),
                        dict(video="v", tid=1, peak_t=55.0, cx=1.0, cy=1.0), dict(video="v", tid=1, peak_t=75.0, cx=1.0, cy=1.0), dict(video="v", tid=1, peak_t=95.0, cx=1.0, cy=1.0),
                        dict(video="other", tid=1, peak_t=5.0, cx=1.0, cy=1.0)])
    box = lambda v, tid, t: B if tid == 1 else (500.0, 0.0, 600.0, 200.0)       # noqa: E731
    out = LG.cycle_labels_from_gt(cyc, gt, box)
    assert out.label.tolist() == ["smoke", "other_neg", "smoke", "ignore", "other_neg", "other_neg"] and len(out) == 6        # клип «other» не размечен — меток нет
    assert LG.cycle_labels_from_gt(cyc, gt, box, unlabeled_as_negative=False).label.tolist() == ["smoke", "other_neg", "smoke", "ignore", "other_neg"]
    assert LG.summarize(out) == {"other_neg": 3, "smoke": 2, "ignore": 1} and LG.summarize(out.iloc[0:0]) == {}


def test_bulk_save_replaces_gt_rows_keeps_manual(tmp_path, monkeypatch):
    from sd import dataset as DS

    monkeypatch.setattr(DS, "LABELS", tmp_path)
    monkeypatch.setattr(DS, "LABELS_CSV", tmp_path / "cycle_labels.csv")
    DS.save_label("v", 1, 5.0, 10.0, 10.0, "smoke", labeler="me")
    rows = pd.DataFrame([dict(video="v", tid=1, peak_t=20.0, cx=3.0, cy=3.0, label="other_neg")])
    assert DS.save_labels_bulk(rows) == 1
    assert DS.save_labels_bulk(rows.assign(peak_t=21.0)) == 1                      # повтор заменяет прежние строки source=gt
    df = DS.load_labels()
    assert len(df) == 2 and set(df.source) == {"manual", "gt"} and df[df.source == "gt"].peak_t.tolist() == [21.0]
