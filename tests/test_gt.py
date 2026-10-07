import numpy as np
import pandas as pd
import pytest

from sd import gt as G


def test_add_load_roundtrip_and_validation(tmp_path):
    f = tmp_path / "gt.csv"
    rid = G.add("курение__sm_3", 10, 25, "positive", person=2, peak=17, box=(10, 20, 110, 220), note="тест", path=f)
    G.add("курение__sm_3", 40, 50, "IGNORE", path=f)
    df = G.load(f)
    assert len(df) == 2 and df.id.iloc[0] == rid and df.label.tolist() == ["POSITIVE", "IGNORE"]
    assert df.person_gt_id.iloc[0] == "2" and df.peak_sec.iloc[0] == 17 and df.x2.iloc[0] == 110
    # IGNORE без рамки допустим, POSITIVE без рамки — предупреждение
    G.add("курение__sm_3", 60, 70, "POSITIVE", path=f)
    warns = G.validate(G.load(f))
    assert any("без рамки" in w for w in warns)


@pytest.mark.parametrize("args", [(10, 5, "POSITIVE"), (-1, 5, "POSITIVE"), (1, 5, "MAYBE"), (np.nan, 5, "POSITIVE")])
def test_add_rejects_bad_input(tmp_path, args):
    with pytest.raises(ValueError):
        G.add("c", *args, path=tmp_path / "g.csv")


def test_add_rejects_inverted_box(tmp_path):
    with pytest.raises(ValueError):
        G.add("c", 1, 5, "POSITIVE", box=(100, 0, 10, 50), path=tmp_path / "g.csv")


def test_peak_is_clamped_into_interval(tmp_path):
    f = tmp_path / "g.csv"
    G.add("c", 10, 20, "POSITIVE", peak=99, box=(0, 0, 1, 1), path=f)
    assert G.load(f).peak_sec.iloc[0] == 20


def test_delete_update_and_clean(tmp_path):
    f = tmp_path / "g.csv"
    a = G.add("c1", 1, 5, "POSITIVE", box=(0, 0, 10, 10), path=f)
    b = G.mark_clean("c2", 120.0, path=f)
    assert G.mark_clean("c2", 120.0, path=f) == b          # повтор не дублирует
    assert G.update(a, path=f, label="negative", note="передумал") and G.load(f).set_index("id").loc[a, "label"] == "NEGATIVE"
    assert G.delete([a], path=f) == 1 and len(G.load(f)) == 1
    assert G.delete(["нет-такого"], path=f) == 0


def test_clip_status_and_fingerprint_changes_with_labels(tmp_path):
    f = tmp_path / "g.csv"
    G.add("c1", 1, 5, "POSITIVE", box=(0, 0, 10, 10), path=f)
    G.mark_clean("c2", 60, path=f)
    df = G.load(f)
    st = G.clip_status(df, ["c1", "c2", "c3"]).set_index("clip_id")
    assert st.loc["c1", "positive"] == 1 and st.loc["c2", "negative"] == 1 and not st.loc["c3", "reviewed"] and st.loc["c1", "reviewed"]
    fp1 = G.fingerprint(df)
    G.add("c1", 10, 15, "POSITIVE", box=(0, 0, 10, 10), path=f)
    assert G.fingerprint(G.load(f)) != fp1


def test_to_eval_gts_filters_clips_and_uses_boxes(tmp_path):
    f = tmp_path / "g.csv"
    G.add("c1", 1, 5, "POSITIVE", person=1, box=(0, 0, 10, 10), path=f)
    G.add("c1", 8, 9, "IGNORE", path=f)
    G.add("c2", 1, 5, "POSITIVE", box=(0, 0, 10, 10), path=f)
    gts = G.to_eval_gts(G.load(f), {"c1"})
    assert [g.label for g in gts] == ["POSITIVE", "IGNORE"] and gts[0].box == (0, 0, 10, 10) and gts[1].box is None
    traj = G.to_eval_gts(G.load(f), {"c1"}, box_at=lambda c, p, t: (1, 1, 2, 2))
    assert traj[0].box_at(3.0) == (1, 1, 2, 2) and traj[1].box_at is None     # у IGNORE нет человека


def test_import_export_organizer_format(tmp_path):
    f = tmp_path / "g.csv"
    new = pd.DataFrame(dict(clip_id=["a", "a"], start_sec=[1.0, 20.0], end_sec=[9.0, 30.0], label=["positive", "x"], x1=[0, np.nan], y1=[0, np.nan], x2=[5, np.nan], y2=[9, np.nan]))
    assert G.import_rows(new, path=f) == 1                  # неизвестная метка отброшена
    out = G.export_organizer(G.load(f))
    assert list(out.columns) == G.ORGANIZER_COLUMNS and out.camera_id.iloc[0] == "cam_local" and out.peak_sec.iloc[0] == 5.0
    with pytest.raises(ValueError):
        G.import_rows(pd.DataFrame(dict(clip_id=["a"])), path=f)


def test_overlapping_positive_same_person_is_reported(tmp_path):
    f = tmp_path / "g.csv"
    G.add("c1", 1, 10, "POSITIVE", person=1, box=(0, 0, 5, 5), path=f)
    G.add("c1", 8, 15, "POSITIVE", person=1, box=(0, 0, 5, 5), path=f)
    assert any("пересекающиеся" in w for w in G.validate(G.load(f)))


def test_load_missing_or_empty_file_gives_empty_frame(tmp_path):
    assert G.load(tmp_path / "нет.csv").empty
    (tmp_path / "e.csv").write_text("")
    assert list(G.load(tmp_path / "e.csv").columns) == G.COLUMNS
