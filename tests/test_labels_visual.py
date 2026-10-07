"""Визуальные метки (video, tid, start) -> формат проекта (peak_t, cx, cy) и признаки группы G (VLM) в таблице циклов."""
import numpy as np
import pandas as pd

from sd import dataset as D


def _table():
    return pd.DataFrame(dict(video=["a", "a", "b"], group=["a", "a", "b"], weak_label=["smoking", "smoking", "fake"], tid=[1, 1, 2],
                             start=[37.6049, 48.0961, 5.0], peak_t=[38.6, 49.2, 5.5], cx=[500.0, 260.0, 300.0], cy=[330.0, 350.0, 250.0]))


def _csv(tmp_path):
    p = tmp_path / "visual.csv"
    pd.DataFrame(dict(video=["a", "a", "b", "b"], k=[0, 1, 0, 1], tid=[1, 1, 2, 2], start=[37.6, 48.1, 5.0, 99.0], label=["smoke", "unsure", "drink", "smoke"],
                      reviewer=["me"] * 4, note=["", "", "", ""])).to_csv(p, index=False)
    return p


def test_visual_labels_are_matched_by_tid_and_rounded_start(tmp_path):
    lab = D.visual_labels_to_project(_csv(tmp_path), _table())
    assert len(lab) == 3                                                       # четвёртая метка (start 99.0) цикла не нашла
    assert set(lab.label) == {"smoke", "unsure", "drink"} and (lab.source == "visual").all()
    r = lab[(lab.video == "a") & (lab.label == "smoke")].iloc[0]
    assert r.peak_t == 38.6 and r.cx == 500.0 and r.cy == 330.0                # положение и пик взяты из таблицы циклов


def test_attach_labels_uses_given_labels_as_manual_and_skips_unsure(tmp_path):
    t = _table()
    lab = D.visual_labels_to_project(_csv(tmp_path), t)
    out = D.attach_labels(t, use_weak=False, labels=lab)
    assert out.label.tolist() == ["smoke", "unsure", "drink"]
    assert out.y.tolist()[0] == 1 and np.isnan(out.y.tolist()[1]) and out.y.tolist()[2] == 0   # unsure в обучение не идёт
    assert (out.label_source == "manual").all()
    weak = D.attach_labels(t.iloc[[2]].assign(video="zzz"), use_weak=True, labels=lab)         # нет ручной метки -> слабая по папке
    assert weak.label_source.iloc[0] == "weak" and weak.label.iloc[0] == "other_neg"


def test_vlm_scores_become_group_g_columns():
    t = _table()
    vlm = pd.DataFrame(dict(video=["a", "b"], tid=[1, 2], start=[37.6049, 5.0], score_yesno=[0.9, 0.1], score_letter=[0.7, 0.2], ms_yesno=[1.0, 1.0]))
    out = D.attach_model_features(t, vlm=vlm)
    assert out.vlm_yesno.tolist()[0] == 0.9 and out.vlm_letter.tolist()[2] == 0.2 and np.isnan(out.vlm_yesno.tolist()[1])
    assert "ms_yesno" not in out and set(D.VLM) <= set(out.columns)
    assert set(D.VLM) == set(D.FEATURE_GROUPS["G_vlm"])
