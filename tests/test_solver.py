import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from rt_helpers import make_bundle
from sd import fusion as FU
from sd import gt as GT
from sd import profiles as PR
from sd import solver as SV

FAST = ["hold", "d_min", "mouth_dur", "p_wm_dist"]
HEAVY = FAST + ["obj_det_a_max_conf", "obj_det_b_max_conf", "photo_p_mean"]


@pytest.fixture()
def env(tmp_path):
    """Две «машины»: исходная (папка src) и целевая (dst) со своими каталогами моделей и профилей."""
    src = tmp_path / "src"
    make_bundle(src / "cycle" / "fast1", FAST)
    make_bundle(src / "cycle" / "heavy1", HEAVY, seed=1)
    make_bundle(src / "photo" / "ph1", ["photo_p_mean"], seed=2)
    return tmp_path, src


def prof(src, bundle="fast1", **opt):
    return PR.Profile("sol", "тест", config={"events.confidence_threshold": 0.6, "pose.imgsz": 640}, options={"cycle_bundle": str(src / "cycle" / bundle), **opt})


# ------------------------------------------------------------------------------------------ классификатор обязателен
def test_classifier_is_required_unless_explicitly_allowed():
    p = PR.default_profile()
    assert "обязателен" in p.classifier_problem() and [i.code for i in SV.check(p)] == ["classifier_required"]
    h = PR.heuristic_profile()
    assert h.classifier_problem() is None and [i.code for i in SV.check(h)] == ["heuristic"] and not SV.errors(h)


def test_classifier_must_exist_and_be_readable(tmp_path):
    p = PR.Profile("x", options={"cycle_bundle": str(tmp_path / "cycle" / "нет")})
    assert [i.code for i in SV.errors(p)] == ["classifier_missing"]
    bad = tmp_path / "cycle" / "bad"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{не json")
    assert [i.code for i in SV.errors(PR.Profile("x", options={"cycle_bundle": str(bad)}))] == ["classifier_broken"]


def test_extractors_must_cover_classifier_features(env):
    _, src = env
    codes = lambda p, m="offline": sorted(i.code for i in SV.errors(p, m))               # noqa: E731
    assert codes(prof(src, "fast1")) == []
    heavy = prof(src, "heavy1")
    assert codes(heavy) == ["objects_missing", "photo_missing"]                           # признаки стали бы тихими пропусками
    ok = prof(src, "heavy1", objects=["det_a", "det_b"], photo_bundle=str(src / "photo" / "ph1"))
    assert codes(ok) == [] and codes(ok, "replay") == [] and codes(ok, "live") == ["live_features"]
    assert codes(prof(src, "heavy1", objects=["det_a"], photo_bundle="x")) == ["objects_missing"]


def test_vlm_in_classifier_vector_requires_vlm_all(env):
    _, src = env
    make_bundle(src / "cycle" / "withvlm", FAST + ["vlm_yesno"], seed=3)
    assert [i.code for i in SV.errors(prof(src, "withvlm"))] == ["vlm_required"]
    assert SV.errors(prof(src, "withvlm", vlm_model="m", vlm_mode="all")) == []
    assert [i.code for i in SV.errors(prof(src, "withvlm", vlm_model="m", vlm_mode="all"), "replay")] == ["replay_vlm"]


def test_fusion_warnings_and_validation(env):
    _, src = env
    w = lambda p: sorted(i.code for i in SV.check(p))                                       # noqa: E731
    assert w(prof(src, fusion={"weights": {"vlm": 1.0}})) == ["fusion_vlm_inactive"]
    assert w(prof(src, fusion={"weights": {"object": 1.0, "photo": 1.0}})) == ["fusion_object_inactive", "fusion_photo_inactive"]
    assert "vlm_unused" in w(prof(src, vlm_model="m", vlm_mode="grey"))                      # VLM включён, а использовать нечем
    assert "vlm_unused" not in w(prof(src, vlm_model="m", vlm_mode="grey", fusion={"weights": {"vlm": 1.0}}))
    assert "fusion_and_full" in w(prof(src, vlm_model="m", vlm_mode="grey", cycle_bundle_full=str(src / "cycle" / "heavy1"), fusion={"weights": {"vlm": 1.0}}))
    assert [i.code for i in SV.errors(prof(src, fusion={"weights": {"vlm": -1}}))] == ["fusion_invalid"]
    assert [i.code for i in SV.errors(prof(src, fusion={"weights": {"смех": 1}}))] == ["fusion_invalid"]


# ------------------------------------------------------------------------------------------ слияние не трогает вектор признаков
def test_fusion_is_identity_without_weights_or_signals_and_never_touches_classifier_inputs(env):
    _, src = env
    from sd.bundle import Bundle

    b = Bundle(src / "cycle" / "fast1")
    rng = np.random.default_rng(0)
    tab = pd.DataFrame({f: rng.normal(size=20) for f in FAST})
    base = b.score(tab)
    # лишние колонки (VLM, предмет, фото) не меняют оценку классификатора: он берёт ровно свои признаки из manifest
    tab2 = tab.assign(vlm_yesno=0.9, obj_any_max_conf=0.7, photo_p_mean=0.8, mystery=1.0)
    assert np.array_equal(b.score(tab2), base) and b.features == sorted(FAST)
    assert np.array_equal(FU.fuse(base, tab2, None)[0], base) and np.array_equal(FU.fuse(base, tab2, {"weights": {"vlm": 0.0}})[0], base)
    nan_tab = tab.assign(vlm_yesno=np.nan)
    assert np.array_equal(FU.fuse(base, nan_tab, {"weights": {"vlm": 1.0}})[0], base)         # нет сигнала — оценка прежняя (даже на краях)


def test_fusion_moves_score_in_the_right_direction_and_is_bounded():
    base = np.array([0.5, 0.5, 0.5, 0.99])
    tab = pd.DataFrame(dict(vlm_yesno=[0.95, 0.05, np.nan, 0.0], obj_any_max_conf=[0.9, 0.0, 0.9, 0.0]))
    out, contrib = FU.fuse(base, tab, {"weights": {"vlm": 1.0, "object": 0.5}, "clip_logit": 1.5})
    assert out[0] > 0.5 > out[1] and out[2] > 0.5 and abs(out[2] - 0.5) <= abs(out[0] - 0.5)
    assert (np.abs(np.log(out / (1 - out)) - np.log(base / (1 - base)))[:3] <= 1.5 + 1e-9).all() and out[3] < 0.99      # поправка ограничена
    assert list(contrib.columns) == ["fusion_vlm", "fusion_photo", "fusion_object"] and (contrib["fusion_photo"] == 0).all()


# ------------------------------------------------------------------------------------------ архитектура и упаковка
def test_architecture_is_detected_from_profile_and_bundle(env):
    _, src = env
    a = SV.detect_architecture(prof(src, "heavy1", objects=["det_a", "det_b"], photo_bundle=str(src / "photo" / "ph1"), vlm_model="qwen3.5:2b-q4_K_M", vlm_mode="grey",
                                    fusion={"weights": {"vlm": 0.8}}))
    assert a["mode"] == "classifier+fusion+evidence+photo+vlm" and [s["id"] for s in a["stages"]] == ["pose_track", "cycle_fsm", "cycle_classifier", "evidence", "photo", "vlm", "fusion", "events"]
    cl = next(s for s in a["stages"] if s["id"] == "cycle_classifier")["summary"]
    assert cl["n_features"] == 7 and cl["groups"] == {"kinematics": 3, "pose": 1, "object": 2, "photo": 1} and cl["requires"]["objects"] == ["det_a", "det_b"] and cl["members"] == {"lr": "logreg"}
    assert a["streaming"] == dict(offline=True, replay=False, live=False) and a["resources_estimate"]["peak_gb"] > 2
    fast = SV.detect_architecture(prof(src))
    assert fast["mode"] == "classifier" and fast["streaming"]["live"] is True and SV.feature_groups(["hold", "obj_x_hit", "vlm_yesno", "xclip_a", "p_q", "photo_p_mean", "zs_x"]).keys() >= {"object", "vlm", "tube", "pose", "photo"}


def test_pack_inspect_install_roundtrip_between_machines(env, monkeypatch):
    tmp, src = env
    p = prof(src, "heavy1", objects=["det_a", "det_b"], photo_bundle=str(src / "photo" / "ph1"), fusion={"weights": {"object": 0.5}})
    res = SV.pack(p, tmp / "out" / "final", name="final")
    assert res["path"].endswith(".sdsolver.zip") and res["size_mb"] < 1
    info = SV.inspect(res["path"])
    assert info["ok"] and info["meta"]["id"] == "final" and info["meta"]["architecture"]["mode"] == "classifier+fusion+evidence+photo"
    names = zipfile.ZipFile(res["path"]).namelist()
    assert "solver.json" in names and "profile.yaml" in names and "bundles/cycle/heavy1/manifest.json" in names and "bundles/photo/ph1/manifest.json" in names
    meta = info["meta"]
    assert meta["profile"]["options"]["cycle_bundle"] == "models/cycle/heavy1" and meta["profile"]["options"]["photo_bundle"] == "models/photo/ph1"      # ссылки переносимые
    # «другая машина»
    mdir, pdir = tmp / "dst" / "models", tmp / "dst" / "profiles"
    from sd import paths as P

    monkeypatch.setattr(P, "MODELS", mdir)                                      # на «другой машине» models/ — здесь
    out = SV.install(res["path"], models_dir=mdir, profiles_dir=pdir)
    assert out["name"] == "final" and out["architecture"] == meta["architecture"]["mode"] and {"cycle/heavy1", "photo/ph1"} <= set(out["installed"])
    assert (mdir / "cycle" / "heavy1" / "manifest.json").exists() and (mdir / "solvers" / "final" / "solver.json").exists()
    q = PR.load("final", pdir)
    assert q.opts()["objects"] == ["det_a", "det_b"] and q.cfg()["pose"]["imgsz"] == 640 and q.threshold == 0.6 and q.opts()["fusion"] == {"weights": {"object": 0.5}}
    assert q.fingerprint() == p.fingerprint()                                   # одна и та же система: отпечаток не зависит от места установки
    assert SV.installed_solvers(mdir)[0]["name"] == "final"
    with pytest.raises(FileExistsError):
        SV.install(res["path"], models_dir=mdir, profiles_dir=pdir)
    assert SV.install(res["path"], models_dir=mdir, profiles_dir=pdir, overwrite=True)["name"] == "final"


def test_pack_refuses_profile_without_classifier_or_with_gaps(env):
    tmp, src = env
    with pytest.raises(ValueError, match="обязателен"):
        SV.pack(PR.default_profile(), tmp / "x")
    with pytest.raises(ValueError, match="objects_missing|детекторов|признаки предмета"):
        SV.pack(prof(src, "heavy1"), tmp / "x")


def test_install_rejects_tampered_archive_and_unsafe_members(env):
    tmp, src = env
    good = SV.pack(prof(src), tmp / "g", name="g")["path"]

    def rewrite(mod, name):
        out = tmp / f"{name}.sdsolver.zip"
        with zipfile.ZipFile(good) as zi, zipfile.ZipFile(out, "w") as zo:
            for n in zi.namelist():
                zo.writestr(n, mod(n, zi.read(n)) if mod else zi.read(n))
        return out

    tampered = rewrite(lambda n, b: b + b"x" if n.endswith("members.json") else b, "t")
    assert not SV.inspect(tampered)["ok"]
    with pytest.raises(ValueError, match="целостности"):
        SV.install(tampered, models_dir=tmp / "m", profiles_dir=tmp / "p")
    # путь за пределами каталога: sha256 в solver.json совпадает, но распаковывать нельзя
    evil = tmp / "evil.sdsolver.zip"
    with zipfile.ZipFile(good) as zi:
        meta = json.loads(zi.read("solver.json"))
    meta["files"]["bundles/../../escape.json"] = "0" * 64
    with zipfile.ZipFile(evil, "w") as zo:
        zo.writestr("solver.json", json.dumps(meta))
        zo.writestr("bundles/../../escape.json", "{}")
    assert not SV.inspect(evil)["ok"]
    with pytest.raises(ValueError):
        SV.install(evil, models_dir=tmp / "m", profiles_dir=tmp / "p")
    assert not (tmp / "m").exists() or not list((tmp / "m").glob("**/escape.json"))
    with pytest.raises(ValueError, match="недопустимый"):
        SV._safe_member("weights/evil.exe")
    with pytest.raises(ValueError, match="недопустимый"):
        SV._safe_member("/etc/passwd")


def test_pack_with_dataset_report_and_weights(env, tmp_path, monkeypatch):
    tmp, src = env
    from sd import experiments as XP
    from sd import models as M

    labels = tmp / "labels"
    GT.add("clip1", 1, 9, "POSITIVE", person=1, box=(0, 0, 10, 10), path=labels / "events_gt.csv")
    exp = tmp / "exp"
    (exp / "r1").mkdir(parents=True)
    (exp / "r1" / "report.json").write_text(json.dumps(dict(mode="events", metrics=dict(f1=0.8, precision=0.9, recall=0.7, tp=7, fp=1, fn=3, threshold=0.6, clips=4), ci=dict(f1=[0.6, 0.9]),
                                                            budget=dict(target=0.8, reached=True))))
    (exp / "r1" / "meta.json").write_text(json.dumps(dict(gt_fingerprint="abc", role="validation", created="2026-10-08")))
    w = tmp / "w"
    w.mkdir()
    (w / "pose.onnx").write_bytes(b"onnx-bytes")
    monkeypatch.setattr(SV, "MODELS", w)
    monkeypatch.setattr(SV, "_spec", lambda i: dict(id=i, registered=True, kind="pose", file="pose.onnx", license="AGPL-3.0", trust="official", sha256=None, hf_repo=None, rtmlib=None, present=True))
    res = SV.pack(prof(src), tmp / "full", name="full", with_weights=True, with_dataset=True, run_id="r1", exp_root=exp, labels_dir=labels)
    meta = SV.inspect(res["path"])["meta"]
    assert meta["evaluation"]["metrics"]["f1"] == 0.8 and meta["evaluation"]["gt_fingerprint"] == "abc" and meta["dataset"]["clips"] == 1
    assert meta["weights"][0]["included"] and meta["weights"][0]["archive_path"] == "weights/pose.onnx"
    mdir = tmp / "dst2" / "models"
    out = SV.install(res["path"], models_dir=mdir, profiles_dir=tmp / "dst2" / "p", labels_dir=tmp / "dst2" / "labels", install_dataset=True)
    assert (mdir / "pose.onnx").read_bytes() == b"onnx-bytes" and (tmp / "dst2" / "labels" / "events_gt.csv").exists() and "labels/events_gt.csv" in out["installed"]


def test_resolve_profile_accepts_name_yaml_and_archive(env, tmp_path, monkeypatch):
    tmp, src = env
    pdir, mdir = tmp / "pp", tmp / "mm"
    monkeypatch.setattr(PR, "EXPERIMENTS_DIR", pdir)
    monkeypatch.setattr(SV, "MODELS", mdir)
    zp = SV.pack(prof(src), tmp / "z", name="zz")["path"]
    p = SV.resolve_profile(zp)                    # архив → установка → профиль
    assert p.name == "zz" and (mdir / "solvers" / "zz" / "solver.json").exists() and SV.resolve_profile("zz").name == "zz"
