import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from sd import model_tools as MT


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Подменяем каталоги моделей на временные: тесты не трогают настоящие models/."""
    monkeypatch.setattr(MT, "MODELS", tmp_path)
    monkeypatch.setattr(MT, "USER_REGISTRY", tmp_path / "user_models.yaml")
    monkeypatch.setattr(MT, "BUNDLE_KINDS", {"photo": tmp_path / "photo", "cycle": tmp_path / "cycle"})
    import sd.models as M

    monkeypatch.setattr(M, "MODELS", tmp_path)
    return tmp_path


def _bundle(root, kind, name, extra=None):
    d = root / kind / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(kind=f"{kind}_x", created="t", n=5, features=["a"])), encoding="utf-8")
    (d / "members.json").write_text("{}", encoding="utf-8")
    for k, v in (extra or {}).items():
        (d / k).write_bytes(v)
    return d


def test_scan_formats(tmp_path):
    (tmp_path / "a.onnx").write_bytes(b"x")
    (tmp_path / "b.safetensors").write_bytes(b"x")
    (tmp_path / "c.exe").write_bytes(b"MZ")
    assert MT.scan_model_path(tmp_path / "a.onnx")["verdict"] == "ok"
    assert MT.scan_model_path(tmp_path / "b.safetensors")["verdict"] == "ok"
    assert MT.scan_model_path(tmp_path / "c.exe")["verdict"] == "unknown"
    import pickle

    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo hi",))

    (tmp_path / "evil.pt").write_bytes(pickle.dumps(Evil()))
    assert MT.scan_model_path(tmp_path / "evil.pt")["verdict"] == "danger"


def test_register_refuses_dangerous_weights_and_validates_schema(sandbox, tmp_path):
    import pickle

    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo hi",))

    bad = tmp_path / "evil.pt"
    bad.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(RuntimeError, match="проверку безопасности"):
        MT.register_user_model(dict(id="evil", kind="detector"), copy_from=bad)
    assert not (sandbox / "user").exists() or not list((sandbox / "user").glob("evil*"))
    with pytest.raises(ValueError):
        MT.register_user_model(dict(id="bad id!", kind="detector"))
    with pytest.raises(ValueError):
        MT.register_user_model(dict(id="ok", kind="nonsense"))


def test_register_onnx_roundtrip_and_registry_merge(sandbox, tmp_path):
    f = tmp_path / "det.onnx"
    f.write_bytes(b"onnx-bytes")
    r = MT.register_user_model(dict(id="my-det", kind="detector", license="MIT"), copy_from=f)
    assert r["spec"]["trust"] == "community" and r["spec"]["file"] == "user/my-det.onnx" and (sandbox / "user" / "my-det.onnx").read_bytes() == b"onnx-bytes"
    assert [m["id"] for m in MT.load_user_models()] == ["my-det"]
    import sd.models as M

    reg = M.load_registry()
    assert "my-det" in reg and reg["my-det"].kind == "detector" and "yolo26n-pose" in reg          # встроенные на месте
    assert M.load_registry(include_user=False).get("my-det") is None
    assert MT.remove_user_model("my-det") and not (sandbox / "user" / "my-det.onnx").exists() and MT.load_user_models() == []


def test_export_import_bundles_roundtrip_and_tamper_detection(sandbox, tmp_path):
    _bundle(sandbox, "photo", "p1", {"w.safetensors": b"1234"})
    _bundle(sandbox, "cycle", "c1", {"lgbm_a.txt": b"tree"})
    assert {(b["kind"], b["name"]) for b in MT.list_bundles()} == {("photo", "p1"), ("cycle", "c1")}
    z = tmp_path / "out.zip"
    man = MT.export_bundles([("photo", "p1"), ("cycle", "c1")], z)
    assert "photo/p1/manifest.json" in man and "cycle/c1/lgbm_a.txt" in man
    import shutil

    shutil.rmtree(sandbox / "photo")
    shutil.rmtree(sandbox / "cycle")
    assert MT.import_bundles(z) == ["cycle/c1", "photo/p1"]
    assert (sandbox / "photo" / "p1" / "w.safetensors").read_bytes() == b"1234"
    with pytest.raises(FileExistsError):
        MT.import_bundles(z)                                         # без overwrite существующий пакет не затирается
    assert MT.import_bundles(z, overwrite=True) == ["cycle/c1", "photo/p1"]
    # подмена содержимого: sha256 не совпадёт
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(z) as zin, zipfile.ZipFile(bad, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, b"HACKED" if n.endswith("lgbm_a.txt") else zin.read(n))
    shutil.rmtree(sandbox / "cycle")
    shutil.rmtree(sandbox / "photo")
    with pytest.raises(ValueError, match="sha256"):
        MT.import_bundles(bad)
    assert not (sandbox / "photo").exists() and not (sandbox / "cycle").exists()   # атомарность: ничего не записано


def test_import_rejects_traversal_unsafe_extension_and_missing_manifest(sandbox, tmp_path):
    def mk(name, members, manifest=True):
        p = tmp_path / name
        with zipfile.ZipFile(p, "w") as z:
            for n, data in members.items():
                z.writestr(n, data)
            if manifest:
                z.writestr("MANIFEST.sha256.json", json.dumps({n: __import__("hashlib").sha256(d).hexdigest() for n, d in members.items()}))
        return p

    with pytest.raises(ValueError, match="MANIFEST"):
        MT.import_bundles(mk("a.zip", {"photo/x/manifest.json": b"{}"}, manifest=False))
    with pytest.raises(ValueError, match="недопустимый путь"):
        MT.import_bundles(mk("b.zip", {"photo/../evil/manifest.json": b"{}"}))
    with pytest.raises(ValueError, match="недопустимый формат"):
        MT.import_bundles(mk("c.zip", {"photo/x/model.pkl": b"x"}))
    with pytest.raises(ValueError, match="недопустимый путь"):
        MT.import_bundles(mk("d.zip", {"other/x/manifest.json": b"{}"}))


def test_export_refuses_unsafe_member(sandbox, tmp_path):
    _bundle(sandbox, "photo", "p1", {"model.pkl": b"x"})
    with pytest.raises(ValueError, match="не допускается"):
        MT.export_bundles([("photo", "p1")], tmp_path / "o.zip")


def test_auc_with_ci_by_group():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(scale=0.7, size=200)
    g = np.repeat(np.arange(40), 5)
    a, lo, hi = MT.auc_with_ci(y, s, g, n_boot=100)
    assert 0.8 < a < 1 and lo < a < hi
    assert np.isnan(MT.auc_with_ci(np.ones(5), np.arange(5.0))[0])
