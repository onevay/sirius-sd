import io
import pickle
import sys
import types
import zipfile

import pytest

from sd.safety import classify, scan_torch_archive


class _Evil:
    def __reduce__(self):  # сканер не должен исполнять это при проверке
        import os
        return (os.system, ("echo pwned",))


def _make_pt(path, obj=None, raw: bytes | None = None, protocol=2):
    if raw is None:
        buf = io.BytesIO()
        pickle.dump(obj, buf, protocol=protocol)
        raw = buf.getvalue()
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", raw)
        z.writestr("archive/version", "3\n")


@pytest.fixture()
def fake_detect(monkeypatch):
    """Поддельный модуль ultralytics.nn.modules.head ТОЛЬКО на время теста (pickle требует, чтобы класс был найден по имени модуля).
    monkeypatch возвращает sys.modules в исходное состояние — иначе другие тесты получали бы поддельный модуль вместо настоящего ultralytics."""
    mod = types.ModuleType("ultralytics.nn.modules.head")
    cls = type("Detect", (), {"__module__": "ultralytics.nn.modules.head"})
    mod.Detect = cls
    monkeypatch.setitem(sys.modules, "ultralytics.nn.modules.head", mod)
    return cls


def test_malicious_pickle_is_flagged_without_execution(tmp_path):
    p = tmp_path / "evil.pt"
    _make_pt(p, {"model": _Evil()})
    rep = scan_torch_archive(p)
    assert rep.verdict == "danger"
    assert any("system" in d for d in rep.danger)


def test_plain_state_dict_is_ok(tmp_path):
    import collections
    p = tmp_path / "ok.pt"
    _make_pt(p, collections.OrderedDict(a=1, b=[1, 2, 3]))
    assert scan_torch_archive(p).verdict == "ok"


def test_eval_builtin_is_flagged(tmp_path):
    class E:
        def __reduce__(self):
            return (eval, ("1+1",))

    p = tmp_path / "e.pt"
    _make_pt(p, E())
    assert scan_torch_archive(p).verdict == "danger"


def test_real_torch_save_checkpoint_ok(tmp_path):
    torch = pytest.importorskip("torch")
    p = tmp_path / "t.pt"
    torch.save({"w": torch.zeros(3, 3), "meta": {"epoch": 1}}, p)
    rep = scan_torch_archive(p)
    assert rep.verdict == "ok", rep.summary()


def test_getattr_forward_on_ultralytics_class_allowed(tmp_path, fake_detect):
    pytest.importorskip("ultralytics")
    """Так официальный YOLO11 сохраняет forward головы: getattr(Detect, 'forward')."""
    Detect = fake_detect

    class G:
        def __reduce__(self):
            return (getattr, (Detect, "forward"))

    p = tmp_path / "g.pt"
    _make_pt(p, G())
    rep = scan_torch_archive(p)
    assert rep.verdict == "ok", rep.summary()


def test_getattr_dunder_chain_is_flagged(tmp_path, fake_detect):
    pytest.importorskip("ultralytics")
    Detect = fake_detect

    class G:
        def __reduce__(self):
            return (getattr, (Detect, "__globals__"))  # первое звено цепочки эксплойта

    p = tmp_path / "g2.pt"
    _make_pt(p, G())
    assert scan_torch_archive(p).verdict == "danger"


def test_getattr_on_non_ultralytics_object_flagged(tmp_path):
    import collections

    class G:
        def __reduce__(self):
            return (getattr, (collections.OrderedDict, "fromkeys"))

    p = tmp_path / "g3.pt"
    _make_pt(p, G())
    assert scan_torch_archive(p).verdict == "danger"


def test_dotted_stack_global_is_flagged(tmp_path):
    # protocol 4: STACK_GLOBAL('torch.nn.modules.module', 'warnings.sys') — обход белого списка через qualified name
    def s(x: str) -> bytes:
        b = x.encode()
        return b"\x8c" + bytes([len(b)]) + b

    raw = b"\x80\x04" + s("torch.nn.modules.module") + s("warnings.sys") + b"\x93" + b"."
    p = tmp_path / "d.pt"
    _make_pt(p, raw=raw)
    assert scan_torch_archive(p).verdict == "danger"


def test_classify_rules():
    assert classify("torch.nn.modules.conv", "Conv2d") == "ok"
    assert classify("ultralytics.nn.modules.block", "C3k2") == "ok"
    assert classify("torch._utils", "_rebuild_tensor_v2") == "ok"
    assert classify("torch.nn.modules.module", "warnings") == "unknown"   # не классоподобное имя
    assert classify("os", "system") == "danger"
    assert classify("subprocess", "Popen") == "danger"
    assert classify("builtins", "eval") == "danger"
    assert classify("some.random", "Thing") == "unknown"
