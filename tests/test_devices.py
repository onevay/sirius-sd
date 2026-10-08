from sd import profiles as PR
from sd.photo_feats import torch_device
from sd.pose_track import _ort_device


def test_torch_device_falls_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setenv("SD_TORCH_DEVICE", "cuda")
    assert torch_device(torch) == ("cuda" if torch.cuda.is_available() else "cpu")
    monkeypatch.setenv("SD_TORCH_DEVICE", "cpu")
    assert torch_device(torch) == "cpu"
    monkeypatch.delenv("SD_TORCH_DEVICE")
    assert torch_device(torch) == "cpu"


def test_ort_device_cpu_by_default():
    assert _ort_device({}) == "cpu"
    assert _ort_device({"device": "cpu"}) == "cpu"
    assert _ort_device({"device": "cuda"}) in ("cpu", "cuda")


def test_machine_profiles_are_valid_and_differ():
    cpu, gpu = PR.load("laptop_cpu"), PR.load("laptop_gpu4")
    c, g = cpu.cfg(), gpu.cfg()
    assert c["pose"]["runtime"] == "openvino" and c["pose"]["imgsz"] == 1280
    assert g["pose"]["runtime"] == "torch" and g["pose"]["device"] == "cuda:0" and g["pose"]["imgsz"] >= 1600
    assert g["evidence"]["device"] == "0" and g["pose"]["refine"]["device"] == "cuda"
    assert cpu.fingerprint() != gpu.fingerprint()
    assert cpu.opts()["cycle_bundle"] == gpu.opts()["cycle_bundle"]
    assert cpu.threshold == gpu.threshold == 0.55
