from sd import profiles as PR
from sd.photo_feats import torch_device
from sd.pose_track import _ort_device


def test_torch_device_falls_back_to_cpu(monkeypatch):
    torch = __import__("pytest").importorskip("torch")

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


def test_final_profile_is_valid_and_gpu_variant_is_an_override():
    p = PR.load("mvp")
    c = p.cfg()
    assert c["pose"]["runtime"] == "openvino" and c["video"]["process_fps"] == 5 and c["pose"]["refine"]["gate_s"] == 3.0
    assert c["evidence"]["runtime"] == "openvino" and p.opts()["objects"] == ["smoking_yolo11m_beehzod"] and p.threshold == 0.55
    gpu = PR.with_overrides(p, ["pose.runtime=torch", "pose.device=cuda:0", "pose.imgsz=1600", "pose.weights=yolo26s-pose", "pose.refine.device=cuda", "evidence.runtime=torch", "evidence.device=0"])
    g = gpu.cfg()
    assert g["pose"]["device"] == "cuda:0" and str(g["evidence"]["device"]) == "0" and gpu.fingerprint() != p.fingerprint()
    fast = PR.with_overrides(p, ["video.process_fps=10", "options.cycle_bundle=models/cycle/cycle_mvp10"])
    assert fast.cfg()["video"]["process_fps"] == 10 and fast.opts()["cycle_bundle"].endswith("cycle_mvp10")
