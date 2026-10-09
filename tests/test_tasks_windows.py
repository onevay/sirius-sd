"""Регресс: на Windows tasklist отдаёт вывод в кодовой странице консоли; stdout=None ронял страницу «Задачи» (AttributeError: 'NoneType' has no attribute 'splitlines')."""
import subprocess
from types import SimpleNamespace

from sd import tasks as T


def test_alive_survives_none_and_cp866_output(monkeypatch):
    monkeypatch.setattr(T.os, "name", "nt")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=None))
    assert T._alive(1234) is False
    row = '"python.exe","1234","Console","1","10 000 К"\r\n'.encode("cp866")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=row))
    assert T._alive(1234) is True
    assert T._alive(999) is False


def test_device_check_flags_cuda_profile_without_cuda(monkeypatch):
    import sys
    import types

    from sd import profiles as PR
    from sd import solver as SV

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    p = PR.heuristic_profile()
    p.config.update({"pose.runtime": "torch", "pose.device": "cuda:0"})
    assert any(i.code == "device_cuda" for i in SV.check(p, devices=True))
    assert not any(i.code == "device_cuda" for i in SV.check(p))          # по умолчанию (тесты, fake-распознаватели) устройства не проверяются
