"""`sd resources` — что занимает память и процессор перед тяжёлым запуском (VLM, эмбеддинги) и освобождение ТОЛЬКО своих ресурсов.

Принцип: чужие процессы эта команда не останавливает и не меняет — она их показывает (с путём и подозрительными признаками), решение за пользователем.
Освобождается то, что запустил сам проект: выгрузка моделей из памяти Ollama (`--free-own`), по желанию остановка его сервера.
Подозрительные признаки — только эвристики (не вердикт антивируса): имя системного процесса вне системных папок, исполняемый файл в AppData под именем «Microsoft/Windows Update»,
в одном каталоге с процессом лежат файлы `miner*`/`*donate*`.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import os
from pathlib import Path

import psutil

CORE_NAMES = {"lsass.exe", "csrss.exe", "winlogon.exe", "services.exe", "svchost.exe", "explorer.exe", "ctfmon.exe", "fontdrvhost.exe", "smss.exe", "wininit.exe",
              "mousocoreworker.exe", "lsaiso.exe"}          # настоящие файлы лежат только в C:\Windows\…
THIRD_PARTY_NAMES = {"msedgewebview2.exe"}                  # допустимо и в Program Files
WINDOWS_DIR = "c:\\windows\\"
SYSTEM_DIRS = (WINDOWS_DIR, "c:\\program files\\", "c:\\program files (x86)\\")
OWN_MARKERS = ("smoking-detection", "streamlit", " -m sd", "\\sd\\")


def suspicious(name: str, exe: str, siblings: list[str] | None = None) -> list[str]:
    """Причины, по которым процесс стоит проверить антивирусом (эвристики; пусто = ничего необычного по этим признакам)."""
    n, e = (name or "").lower(), (exe or "").lower().replace("/", "\\")
    why = []
    if e and ((n in CORE_NAMES and not e.startswith(WINDOWS_DIR)) or (n in THIRD_PARTY_NAMES and not e.startswith(SYSTEM_DIRS))):
        why.append(f"имя системного процесса ({name}), но файл вне системных папок: {exe}")
    if e and "\\appdata\\" in e and ("microsoft\\windows\\update" in e or "microsoftupdate" in n):
        why.append("«Microsoft Update» запускается из AppData (настоящие компоненты Windows Update лежат в C:\\Windows)")
    bad = [s for s in (siblings or []) if any(k in s.lower() for k in ("miner", "donate", "xmrig", "taskmgr_hook"))]
    if bad:
        why.append(f"рядом с файлом процесса: {', '.join(sorted(set(bad))[:4])}")
    return why


def _siblings(exe: str) -> list[str]:
    try:
        return [p.name for p in Path(exe).parent.iterdir()][:200] if exe else []
    except OSError:
        return []


def snapshot(top: int = 10, min_mb: float = 150.0) -> dict:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    procs = []
    for p in psutil.process_iter(["pid", "name", "exe", "memory_info", "cmdline"]):
        try:
            mi = p.info["memory_info"]
            if mi is None:
                continue
            exe = p.info["exe"] or ""
            cmd = " ".join(p.info["cmdline"] or [])
            procs.append(dict(pid=p.info["pid"], name=p.info["name"] or "", mb=round(mi.rss / 2**20), exe=exe, own=any(m in (cmd + exe).lower() for m in OWN_MARKERS)))
        except (psutil.Error, OSError):
            continue
    procs.sort(key=lambda r: -r["mb"])
    big = [r for r in procs if r["mb"] >= min_mb][:top]
    for r in big:
        r["suspicious"] = suspicious(r["name"], r["exe"], _siblings(r["exe"]) if r["name"].lower().startswith(("microsoftupdate", "svchost")) or r["exe"].lower().find("\\appdata\\") >= 0 else None)
    return dict(ram_total_gb=round(vm.total / 2**30, 1), ram_available_gb=round(vm.available / 2**30, 2), swap_used_gb=round(sw.used / 2**30, 1),
                cpu_percent=psutil.cpu_percent(interval=0.5), disk_free_gb=round(psutil.disk_usage(os.path.abspath(os.sep)).free / 2**30, 1), top=big,
                enough_for_vlm=vm.available / 2**30 >= 2.5)


def free_own(stop_ollama: bool = False) -> list[str]:
    """Освобождает только своё: выгрузка моделей из Ollama (если он отвечает); `stop_ollama` — ещё и остановка его сервера (если он локальный)."""
    from . import ollama_ctl as O

    done = []
    if O.is_up():
        names = O.unload()
        done.append(f"Ollama: выгружены модели {names}" if names else "Ollama: в памяти моделей не было")
        if stop_ollama and not O.EXTERNAL:
            done.append(f"Ollama: остановлено процессов {O.stop()}")
    else:
        done.append("Ollama не запущен — выгружать нечего")
    return done


def advice(snap: dict) -> list[str]:
    out = []
    if not snap["enough_for_vlm"]:
        out.append(f"свободно {snap['ram_available_gb']} ГБ — для VLM (Qwen3.5-2B q4_K_M ≈ 3.4 ГБ в общей памяти iGPU) мало: закройте браузер/мессенджеры или запускайте VLM только для серой зоны")
    for r in snap["top"]:
        if r["suspicious"]:
            out.append(f"{r['name']} (PID {r['pid']}, {r['mb']} МБ): " + "; ".join(r["suspicious"]) + " → проверьте антивирусом; проект этот процесс не трогает")
    if snap["swap_used_gb"] > snap["ram_total_gb"]:
        out.append(f"в подкачке {snap['swap_used_gb']} ГБ при ОЗУ {snap['ram_total_gb']} ГБ — все расчёты идут в разы медленнее")
    return out
