from sd import resources as R


def test_system_name_outside_system_folders_is_flagged():
    assert R.suspicious("lsass.exe", r"C:\Program Files (x86)\Common Files\lsass.exe")
    assert R.suspicious("lsass.exe", r"C:\Windows\System32\lsass.exe") == []
    assert R.suspicious("svchost.exe", r"C:\Windows\System32\svchost.exe") == []
    assert R.suspicious("explorer.exe", r"C:\Users\u\AppData\Local\obsidian-updater\explorer.exe")


def test_fake_windows_update_in_appdata_and_miner_files_next_to_it():
    exe = r"C:\Users\u\AppData\Local\Microsoft\Windows\Update\MicrosoftUpdateWorker.exe"
    why = R.suspicious("MicrosoftUpdateWorker.exe", exe, ["MicrosoftUpdateHealth.exe", "miner.exe", "runtime_donate.json", "taskmgr_hook.dll"])
    assert len(why) == 2 and "AppData" in why[0] and "miner.exe" in why[1]
    assert R.suspicious("python.exe", r"C:\Python313\python.exe", ["python313.dll"]) == []


def test_snapshot_shape_and_advice():
    s = R.snapshot(top=3, min_mb=1)
    assert {"ram_total_gb", "ram_available_gb", "swap_used_gb", "cpu_percent", "disk_free_gb", "top", "enough_for_vlm"} <= set(s)
    assert len(s["top"]) <= 3 and all("suspicious" in r for r in s["top"])
    low = dict(s, ram_available_gb=0.2, enough_for_vlm=False, top=[dict(name="x.exe", pid=1, mb=2000, suspicious=["причина"])], swap_used_gb=20.0, ram_total_gb=7.7)
    a = R.advice(low)
    assert any("мало" in x for x in a) and any("x.exe" in x and "антивирус" in x for x in a) and any("подкачке" in x for x in a)
