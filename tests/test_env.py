"""Окружение: телеметрия Ultralytics выключена ДО импорта библиотеки (иначе при чистых настройках первый запуск мог бы отправить анонимное событие)."""
import os
import subprocess
import sys
import textwrap


def test_ultralytics_analytics_disabled_even_with_fresh_settings(tmp_path):
    code = textwrap.dedent("""
        import os
        import sd._env  # noqa: F401
        from ultralytics import settings
        from ultralytics.utils.events import events
        print("SYNC", settings["sync"], "OFFLINE", os.environ.get("YOLO_OFFLINE"), "ENABLED", events.enabled)
    """)
    env = {**os.environ, "YOLO_CONFIG_DIR": str(tmp_path / "cfg"), "PYTHONUTF8": "1"}
    env.pop("YOLO_OFFLINE", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
    line = [x for x in r.stdout.splitlines() if x.startswith("SYNC")]
    assert line, r.stdout + r.stderr
    assert "SYNC True" in line[0] and "OFFLINE 1" in line[0] and "ENABLED False" in line[0]       # настройки «по умолчанию» (sync=True), но события не включаются
    assert (tmp_path / "cfg").exists()                                                            # свой каталог настроек, не C:\tmp\Ultralytics
