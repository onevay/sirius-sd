import os
import subprocess
import sys

# FFmpeg внутри OpenCV читает OPENCV_FFMPEG_LOGLEVEL только из окружения, заданного ДО старта процесса (из os.environ в рантайме не работает).
# Поэтому при первом запуске перезапускаем себя с нужным окружением — иначе консоль засыпает «non-existing SPS ... buffering period».
if os.environ.get("SD_ENV_READY") != "1":
    env = dict(os.environ, SD_ENV_READY="1", OPENCV_FFMPEG_LOGLEVEL="-8", OPENCV_LOG_LEVEL="ERROR", PYTHONUTF8="1")
    raise SystemExit(subprocess.call([sys.executable, "-m", "sd", *sys.argv[1:]], env=env))

from .cli import main  # noqa: E402

if __name__ == "__main__":
    main()
