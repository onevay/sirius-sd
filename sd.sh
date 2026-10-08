#!/usr/bin/env bash
# Обёртка для Linux/macOS (аналог sd.cmd): sd.sh <команда> ...  — запускает CLI из .venv проекта, активировать окружение не нужно.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUTF8=1 SD_ENV_READY=1 OPENCV_FFMPEG_LOGLEVEL=-8 OPENCV_LOG_LEVEL=ERROR
if [ -x "$HERE/.venv/bin/python" ]; then PY="$HERE/.venv/bin/python"; else PY="${PYTHON:-python3}"; fi
exec "$PY" -m sd "$@"
