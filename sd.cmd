@echo off
rem Wrapper: sd.cmd <command> ...   (runs the CLI from the project venv, no activation needed)
set PYTHONUTF8=1
set SD_ENV_READY=1
set OPENCV_FFMPEG_LOGLEVEL=-8
set OPENCV_LOG_LEVEL=ERROR
"%~dp0.venv\Scripts\python.exe" -m sd %*
