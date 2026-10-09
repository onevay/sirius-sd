# Короткие команды (Linux/macOS; на Windows — sd.cmd или `python scripts/bootstrap.py`).   make help
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
SD  = $(PY) -m sd

.PHONY: help setup setup-min ui app monitor test doctor eval
help:            ## список целей
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ —/'
setup:           ## .venv + зависимости (CPU torch) + .env + doctor
	python3 scripts/bootstrap.py --dev --cpu-torch
setup-min:       ## лёгкое ядро без torch/ultralytics (оценка, разметка, мониторинг с готовыми событиями)
	python3 scripts/bootstrap.py --minimal --dev
ui:              ## интерфейс разработчика: http://localhost:8501
	$(SD) ui
app:             ## интерфейс оператора (мониторинг): http://localhost:8502
	$(SD) app
monitor:         ## воркер: имитация камер из SD_STREAMS → тревоги в базу
	$(SD) monitor --watch
test:            ## тесты
	$(PY) -m pytest -q
doctor:          ## проверка окружения
	$(SD) doctor
