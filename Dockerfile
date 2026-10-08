# syntax=docker/dockerfile:1
# Изолированная среда проекта: CPU-сборка (работает на любом ПК с Docker). Веса и данные НЕ запекаются в образ — монтируются томами (docker-compose.yml).
#   docker build -t smoking-detection:cpu .
#   docker run --rm smoking-detection:cpu doctor
# Другая сборка PyTorch (например, CUDA): --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 (и --gpus all при запуске).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SD_ROOT=/app SD_STREAMS=/app/streams SD_ENV_READY=1 OPENCV_LOG_LEVEL=ERROR OPENCV_FFMPEG_LOGLEVEL=-8 SD_THREADS=2

# libgl1/libglib2.0-0 нужны opencv-python; ffmpeg приходит в составе imageio-ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# 1) torch/torchvision отдельно из CPU-индекса (иначе pip тянет CUDA-колёса на гигабайты), 2) остальное по requirements.txt, 3) rtmlib — без зависимостей (он тянет opencv-contrib)
COPY requirements.txt requirements.lock.txt ./
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url ${TORCH_INDEX} \
    && grep -v -E '^(torch|torchvision|rtmlib)==' requirements.txt > /tmp/requirements.docker.txt \
    && pip install -r /tmp/requirements.docker.txt \
    && pip install --no-deps rtmlib==0.0.16

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
# метки и сводки, на которые ссылаются команды по умолчанию (docs/review/*.csv); остальные документы в образ не нужны
COPY docs/review ./docs/review
RUN pip install --no-deps -e . && mkdir -p data data_external models outputs labels streams

# том с весами/данными/результатами монтируется снаружи; пользователь без прав root
RUN useradd -m -u 1000 sd && chown -R sd:sd /app
USER sd

EXPOSE 8501 8502
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s CMD python -c "import sd" || exit 1
ENTRYPOINT ["sd"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8501"]
