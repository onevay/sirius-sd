# Запуск в изолированной среде (Docker) и на другом ПК

Образ содержит только **код и конфиги** (Python 3.13, зависимости строго по `requirements.txt`, CPU-сборка PyTorch). Данные, веса, метки и результаты в образ не попадают — они монтируются томами
(`docker-compose.yml`), поэтому образ не раздувается, записи объекта не копируются внутрь образа, а веса меняются без пересборки.

> **Статус проверки (честно).** На этом ноутбуке Docker Desktop не запущен (демон недоступен), а свободно ≈ 0.3 ГБ ОЗУ и ≈ 10 ГБ диска — собрать образ (≈ 3–4 ГБ загрузок) здесь нельзя, **сборка и запуск контейнера НЕ проверены**. Проверено статически:
> `docker compose config` — файл валиден; `pip install --dry-run --platform manylinux2014_x86_64 --python-version 3.13` по `requirements.txt` — **все пакеты имеют готовые Linux-колёса** (в т.ч. `lap`, `openvino`, `lightgbm`), компилятор не нужен; `sd doctor` и `sd ui --host` проверены в обычном окружении.
> Первая проверка на машине с Docker: `docker compose build && docker compose run --rm cli doctor` (ожидается: все пункты OK, кроме «пакеты моделей», «Ollama» и «видео данных», пока вы их не примонтировали).

## Быстрый старт
```bash
docker compose build                                   # один раз (≈ 10–15 мин, сеть нужна только здесь и для загрузки весов)
docker compose up ui                                   # веб-интерфейс: http://127.0.0.1:8501 (порт опубликован только на этой машине)
docker compose run --rm cli doctor                     # проверка окружения
docker compose run --rm cli models get all             # скачать веса в ./models (или скопировать готовую папку / импортировать zip, см. ниже)
docker compose run --rm cli recognize "/app/data/курение/sm_6.mp4" --start 0 --end 30 \
    --cycle-bundle models/cycle/cycle_v1 --photo-bundle models/photo/photo_v2          # результат: ./outputs/recognize/…
docker compose --profile vlm up -d ollama              # локальный Ollama для VLM (внутри сети compose; наружу не публикуется)
docker compose exec ollama ollama pull qwen3.5:2b-q4_K_M
docker compose run --rm cli recognize … --vlm qwen3.5:2b-q4_K_M --vlm-mode grey --cycle-bundle-full models/cycle/cycle_full
```

## Что где лежит (тома)
| на хосте | в контейнере | режим | зачем |
|---|---|---|---|
| `./data` | `/app/data` | **только чтение** | ваши записи; контейнер их не меняет |
| `./data_external` | `/app/data_external` | чтение/запись | открытые датасеты (`sd photos fetch …` пишет сюда) |
| `./models` | `/app/models` | чтение/запись | веса `.pt`, кэш Hugging Face, пакеты `models/cycle`, `models/photo`, пользовательские модели |
| `./outputs` | `/app/outputs` | чтение/запись | кэши позы, таблицы, результаты `recognize` |
| `./labels`, `./docs/review` | `/app/labels`, `/app/docs/review` | чтение/запись | ваши метки циклов и метки жестов |
| `./configs/experiments`, `./configs/classifiers` | `/app/configs/…` | чтение/запись | профили моделей и спецификации классификатора, создаваемые в вебе (иначе пропали бы при пересоздании контейнера) |

## Изоляция и приватность
* Процесс в контейнере работает от обычного пользователя (uid 1000), не от root; в образе нет данных объекта.
* Интерфейс слушает `0.0.0.0` **внутри** контейнера, но порт публикуется как `127.0.0.1:8501` — из сети он не виден.
* Полностью офлайн (веса уже в `./models`): `docker run --rm --network none -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" -v "$PWD/outputs:/app/outputs" -v "$PWD/docs/review:/app/docs/review" smoking-detection:cpu recognize "/app/data/курение/sm_6.mp4" --start 0 --end 30 --cycle-bundle models/cycle/cycle_v1` — у контейнера нет сети вообще (`docker compose run` флага `--network` не имеет).
* Ollama: `OLLAMA_NO_CLOUD=1`, порт не публикуется, модели в отдельном томе `ollama_models`; приложение находит его по `SD_OLLAMA_URL=http://ollama:11434` (в обычном окружении по умолчанию `127.0.0.1:11434`).
* Телеметрия Hugging Face/Ultralytics выключена переменными окружения; сканирование `.pt` и проверка пакетов моделей (`sd.model_tools`) работают так же, как вне контейнера.

## Ускорители
* **Intel iGPU (OpenVINO GPU)** в контейнере на Windows/Mac недоступна (виртуальная машина без проброса); на Linux-хосте: `--device /dev/dri` и пакеты Intel compute-runtime в образе (в базовый образ не включены). Без ускорителя OpenVINO работает на CPU — эмбеддинги/поза медленнее, чем на iGPU в 3–10 раз.
* **NVIDIA:** готовая надстройка `docker-compose.gpu.yml` (сборка с CUDA-PyTorch cu124, `SD_TORCH_DEVICE=cuda`, резервирование GPU): `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build ui`; на хосте — драйвер и NVIDIA Container Toolkit (Docker Desktop: WSL2 с GPU). Профиль для такой машины — `laptop_gpu4` (не измерен на ПК автора). Сборка образов НЕ проверялась (на машине автора демон Docker выключен); проверено только `docker compose config`.
* **Всё делается в вебе:** страницы «Задачи» (проверка окружения, загрузка весов, признаки, оценки, лестница), «Классификатор», «Модели» — консоль внутри контейнера не нужна.
* Потоки: `SD_THREADS` (по умолчанию 2). Память: на слабой машине задайте `mem_limit` сервиса, чтобы контейнер не вытеснил систему в своп; для Docker Desktop (WSL2) ограничьте `memory=` в `%UserProfile%\.wslconfig`.

## Перенос на другую машину без Docker или с ним
1. **Пакеты моделей** (классификаторы цикла и фото-модели, без pickle): вкладка «Модели» → «Перенос на другую машину» → экспорт в zip (`outputs/export/…`) → на другой машине импорт того же zip (проверка путей, расширений, sha256 из манифеста; ничего не записывается, пока проверка не прошла целиком). Или `models/cycle/<имя>/`, `models/photo/<имя>/` копируются как есть.
2. **Веса** (`.pt`, кэш HF): `sd.cmd models get all` на новой машине (нужна сеть) либо копирование `models/` целиком; `.pt` из чужих рук — только после `sd.cmd models scan`.
3. **Проверка:** `sd doctor` (в контейнере: `docker compose run --rm cli doctor`) — сверяет версии пакетов с `requirements.lock.txt`, видит ли OpenVINO GPU/CUDA, на месте ли веса и пакеты моделей, запущен ли Ollama.
4. **Добавить и проверить новую модель** (например, детектор с другой машины): вкладка «Модели» → «Подключить свою модель» (файл `.pt` сканируется, `.onnx`/`.safetensors` принимаются) → «Проверить модель»; из командной строки — `sd.cmd detector-eval <путь-к-весам> --data data_external/…/data.yaml`.

## Типичные проблемы
| симптом | причина / что делать |
|---|---|
| `no matching manifest` / долгая сборка | первая сборка тянет ≈ 3 ГБ: torch CPU (≈ 200 МБ), openvino, transformers, ultralytics; повторные — из кэша слоёв |
| кириллица в путях `data/курение` | в образе `LANG=C.UTF-8`, `PYTHONUTF8=1`; на Windows-хосте путь к проекту лучше без пробелов |
| `OpenVINO GPU не найден`, всё медленно | ожидаемо внутри Docker Desktop: используется CPU; для iGPU запускайте `sd` напрямую на хосте |
| `Ollama не запущен` в интерфейсе | профиль `vlm` не поднят: `docker compose --profile vlm up -d ollama`; кнопка «запустить сервер» в контейнере не работает по замыслу (сервер внешний) |
| права на `./outputs` | том создаётся пользователем Docker; при ошибках записи: `chown -R 1000:1000 outputs models labels` на Linux-хосте |
