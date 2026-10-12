# Запуск, окружение и перенос между устройствами

## 1. Быстрый старт на любой ОС

```bash
python scripts/bootstrap.py --dev --cpu-torch      # .venv + зависимости (CPU torch) + .env + sd doctor
python scripts/bootstrap.py --minimal --dev        # лёгкое ядро БЕЗ torch/ultralytics/openvino/transformers (см. ниже)
```

| ОС | запуск команд | интерфейсы |
|---|---|---|
| Windows | `sd.cmd <команда>` | `sd.cmd ui` (разработчик, :8501), `sd.cmd app` (оператор, :8502) |
| Linux / macOS | `./sd.sh <команда>` или `make <цель>` | то же; `make help` — список целей |
| любая, окружение активировано | `sd <команда>` или `python -m sd <команда>` | |
| контейнер | `docker compose up ui` / `up app` / `--profile monitor up -d monitor` | порты опубликованы только на 127.0.0.1 |

`make`: `setup`, `setup-min`, `ui`, `app`, `monitor`, `test`, `doctor`. Обёртки `sd.cmd`/`sd.sh` сами выбирают `.venv` проекта и выставляют `PYTHONUTF8`, `OPENCV_FFMPEG_LOGLEVEL` (без них FFmpeg засыпает консоль сообщениями).

### Лёгкое ядро (`--minimal`)

Ставит только то, что нужно для **оценки по готовым событиям, разметки, просмотра, мониторинга и интерфейса оператора** (pandas, numpy, scipy, scikit-learn, lightgbm, streamlit, plotly, opencv, imageio-ffmpeg …). Не ставит `torch`, `ultralytics`, `openvino`, `rtmlib`, `transformers`: без них не работает **распознавание** (поза и трекинг) и тесты, которым нужны эти пакеты (они пропускаются). Удобно для машины разметчика и оператора. Проверено: тест-сьют проходит в окружении без этих пакетов (6 пропусков).

## 2. Где что лежит и как это переопределить

Все каталоги по умолчанию лежат внутри проекта. Чтобы вынести большое (данные, веса, результаты) на другой диск, задайте переменные окружения или файл `.env` в корне (шаблон — `.env.example`; уже заданные в окружении переменные `.env` не перезаписывает; `~` и `$VAR` раскрываются):

| переменная | по умолчанию | что |
|---|---|---|
| `SD_ROOT` | каталог проекта | корень (конфиги берутся отсюда) |
| `SD_DATA` | `data/` | ваши записи: папки `курение/`, `лжекурение/`, … |
| `SD_EXTERNAL` | `data_external/` | открытые датасеты |
| `SD_MODELS` | `models/` | веса, пакеты, кэш HF и OpenVINO |
| `SD_OUTPUTS` | `outputs/` | кэши позы, прогоны, журнал экспериментов, база мониторинга, копии видео |
| `SD_LABELS` | `labels/` | разметка (`events_gt.csv`, `cycle_labels.csv`) |
| `SD_STREAMS` | `streams/` | имитация камер для мониторинга |
| `SD_MONITOR_DB` | `$SD_OUTPUTS/monitor/monitor.db` | база тревог (например, общий диск для воркера и приложения) |
| `SD_OLLAMA_URL` | `http://127.0.0.1:11434` | локальный Ollama для VLM |
| `SD_PROXY_CODEC` | `h264` | `vp8` — копии видео для браузеров без H.264 |
| `SD_THREADS` | — | число потоков вычислений |

## 3. Переносимые пути в артефактах

Раньше мета запусков хранила **абсолютные** пути (`C:\Users\…\data\курение\a.mp4`): скопированные на другую машину результаты не открывались в просмотре, а обучение и анализ не находили видео. Теперь пути пишутся «от корня» — `$DATA/курение/a.mp4`, `$OUTPUTS/runs/…` — и разворачиваются под текущую машину (`paths.portable` / `paths.from_portable`). Старые артефакты с чужими абсолютными путями (в т. ч. Windows-стиля) тоже находятся: если файла по записанному пути нет, ищется по хвосту пути (1–3 последних компонента) в `data/` и `data_external/`. Токены не выводят за пределы корня (`$DATA/../..` игнорируется).

Кэши прогонов оценки (`outputs/eval_cache`) привязаны к `$`-пути, размеру и времени изменения файла: при копировании файлов время меняется — кэш пересчитается, это безопасно.

## 4. Чек-лист переноса на другую машину

Что копировать (по важности):

1. **`labels/`** — ваша разметка. В git не попадает (приватность): без копирования потеряется. Проверка: `sd gt` покажет число строк и отпечаток.
2. **`configs/experiments/`** — профили моделей (в git).
3. **`models/`** — веса и пакеты классификаторов (большие; либо `sd models get all`, либо zip-экспорт со страницы «Модели» — пакеты без pickle, с проверкой sha256).
4. **`outputs/experiments/`** — журнал оценок (маленький, переносимый); `outputs/monitor/` — база тревог и миниатюры; `outputs/runs/` и кэш позы — по желанию (экономят часы пересчёта).
5. **Видео** — в `data/` / `SD_DATA` (в git не лежат).

На новой машине: `python scripts/bootstrap.py …` → при необходимости `.env` → `sd doctor` (покажет, чего не хватает: версии пакетов против lock-файла, веса, пакеты, данные, Ollama) → `sd eval …` или `sd ui`.

Lock-файл `requirements.lock.txt` снят на Windows; платформенные пакеты (`colorama`, `pyreadline3`) помечены `sys_platform == "win32"`, на Linux/macOS они не ставятся. `requirements.txt` — верхний уровень (проверялся на Python 3.13); тесты прошли и на более новых numpy 2.5 / pandas 3.0, то есть жёсткие версии нужны для воспроизводимости чисел, а не для работоспособности.

## 5. Docker

Образ содержит только **код и конфиги** (Python 3.13, зависимости по `requirements.txt`, CPU-сборка PyTorch). Данные, веса, метки и результаты монтируются томами (`docker-compose.yml`), поэтому записи объекта не копируются в образ, а веса меняются без пересборки.

> **Статус проверки.** Образ ни разу не собирался (на машине разработки Docker недоступен). Проверено статически: `docker compose config` валиден; `pip install --dry-run` для Linux показал готовые колёса для всех пакетов. Первая проверка на машине с Docker: `docker compose build && docker compose run --rm cli doctor` (ожидается OK, кроме «пакеты моделей», «Ollama» и «видео данных», пока они не примонтированы). Это пункт E0 в [REPORT.md](REPORT.md).

```bash
docker compose build                                   # один раз (≈ 10–15 мин; сеть нужна только здесь и для весов)
docker compose up ui                                   # Dev Studio: http://127.0.0.1:8501
docker compose up app                                  # приложение оператора: http://127.0.0.1:8502
docker compose --profile monitor up -d monitor         # воркер потока
docker compose run --rm cli doctor                     # проверка окружения
docker compose run --rm cli models get all             # веса в ./models
docker compose run --rm cli recognize "/app/data/курение/sm_6.mp4" --start 0 --end 30     --cycle-bundle models/cycle/cycle_mvp              # результат: ./outputs/recognize/…
docker compose --profile vlm up -d ollama              # локальный Ollama для VLM (наружу не публикуется)
docker compose exec ollama ollama pull qwen3.5:2b-q4_K_M
```

| на хосте | в контейнере | режим | зачем |
|---|---|---|---|
| `./data` | `/app/data` | только чтение | ваши записи |
| `./data_external` | `/app/data_external` | чтение/запись | открытые датасеты |
| `./models` | `/app/models` | чтение/запись | веса, кэш HF, пакеты классификатора и фото-модели |
| `./outputs` | `/app/outputs` | чтение/запись | кэши позы, журнал, результаты, база тревог |
| `./labels` | `/app/labels` | чтение/запись | эталон, метки циклов и метки жестов (`labels/review/` входит в образ, `COPY labels/review`) |
| `./configs/experiments`, `./configs/classifiers` | `/app/configs/…` | чтение/запись | профили и спецификации, создаваемые в интерфейсе |

**Изоляция и приватность:**
- процесс работает от обычного пользователя (uid 1000); в образе нет данных объекта;
- интерфейс слушает `0.0.0.0` внутри контейнера, но порт публикуется как `127.0.0.1:…`;
- полностью офлайн (веса уже в `./models`): `docker run --rm --network none -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" -v "$PWD/outputs:/app/outputs" -v "$PWD/labels:/app/labels" smoking-detection:cpu recognize "/app/data/курение/sm_6.mp4" --start 0 --end 30 --cycle-bundle models/cycle/cycle_mvp`;
- Ollama: `OLLAMA_NO_CLOUD=1`, порт не публикуется, адрес `SD_OLLAMA_URL=http://ollama:11434`;
- телеметрия Hugging Face и Ultralytics выключена переменными окружения.

**Ускорители:**
- Intel iGPU (OpenVINO GPU) в Docker Desktop на Windows и Mac недоступна: OpenVINO работает на CPU, в 3–10 раз медленнее. На Linux — `--device /dev/dri` и Intel compute-runtime (в образ не входит).
- NVIDIA: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build ui` (CUDA-PyTorch cu126, `SD_TORCH_DEVICE=cuda`); нужны драйвер и NVIDIA Container Toolkit. Не проверялось.
- Потоки — `SD_THREADS` (по умолчанию 2). На слабой машине ограничьте память: `mem_limit` сервиса и `memory=` в `%UserProfile%\.wslconfig`.

**Перенос моделей на другую машину** (без Docker и с ним):
1. Пакеты классификатора и фото-модели (без pickle) — вкладка «Модели → Перенос»: экспорт в zip и импорт с проверкой путей и sha256; либо копирование `models/cycle/<имя>/` и `models/photo/<имя>/`. Полный конвейер — решатель `.sdsolver.zip` ([ARCHITECTURE.md](ARCHITECTURE.md), раздел 4).
2. Веса: `sd models get all` или копирование `models/`; `.pt` из чужих рук — только после `sd models scan`.
3. Проверка: `sd doctor` (`docker compose run --rm cli doctor`).
4. Новая модель: «Модели → Подключить» и «Проверить»; из консоли — `sd detector-eval <веса> --data …/data.yaml`.

**Проблемы Docker:**

| симптом | что делать |
|---|---|
| долгая первая сборка | тянет ≈ 3 ГБ (torch CPU, openvino, transformers, ultralytics); повторные — из кэша слоёв |
| кириллица в путях | в образе `LANG=C.UTF-8`, `PYTHONUTF8=1`; на Windows путь к проекту лучше без пробелов |
| «OpenVINO GPU не найден», медленно | ожидаемо внутри Docker Desktop; для iGPU запускайте `sd` на хосте |
| «Ollama не запущен» | поднимите профиль: `docker compose --profile vlm up -d ollama` |
| права на `./outputs` | на Linux-хосте: `chown -R 1000:1000 outputs models labels` |

## 6. Частые проблемы

| симптом | причина и решение |
|---|---|
| `ModuleNotFoundError: ultralytics/torch` при распознавании | поставлено лёгкое ядро (`--minimal`) или нет пакета; нужен полный `bootstrap.py` |
| Консоль засыпана `non-existing SPS …` | запуск мимо обёртки; используйте `sd.cmd` / `sd.sh` / `python -m sd` (он сам перезапускается с нужным окружением) |
| Видео в браузере чёрное | у браузера нет H.264 — `SD_PROXY_CODEC=vp8` |
| «файл эталона занят другим процессом» | разметчик и `feedback-export` пишут один файл по очереди; замок снимается сам через 60 с, если процесс упал |
| Интерфейс не видит правок кода | `sd ui`/`sd app` запускаются без наблюдателя файлов — перезапустите процесс |
| Порт занят | `sd ui --port 8511`, `sd app --port 8512` |
