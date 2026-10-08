# smoking-detection — детекция курения по видео

Событийный детектор: **человек (трек) → циклы жеста «рука ко рту» → события по регламенту** (2 цикла за 20 с или цикл + прямой признак, склейка ≤ 15 с, разных людей не склеиваем).
Это каркас и результаты предварительного анализа к хакатону (метрика — Event F1 на скрытом наборе). Руководство по реализации — `Детекция курения по видео — руководство по реализации.pdf`.

| документ | что внутри |
|---|---|
| [docs/MVP.md](docs/MVP.md) | **итог итерации MVP (8.10.2026): что готово, как запустить и показать, честные числа на событиях, что не помогло, лестница вычислений («что даст мощный ПК»), чек-лист до скрытого набора** |
| [docs/AUDIT_AND_CHANGES.md](docs/AUDIT_AND_CHANGES.md) | **аудит версии, что изменено в интерфейсе/разметке/оценке, как пользоваться, анализ улучшений** |
| [docs/REALTIME.md](docs/REALTIME.md) | **мониторинг и real-time**: камеры-папки `район-индекс-время`, движок потока, тревоги, приложение оператора, ограничения |
| [docs/SOLVER_AND_STREAM.md](docs/SOLVER_AND_STREAM.md) | **решатель** `.sdsolver.zip` (перенос на другое устройство, метаданные архитектуры), обязательный классификатор, слияние VLM/предмета, `sd replay`, профили под 2 ПК, код-ревью |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | **запуск и перенос между устройствами**: установка на любой ОС, `.env`, переменные `SD_*`, переносимые пути, чек-лист переноса, Docker |
| [docs/ANALYSIS.md](docs/ANALYSIS.md) | **главный файл с выводами**: источники и подходы, EDA, качество каждой части пайплайна на ваших видео, железо и VLM, решения и следующие шаги |
| [docs/CLI.md](docs/CLI.md) | как проверять каждую часть пайплайна из командной строки, что рисуется на видео |
| [docs/DATA_AND_WEIGHTS.md](docs/DATA_AND_WEIGHTS.md) | какие веса уже скачаны, что можно скачать вручную (Kaggle/Roboflow), лицензии, безопасность `.pt` |
| [docs/TRAINING.md](docs/TRAINING.md) | подготовка обучения классификатора циклов: разметка → таблица → LightGBM → порог |
| [docs/STATE.md](docs/STATE.md) | **текущее положение**: что сделано и проверено, ключевые числа и их границы, что не сделано или заблокировано, окружение, как воспроизвести |
| [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) | **дальнейшие шаги** по источникам данных: загрузка → разметка → проверка детекции / распознавания / классификации, с командами, ожидаемыми результатами и критериями решений |
| [docs/DOCKER.md](docs/DOCKER.md) | запуск в изолированной среде (Docker) и перенос на другой ПК |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | журнал прогонов: дата, команда, результат, решение |

## Установка и запуск
```bash
python scripts/bootstrap.py --dev --cpu-torch   # любая ОС: .venv, зависимости, .env, проверка (--minimal — без torch/ultralytics)
./sd.sh <команда>   |   make ui | app | monitor | test   # Linux/macOS;  Windows: sd.cmd <команда>
```
Пути к данным, весам и результатам можно вынести переменными `SD_*` или файлом `.env` — [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Быстрый старт (Windows, окружение активировать не нужно)
**MVP одной командой** (после `sd.cmd doctor`): `python -m sd.run --input <папка_с_клипами> --profile mvp --out preds.csv --render out_videos/`; интерфейс — `sd.cmd ui` (Оценка · Просмотр · Разметка · Анализ · Классификатор · Модели · Задачи · Этапы); профили для машин — `laptop_cpu`, `laptop_gpu4`.

```bash
sd.cmd hw                                   # железо и какие VLM помещаются в память
sd.cmd models list                          # веса: скачаны ли, лицензии, результат проверки безопасности
sd.cmd info                                 # метаданные всех видео из data/
sd.cmd pose sm_3 --start 20 --end 50 --frame-at 30     # этап 1: поза+трекинг, скриншот с точками и процентами
sd.cmd cycles sm_3 --start 20 --end 50      # этап 3: автомат циклов, видео с состояниями
sd.cmd analyze-cycles                       # предмет, X-CLIP и VideoMAE по всем найденным циклам (долго: несколько минут на каждую модель)
sd.cmd feature-auc                          # какие признаки отделяют курение от ложных жестов: AUC с интервалами, проверка на смешение со сценой, абляция
sd.cmd ollama up --igpu                     # локальный Ollama для VLM (127.0.0.1, без облака; встроенная графика ×2 быстрее)
sd.cmd vlm-eval --per-class 12              # VLM на размеченных циклах: AUC, решение при 0.5, время на цикл
sd.cmd event-check                          # сквозная проверка: циклы → оценка → события → протокол оценки (эталон из меток)
sd.cmd clips-run data_external/<источник>/<датасет> --per-class 20   # конвейер по папкам-классам (открытые датасеты)
sd.cmd recognize data/курение/sm_6.mp4 --start 0 --cycle-bundle models/cycle/cycle_fast   # полный цикл: поза → циклы → оценка → события → видео с разметкой (outputs/recognize/…)
sd.cmd photos fetch stanford40              # открытые фото «курит / не курит»; дальше audit, embed, eval, train — docs/NEXT_STEPS.md
sd.cmd fetch-videos hmdb51 --per-class 60   # открытые видео (курение против питья/еды из одного источника) → sd.cmd clips-run …
sd.cmd start-eval                           # качество детекции начала цикла по меткам жестов
sd.cmd doctor                               # проверка окружения (на другой машине / в контейнере)
sd.cmd resources                            # что занимает ОЗУ/CPU (подозрительные процессы помечаются, чужие не останавливаются)
sd.cmd ui                                   # веб-интерфейс: Оценка · Просмотр · Разметка · Анализ · Классификатор · Модели · Задачи · Этапы: http://localhost:8501
sd.cmd eval -d data/курение -d data/лжекурение -p baseline   # Event F1 по эталону событий на выбранных папках (журнал outputs/experiments/)
sd.cmd gt -d data/курение                   # состояние эталона событий (labels/events_gt.csv)
sd.cmd monitor --watch --speed 1            # мониторинг: камеры = папки streams/<район>-<индекс>-<время начала>, тревоги → outputs/monitor/monitor.db
sd.cmd app                                  # интерфейс оператора (дашборд тревог, подтверждение): http://localhost:8502
sd.cmd monitor-demo                         # демо-тревоги для работы над интерфейсом без моделей
python -m sd.run --input clips/ --profile final --out preds.csv   # единая команда запуска: preds.csv + manifest
sd.cmd test                                 # юнит-тесты
```
Видео из `data/` можно называть коротким именем (`sm_3`, `5`) или `video_id` (`курение__5`). Все команды и опции — [docs/CLI.md](docs/CLI.md).

## Окружение
* **Ollama 0.40.0** установлен для VLM (`%LOCALAPPDATA%\Programs\Ollama`, модели в `%USERPROFILE%\.ollama\models`); сервер поднимает `sd.cmd ollama up`, он слушает только 127.0.0.1 и работает без облака. В автозагрузке пользователя появился ярлык Ollama (Диспетчер задач → Автозагрузка, если не нужен).
* Python 3.13, проектный `.venv` (CPU-сборка torch 2.7.1, ultralytics 8.4.x с YOLO26, OpenVINO, ONNX Runtime, rtmlib, transformers, LightGBM, Streamlit).
  Создание с нуля: `python -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt && .venv\Scripts\python -m pip install -e . --no-deps`.
  `requirements.txt` — верхний уровень с версиями, `requirements.lock.txt` — полный `pip freeze`.
* ПК: i3-1115G4 (2 ядра), 7.7 ГБ ОЗУ, встроенная Intel UHD, **без NVIDIA** — модели подбирались под CPU/OpenVINO (см. ANALYSIS.md, раздел «Железо»).
* Конфиг: `configs/default.yaml` — все пороги в одном месте; переопределение из CLI `-s секция.ключ=значение`.
* **Внимание:** на этом ПК обнаружен криптомайнер (`docs/NEXT_STEPS.md`, шаг 0; `docs/ANALYSIS.md`, раздел 10, пункт 7) — проверьте машину антивирусом; записи с площадки до очистки сюда не копируйте.

## Структура
```
configs/default.yaml          пороги и параметры (не в коде)         configs/models.yaml   реестр весов (id, url, лицензия, доверие)
src/sd/
  video_io.py                 чтение с таймкодами контейнера, запись H.264
  pose_track.py tracks.py     поза+трекинг (ultralytics torch/OpenVINO, rtmlib), склейка треков
  features.py cycles.py       d_t «запястье–рот», автомат циклов (потоковый)
  evidence.py tube.py         предмет на кропе кисть–рот; X-CLIP/VideoMAE по «трубке» человека
  events.py evaluate.py       события по регламенту; копия протокола оценки организаторов
  dataset.py train.py         вектор цикла, метки, LightGBM+калибровка+абляция
  analysis.py feature_auc.py  массовый прогон предмета/X-CLIP/VideoMAE по циклам; AUC признаков, проверка на смешение со сценой
  vlm.py vlm_eval.py          VLM-верификатор (клиент Ollama, оценка по логитам ответа) и его проверка на размеченных циклах
  ollama_ctl.py               сервер Ollama: запуск только на 127.0.0.1 и без облака, статус, загрузка/выгрузка моделей
  event_check.py clips.py     сквозная проверка событий по эталону из меток; прогон конвейера по папкам-классам (в т.ч. открытые датасеты)
  stages.py render.py cli.py  этапы с кэшем артефактов, рендер точек/процентов на видео, CLI
  pipeline.py                 полный цикл распознавания `sd recognize` (поза → циклы → признаки → оценка пакетами → события)
  pose_feats.py bundle.py     признаки положения точек; пакеты классификатора цикла без pickle (логрегрессия, LightGBM, MLP)
  cycle_models.py             наборы признаков и сравнение ансамблей (`sd compare-sets`, `sd train-bundle`)
  start_eval.py pool_features.py   качество начала цикла: пул кандидатов, метки жестов, признаки пула
  photo_*.py cli_photos.py    фото-датасеты: загрузка, аудит, эмбеддинги (OpenVINO), классификатор, перенос на кропы видео
  detector_eval.py video_fetch.py clips.py   проверка детекторов предмета; открытые видео; прогон по клипам «папка = класс»
  model_tools.py doctor.py resources.py      подключение и проверка своих моделей, экспорт/импорт пакетов; окружение; ресурсы
  gt.py evaluation.py runner.py   эталон событий; метрики вокруг F1 (порог, интервал, бюджет ошибок, причины); прогон выбранных папок и журнал экспериментов
  profiles.py catalog.py roi.py   профили моделей (configs/experiments/*.yaml), каталог доступных моделей, зона интереса
  run.py                      `python -m sd.run`: единая команда запуска (preds.csv + manifest)
  ui/                         Streamlit: app.py (навигация), page_eval / page_view / page_label / page_models / page_stages, labeler (видео-разметчик)
  eda.py bench.py hw.py       EDA, бенчмарки, отчёт о железе
  safety.py models.py         сканер .pt перед загрузкой, реестр и загрузка весов
tests/                        323 юнит-теста (циклы, события, evaluate, эталон и оценка, профили, поток и мониторинг, интерфейс, фото-конвейер, пакеты моделей, окружение, безопасность весов, обучение): sd.cmd test
Dockerfile docker-compose.yml  запуск в контейнере (docs/DOCKER.md); сборка не проверялась
data/  models/  outputs/      видео, веса, артефакты запусков (не коммитить; приватность)
```

## Перед публикацией репозитория (GitHub и др.)
В локальном репозитории уже есть коммит и настроен `origin` (`github.com/onevay/sirius-sd`); на момент проверки (7.10, 22:00) **ничего не отправлено** (нет `refs/remotes`). Что в коммите и что с этим делать **до** первого `git push`:
* **Кадры из записей площадки** — четыре иллюстрации в `docs/img/` (`pose_sm6_t12.jpg`, `cycles_night_ir_1.jpg`, `cycles_group_canopy.jpg`, `vlm_crops_mouth_vs_tube.jpg`; на них люди и время съёмки камеры, на них ссылается `docs/ANALYSIS.md`). Запись объекта нельзя выкладывать в облачные сервисы без письменного разрешения владельца — убрать из репозитория (и из ссылок в документах) или получить разрешение.
* **`data_external/`** (6209 файлов, ≈ 0.8 ГБ: Stanford40, Mendeley, CigDet, HMDB51) — чужие наборы: HMDB51 и Stanford40 исследовательские (не распространять), Mendeley и CigDet — CC BY 4.0 (нужно указать авторство). Они скачиваются командами `sd.cmd photos fetch …` и `sd.cmd fetch-videos …`, в репозиторий им не место; в `.gitignore` добавлены, но уже закоммиченные файлы остаются отслеживаемыми.
* Не отслеживаются (проверено): `data/` (ваши записи), `models/`, `outputs/`, `labels/*.csv`, `.venv/`.
Убрать из ещё не отправленного коммита (выполняете вы; одна команда на строку):
```bash
git rm -r --cached data_external
git rm --cached docs/img/pose_sm6_t12.jpg docs/img/cycles_night_ir_1.jpg docs/img/cycles_group_canopy.jpg docs/img/vlm_crops_mouth_vs_tube.jpg
git commit --amend -m "For github"
git reflog expire --expire=now --all
git gc --prune=now
```
Если репозиторий уже где-то опубликован — удаление из последнего коммита не поможет (файлы остаются в истории): нужна очистка истории и, для кадров, решение владельца записей.

## Приватность и лицензии
* Обработка полностью локальная; телеметрия Ultralytics/HF выключена. Записи с объекта не загружать в облачные сервисы без письменного разрешения владельца.
* Ultralytics YOLO — AGPL-3.0; VideoMAE — CC-BY-NC-4.0 (некоммерческое). Полная таблица — [docs/DATA_AND_WEIGHTS.md](docs/DATA_AND_WEIGHTS.md).
