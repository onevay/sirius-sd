# Данные и веса: что уже есть, что можно скачать вручную

Все пути — относительно корня проекта `C:\Users\OneVay\Documents\smoking-detection`.

## 1. Что уже лежит в проекте

| Что | Где | Статус |
|---|---|---|
| Ваши видео (31 шт., ~60 мин) | `data/курение` (19), `data/лжекурение` (12) | есть; слабая метка = имя папки |
| Веса позы YOLO26 n/s/m/x, YOLO11 n/s/m | `models/*.pt` | скачаны, sha256 в `models/registry.lock.json`, скан безопасности `ok` |
| X-CLIP base/32-16 (MIT), VideoMAE base K400 (CC-BY-NC-4.0) | `models/hf/` (кэш Hugging Face) | скачаны (safetensors — код не исполняют) |
| Детекторы сигареты (community) | `models/smoking_*.pt`, `models/smoke_cig_*.pt` | скачаны, **прошли статическое сканирование** pickle (см. ниже) |
| Backbone'ы фото-классификатора: CLIP ViT-B/32 (LAION-2B, MIT), ConvNeXt-B (Meta, ImageNet-22k, Apache-2.0) | `models/hf/hub/…`, IR для OpenVINO — `models/ov/photo_*` | скачаны (safetensors — код не исполняют); реестр: `clip-vit-b32`, `convnext-b-in22k` |
| Фото-наборы «курит / не курит» | `data/Smoker Detection` (1120), `data_external/stanford40/stanford40_actions` (2316), `data_external/mendeley/smoking_vs_notsmoking_2400` (2010 с метками + 400 без) | скачаны публичные, без входа в аккаунт (`sd.cmd photos fetch …`), лицензия — в `LICENSE.txt` каждого набора |
| CigDet (557 снимков с рамками сигарет, YOLO) | `data_external/mendeley/cigdet` | скачан публичный (`sd.cmd photos fetch cigdet`, CC BY 4.0, sha256 сверен); тест детектора предмета, не фото-набор |
| HMDB51: `smoke`, `drink`, `eat`, `chew`, `talk`, `smile` (по 60 клипов) | `data_external/hmdb51/hmdb51_org/<класс>/` | скачаны из зеркала Hugging Face `divm/hmdb51` без входа (`sd.cmd fetch-videos hmdb51 --per-class 60`); только внутренняя оценка |
| Пакеты моделей | `models/photo/<имя>`, `models/cycle/<имя>` | без pickle (JSON/текст), переносятся zip-экспортом вкладки «Модели» |

Статус и проверка в любой момент: `sd.cmd models list`, `sd.cmd models scan <id> --details`.

### Безопасность весов `.pt`
`.pt` — это pickle: при загрузке он может выполнить произвольный код. Ultralytics грузит чекпойнты именно так. Поэтому community-`.pt`
перед использованием сканируются (`src/sd/safety.py`: опкоды pickle читаются без исполнения; допускаются только классы torch/ultralytics.nn/numpy/collections,
`getattr` — только по шаблону `getattr(<голова Ultralytics>, 'forward')`, точечные имена запрещены). Если вы кладёте вес вручную — прогоните
`sd.cmd models scan <путь>` до первого запуска. Форматы `safetensors` и `onnx` код не исполняют.

## 2. Что имеет смысл скачать вручную (по приоритету)

Скачивание с Kaggle/Roboflow требует вашей учётной записи (токен/логин) — я их не использую и не ввожу. Кладите файлы так, как описано ниже, и запускайте `sd.cmd datasets` — он покажет, что увидел.

> **Приватность:** записи с объекта организаторов нельзя загружать в облако (Colab, Roboflow, Kaggle-ноутбуки, облачные VLM). Публичные датасеты — можно.
> Ваши ролики из `data/` по виду — реальные камеры наблюдения; не выкладывайте их в облачные сервисы без разрешения владельца.

### Какие открытые данные брать для ЭТОГО проекта (выбор по критериям)
Что нужно конвейеру от открытых данных и чего нет в ваших клипах: **затяжки и сложные негативы из одних и тех же источников** (в ваших данных все затяжки из папки «курение», а негативы
в основном из «лжекурения» — см. `ANALYSIS.md`, 7.6: VideoMAE выучил сцену), **много разных людей и сцен**, **видео, а не картинки** (цикл «рука ко рту» виден только во времени).
Критерии отбора: (1) оба класса из одного источника; (2) видео с людьми ≥ 80 px хотя бы местами (мельче можно получить уменьшением — так мы имитировали удаление);
(3) метка хотя бы на уровне клипа; (4) лицензия, которую можно записать в README сдачи.

| # | Источник | Что даёт | Как взять минимум | Лицензия | Подводные камни |
|---|---|---|---|---|---|
| 1 | **Kinetics-400** (классы `smoking` №316, `smoking hookah` №317) | клипы по 10 с из YouTube; негативы из того же набора: `drinking`, `drinking beer`, `drinking shots`, `tasting beer`, `eating burger/cake/…`, `texting`, `blowing nose`, `yawning` | **не качать целиком** (K700-2020: train 603 ГБ). Выборка по классам через FiftyOne zoo (`classes=[…]`, `max_samples=`) — ролики подтягиваются с YouTube (`yt-dlp`), часть удалена | аннотации CC BY 4.0, ролики остаются под лицензиями авторов (для исследований) | **VideoMAE из `models/hf` дообучен на train Kinetics-400 → оценивать признаки VideoMAE только на val-части или на других наборах**; в основном крупные планы и вебкамеры, не CCTV |
| 2 | **HMDB51** (`smoke` ≈ 100 клипов; негативы `drink`, `eat`, `chew`, `talk`, `smile`) | один небольшой архив (≈ 2 ГБ), без YouTube; низкое разрешение (240 px по высоте) — удобный стресс-тест на мелких людей | `hmdb51_org.rar` со страницы датасета (Brown) или зеркала на Hugging Face (`Serrelab/hmdb51`, `divm/hmdb51`) | исследовательская (CC BY-NC-SA / CC BY 4.0 — на зеркалах указано по-разному, проверьте страницу источника) | ≈ 100 затяжек — мало для обучения, достаточно для быстрой проверки конвейера |
| 3 | **AVA v2.2** (метки `smoke`, 3 528 экземпляров; негативы `drink`, `eat`, `answer phone`, `talk to`) | **рамка человека + действие каждую секунду** — формат ближе всего к задаче «событие на человека»; позволяет считать IoU и совпадение по времени | аннотации `ava_v2.2.zip` + отдельные ролики с CVDF (`https://s3.amazonaws.com/ava-dataset/trainval/<файл>`); выбрать 10–20 роликов с наибольшим числом `smoke` в CSV | аннотации CC BY 4.0; сами фильмы — для исследований | кино: ракурсы близкие, резкие склейки (трекинг ломается), ролики по 15 мин — большие |
| 4 | Картинки для детектора предмета (Roboflow / Kaggle / Mendeley, список ниже) | только если recall детектора на кропах (48–55%, `ANALYSIS.md` 7.5) нужно поднимать | см. «Приоритет 1» и «3» | CC BY 4.0 / MIT на страницах | дообучение на этом CPU нецелесообразно; на Kaggle/Colab — **только публичные** датасеты |
| — | UAV-Human (класс `smoking`, съёмка с дрона) | только если на площадке камера сверху/дрон | по запросу авторов (GitHub SUTDCV/UAV-Human) | исследовательская | тяжёлый, ракурс не CCTV — пока пропускаем |

**Рекомендация.** Начать с пары «HMDB51 (быстро, без интернета-ловушек) + Kinetics-400 val по 100–150 клипов `smoking` и по 50 на каждый из 6–8 негативных классов». AVA — вторым шагом, когда нужен
event-F1 по рамкам. Ваши `data/` остаются dev-набором для порогов; открытые клипы идут в обучение/кросс-валидацию **как отдельные источники** (группа = клип, колонка `source`).

**Как положить и проверить.** `data_external/<источник>/<датасет>/<класс>/клип.mp4` (имя папки-класса = метка: `smoke`, `smoking`… → курение, любая другая → «не курение»; свои имена —
`--positive`), `LICENSE.txt` рядом. Дальше:
```bash
sd.cmd datasets                                          # что увидел: число клипов, размер, лицензия
sd.cmd clips-run data_external/hmdb51/hmdb51_org --per-class 20 --max-sec 10   # конвейер по клипам, сводка по классам
```
`clips-run` пишет `outputs/external/<набор>/clips.csv` (доля клипов с циклом/событием по классам, циклов в минуту на человека, рост человека) и `cycles.parquet` — таблицу циклов для `sd.cmd train`.
Тот же прогон на ваших данных: `sd.cmd clips-run data --max-sec 30` (классы `курение` / `лжекурение`).

### Приоритет 1 — детекторы сигареты/вейпа (для прямого признака, Roboflow Universe, экспорт **YOLOv8**)
Нужны, если проверка community-детекторов (раздел 7.5 `docs/ANALYSIS.md`) покажет низкий recall на кропах кисть–рот.

| Датасет | Ссылка | Лицензия (на странице) |
|---|---|---|
| Cigarette Vape Detection (telai) | https://universe.roboflow.com/telai/cigarette-vape-detection-xvtxe | CC BY 4.0 |
| Cigarette Vape Detection (hugefissure) | https://universe.roboflow.com/hugefissure/cigarette-vape-detection-lagrc | CC BY 4.0 |
| vape-cigarette-detection-2 (stelar), ~1150 изобр. | https://universe.roboflow.com/stelar/vape-cigarette-detection-2 | CC BY 4.0 |
| cigarette-detection (mi-privado) | https://universe.roboflow.com/mi-privado/cigarette-detection-iggcm | MIT |

Как: на странице датасета → *Download Dataset* → формат **YOLOv8** → *download zip*. Распаковать в
`data_external/roboflow/<имя_датасета>/` (должны получиться `data.yaml`, `train/`, `valid/`, `test/`). Рядом положить `LICENSE.txt` со строкой лицензии.
Лицензию каждого датасета нужно указать в README сдачи (требование кейса).

### Приоритет 2 — готовые видео с курением (для проверки автомата циклов и вектора признаков)
* **HMDB51**, класс `smoke` (109 клипов в зеркале `divm/hmdb51`) и двойники `drink`, `eat`, `chew`, `talk`, `smile`: **`sd.cmd fetch-videos hmdb51 --per-class 60`** (без входа в аккаунт, ≈ 83 МБ на шесть классов целиком) → `data_external/hmdb51/hmdb51_org/<класс>/*.mp4`; карточка зеркала — `license: other` (у зеркала Serrelab/hmdb51 — CC BY 4.0), оригинал исследовательский: использовать для внутренней оценки.
* **Kinetics-400/700**, класс `smoking` (через FiftyOne zoo, скачивание с YouTube — выбрать 50–100 клипов). `data_external/kinetics/smoking/*.mp4`.
  Это крупные планы (не CCTV) — полезны для калибровки d_t и ритма затяжек, но не заменяют ваши съёмки.

### Приоритет 3 — картинки «курит / не курит» (классификатор кропа, проверка признака «внешний вид»)
* Kaggle/Mendeley **Smoker Detection** (1120 изображений 250×250: 560 smoking / 560 not-smoking, в not-smoking — питьё, телефон, ингалятор): https://data.mendeley.com/datasets/j45dj8bgfc/1 → `data_external/kaggle/smoker_detection/`.
* **CigDet** (557 изображений с рамками сигарет): https://data.mendeley.com/datasets/6hyrr8typ7 — **уже скачан**: `sd.cmd photos fetch cigdet` → `data_external/mendeley/cigdet/` (`data.yaml`, train 446 / test 111, CC BY 4.0, Ali Khan 2024).

### Фото-наборы, уже подключённые (этап 3; публичные, без входа)
| Набор | Откуда | Что в нём | Лицензия | Команда |
|---|---|---|---|---|
| **Smoker Detection** (1120: 560/560, 250×250) | Mendeley Data [j45dj8bgfc](https://data.mendeley.com/datasets/j45dj8bgfc/1) | стоковые портреты; «не курит» — вода, телефон, ингалятор, кашель | на странице набора | уже в `data/Smoker Detection` |
| **Stanford 40 Actions** (в выборке 2316: `smoking` 241; «трудные» негативы: playing_violin 260, phoning 259, blowing_bubbles 259, drinking 256, looking_through_a_telescope 203, brushing_teeth 200, taking_photos 197, texting_message 193; ещё по 8 снимков из 31 другого действия) | зеркало Hugging Face [zrchen03/Stanford40_Dataset](https://huggingface.co/datasets/zrchen03/Stanford40_Dataset) (Stanford40.zip, 308 МБ; удаляется после распаковки) | реальные фото людей в разных сценах; у «курит» официальные `train/test`-списки | зеркало — MIT; оригинал (Yao et al., ICCV 2011) — исследовательский: для коммерции проверить отдельно | `sd.cmd photos fetch stanford40` |
| **Smoking vs Not-smoking, 2400** (2010 с метками: 1005/1005; 400 тестовых без меток) | Mendeley Data [7b52hhzs3r](https://data.mendeley.com/datasets/7b52hhzs3r/1) (651 МБ, `.rar`) | «не курит» содержит похожие жесты: вода, ингалятор, телефон, обкусывание ногтей | **CC BY 4.0** (указывать авторство: Ali Khan, 2020, DOI 10.17632/7b52hhzs3r.1) | `sd.cmd photos fetch mendeley_smoker_2400` (распаковка системным bsdtar; остаются только картинки) |
| **CigDet** (557, рамки сигарет YOLO: train 446 / test 111) | Mendeley Data [6hyrr8typ7](https://data.mendeley.com/datasets/6hyrr8typ7/1) (35 МБ, zip) | 557 снимков класса «курит» набора Smoker Detection (тот же автор), по одной рамке сигареты; без негативов, пересекается с «Smoker Detection» — **не** фото-набор «курит / не курит» (`labels.csv` не создаётся) | **CC BY 4.0** (указывать авторство: Ali Khan, 2024, DOI 10.17632/6hyrr8typ7.1) | `sd.cmd photos fetch cigdet`, затем `sd.cmd detector-eval <детектор> --data data_external/mendeley/cigdet/data.yaml --split test` |
| HF `ccclllwww/smoking_img_final` (тест: 188 курит / 103 нет) | [Hugging Face](https://huggingface.co/datasets/ccclllwww/smoking_img_final) | независимая проверка; лицензия не указана | не указана → только внутренняя оценка | `sd.cmd photos fetch smoking_img_final_test` (≈ 96 МБ) |

**Что выяснил аудит (`sd.cmd photos audit`).** Внутри «Smoker Detection» 2 пары дублей между train и test (обе «курит»); в «2400» — **292 группы почти-дубликатов внутри набора** (606 снимков) и 123 группы, пересекающиеся с «Smoker Detection» (тот же автор): обычное CV и перенос между этими наборами завышены — все оценки в проекте считаются по **группам дублей**. Классификатор, которому дали только метаданные (размер/яркость/резкость), получает AUC 0.82 («Smoker Detection»), 0.85 («2400»), 0.68 (Stanford40) — наборы содержат шорткаты, поэтому проверяется и качество внутри страт яркости, и перенос между наборами (`ANALYSIS.md`, раздел 9).

### Датасеты с рамками и «неоднозначными» случаями: ссылки для ручной загрузки (нужен вход в аккаунт)
Roboflow Universe закрыт Cloudflare-проверкой, а Ultralytics Platform и Kaggle отдают файлы только после входа — **обходить это я не буду**, ниже ссылки, которые нашёл. Скачанное кладите в `data_external/<источник>/<имя>/` с `LICENSE.txt` и проверяйте `sd.cmd datasets`.

| Датасет | Ссылка | Что даёт | Как скачать |
|---|---|---|---|
| **cigarrette_detection** (cigarette) — ваш пример | https://universe.roboflow.com/cigarette/cigarrette_detection | детектор сигареты | Download Dataset → **YOLOv8** → `data_external/roboflow/cigarrette_detection/` |
| **smoking** (unihos-monitor) — ваш пример | https://universe.roboflow.com/unihos-monitor-jq4ff/smoking-o8lph | курение/сигарета, рамки | то же → `data_external/roboflow/unihos_smoking/` |
| **Cigarette detection** (ivisionbeta), 1230 изобр. | https://universe.roboflow.com/ivisionbeta/cigarette-detection-5g76q | **неоднозначные случаи**: классы `pen`, `cigarette_in_hand`, `cigarette_near_mouth`, `cigarette_spotted`, `negative` — ручка у рта против сигареты | то же → `data_external/roboflow/ivisionbeta_cigarette/` |
| smoking (thaicharslsp) | https://universe.roboflow.com/thaicharslsp/smoking-ff95p | курение, рамки | то же |
| smoking-detection (smoking-bycpp) | https://universe.roboflow.com/smoking-bycpp/smoking-detection-8fgr8 | курение, рамки | то же |
| cigarette-detection (mi-privado), MIT | https://universe.roboflow.com/mi-privado/cigarette-detection-iggcm | детектор сигареты | то же |
| **limc/smoking** (Ultralytics Platform), 3708 изобр., 3868 рамок, ≈ 130 МБ | https://platform.ultralytics.com/limc/datasets/smoking | одноклассовый детектор сигареты (происхождение — Roboflow; лицензия на странице не указана) | после входа: Download → YOLO-формат → `data_external/ultralytics/limc_smoking/` (публичный API отдаёт только список и миниатюры) |
| **CigDet**, 557 изобр. (446/111), рамки YOLO | https://data.mendeley.com/datasets/6hyrr8typ7 | сигарета на разных фонах | **уже скачан** — публичный файл Mendeley: `sd.cmd photos fetch cigdet` (входа не требует) |
| **CDAD** — Common Daily Action Dataset with Hard Negatives (CVPRW 2022), 57 824 клипа, 23 действия, жесты-двойники (ручка у рта против курения) | https://mlanthology.org/cvprw/2022/xiang2022cvprw-cdad (ссылка на код в статье — `github.com/MartinXM/CDAD` — **не открылась (404)**) | единственный из найденных видео-наборов, **специально собранный под «трудные» негативы** | искать актуальную ссылку на данные у авторов статьи |

**Встроенная модель Roboflow («built-in sigarete detection»).** Это размещённый на Roboflow API-детектор: его вызов нужен API-ключ (ваш) и отправляет картинки в облако — для записей объекта это запрещено; для публичных фото — на ваше усмотрение, но без ключа я не могу. Без потери смысла: тот же класс задач закрывают три community-детектора из реестра (`smoking_yolo11m_beehzod`, `smoking_yolo26s_basant18`, `smoke_cig_beehzod_yolo11m`, обучены на похожих Roboflow-наборах), а при наличии скачанного YOLO-набора `sd.cmd detector-eval` сравнивает любой детектор на нём (P/R/mAP по рамкам и AUC уровня картинки); дообучение — `yolo train` на публичном наборе (на CPU медленно: обучайте на Kaggle/Colab **только публичные** данные).

**Подключение скачанного YOLO-набора к фото-конвейеру:** `sd.cmd photos from-yolo <каталог набора> --positive <классы-«курит» из data.yaml через запятую>` пишет `labels.csv` (снимок с рамкой нужного класса = «курит», без рамок = `background`, а не «трудный негатив»); проверка детектора на рамках — `sd.cmd detector-eval <детектор> --data <каталог>/data.yaml --split test`. Пошагово, с ожидаемыми результатами — `NEXT_STEPS.md`, блок B.

### Что НЕ нужно
Датасеты дыма/огня (fire & smoke) — это другой физический объект; по литературе (SIT 2024) детекция дыма на CCTV не сработала.

## 3. Ручная загрузка весов
Если автоматическая загрузка недоступна (прокси, обрывы):
1. Скачайте файл из колонки `url` / `hf_repo` в `configs/models.yaml`.
2. Положите в `models/` под именем из колонки `file` (для HF-репозиториев: в каталог кэша `models/hf/hub/…` или просто запустите `sd.cmd models get <id>` позже).
3. `sd.cmd models scan <id>` — проверка безопасности; `sd.cmd models list` — sha256 появится после первой загрузки через `sd.cmd models get`.

Веса, которые стоит добавить, если захотите расширить сравнение: RTMW/DWPose (whole-body, точки губ и пальцев — для дальней зоны не поможет, для ближней да),
VLM-верификатор — через **Ollama** (установлен, сервер только на 127.0.0.1 и без облака): `sd.cmd ollama up --igpu`, `sd.cmd ollama pull qwen3.5:2b-q4_K_M`; веса лежат в `%USERPROFILE%\.ollama\models` (не в `models/` проекта), результаты и выбор модели — раздел 8 `docs/ANALYSIS.md`.

## 4. Лицензии используемых компонентов
| Компонент | Лицензия | Замечание |
|---|---|---|
| Ultralytics YOLO11/YOLO26 (код и веса) | AGPL-3.0 | для коммерции — Enterprise; в отчёте указать, предложить RTMPose (Apache-2.0) как замену |
| rtmlib | Apache-2.0 | веса OpenMMLab: лицензия по репозиторию-источнику |
| X-CLIP base | MIT | |
| VideoMAE base K400 | **CC-BY-NC-4.0** | только некоммерческое использование |
| Qwen3.5 / Qwen3-VL / SmolVLM2 | Apache-2.0 | |
| CLIP ViT-B/32 (LAION-2B, OpenCLIP) | MIT | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, `model.safetensors`; у `openai/clip-vit-base-patch32` safetensors нет (только pickle) — не используется |
| ConvNeXt-B (Meta, ImageNet-22k, через timm) | Apache-2.0 | `timm/convnext_base.fb_in22k_ft_in1k`, safetensors |
| CigDet (рамки сигарет) | CC BY 4.0 | указывать авторство: Ali Khan, 2024, DOI 10.17632/6hyrr8typ7.1 |
| HMDB51 (зеркало `divm/hmdb51`) | карточка зеркала: other; Serrelab/hmdb51: CC BY 4.0; оригинал (Kuehne et al., ICCV 2011) — исследовательский | клипы из фильмов и YouTube: внутренняя оценка, не распространять |
| Фото-наборы | см. таблицу выше | Mendeley «2400» — CC BY 4.0 (указать авторство); Stanford40 — исследовательский; HF-набор без лицензии — только внутренняя оценка |
| Ollama (сервер) | MIT | установщик `OllamaSetup.exe` 0.40.0 с релиза на GitHub, SHA-256 сверен с `sha256sum.txt` релиза, подпись Ollama Inc. (DigiCert) валидна |
| MiniCPM-V 4.6 | проверить по репозиторию OpenBMB/MiniCPM-V | лицензию на странице Ollama не нашёл; пока только для проверки, не для сдачи |
| Community-детекторы сигареты | MIT / Apache-2.0 (по карточкам) | обучены на Roboflow-датасетах — лицензии датасетов см. выше |
