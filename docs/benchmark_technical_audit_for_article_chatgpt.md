# Technical audit: benchmark / dataset / evaluation (ответ для подготовки статьи)

Документ составлен по запросам из `questions_for_cursor.txt`. Основан на коде репозитория `svg-segmentator`, файле **`dataset (1).jsonl`** в корне репо, документации в **`docs/`**, каталоге **`svg_gt_baseline/`**, скриптах **`scripts/`**, архиве **`_unpacked_openrouter_samples_v2_20260508/`** (README там же).

**Дисклеймер:** где данных в репозитории нет или явно не считалось — указано «не зафиксировано в репо» / «не посчитано». Цифры по JSONL пересчитаны локально по **`dataset (1).jsonl`** (не по имени `dataset.cleaned.jsonl` из README OpenRouter — это может быть тот же экспорт с другим именем файла; **идентичность двух файлов не верифицировалась**).

---

## Задача 1 — структура benchmark / данных

### 1. Какие данные используются

| Артефакт | Роль |
|----------|------|
| **`dataset (1).jsonl`** | Экспорт аннотаций в стиле сервиса (`DatasetAnnotationRow` в `svg_gt_baseline/dataset_io.py`): одна строка JSON на запись. |
| **`svg_annotation_library.zip`** / **`_unpacked_svg_annotation_library/`** | Библиотека SVG + **`svg_archive_manifest.json`** (`svg_id`, `filename_in_archive`, …). Загрузка путей: `svg_gt_baseline/dataset_io.load_manifest_paths`. |
| **`output/gt_baseline_full/`** (пример полного прогона) | **`predictions.ndjson`**, **`previews/`**, **`run_config.json`**, файлы метрик `metrics_*.json/csv/ndjson`. |
| **`_unpacked_openrouter_samples_v2_20260508/`** | Архив LLM samples v2: **`predictions/*.jsonl`**, **`manifests/eligible_svg_samples_le32768.jsonl`**, README со статистикой прогона. |

**Количество уникальных `svg_id` в `dataset (1).jsonl`:** **268** (по всем строкам файла).

**Количество строк JSONL:** **2275** непустых строк.

**Фильтр для инференса/метрик бейзлайна в коде:** строки с **`review_decision == "declined"`** отбрасываются (`svg_gt_baseline/dataset_io.filter_rows`). После фильтра: **2271** строка, **268** уникальных `svg_id` (четыре declined не выкидывают целый SVG из множества id — они точечные строки).

**«Queries» в смысле текстовых запросов к модели:**

- В каждой строке поле **`prompt_texts`**: список строк (часто один элемент).
- Число **отдельных DINO/LLM задач** при полном охвате строк датасета ≈ сумма **`len(prompt_texts)`** по отфильтрованным строкам = **3326** (посчитано по `dataset (1).jsonl`, kept).

**Train / val / test split:** в коде загрузки датасета и в просмотренных доках **явного split не найдено**. Бенчмарк в текущем виде — **одна совокупность аннотаций**, без обязательного разбиения для обучения.

**Именование / форматы:**

- Датасет: **JSONL**, UTF-8, одна запись = один объект JSON.
- SVG в библиотеке: **`{svg_id}.svg`** согласно манифесту (`filename_in_archive`).
- Превью прогона: **`previews/<svg_id>/ann_<annotation_id>/p<prompt_index>/<detector_id>/<segmenter_id>/`** — см. `docs/gt_baseline_metrics.md`, `docs/baseline_gt_dataset_inference.md`.

**Несовпадение чисел «SVG в zip» vs «SVG в датасете»:** в распакованном архиве **292** записи манифеста / файла `.svg` (факт распаковки в сессии разработки); в **`dataset (1).jsonl`** фигурируют **268** уникальных `svg_id`. Вывод: **не каждый файл библиотеки имеет строки в этом JSONL** (или не весь архив использовался при экспорте). Точное соответствие множеств **не проверялось скриптом в этом документе**.

---

## 2. Что является ground truth

### Канон в экспорте (строка JSONL)

**GT задаётся как бинарный вектор по узлам SVG**, не как bbox и не как отдельный подграф в файле датасета.

Обязательные поля, которые читает код (`svg_gt_baseline/dataset_io.load_dataset_rows`):

| Поле | Тип / смысл |
|------|-------------|
| **`svg_id`** | int, ключ файла в библиотеке |
| **`annotation_id`** | int, уникальность на уровне экспорта (не проверяется глобально в этом модуле) |
| **`annotation_kind`** | string: минимум три значения в данных — см. ниже |
| **`node_count`** | int **N** = длина **`binary_mask`** |
| **`binary_mask`** | список **0/1** длины **N**, позиция = индекс узла в каноническом обходе дерева |
| **`prompt_texts`** | список строк запросов |
| **`object_key`**, **`original_filename`** | метаданные происхождения файла |
| **`review_decision`** | `null`, `"approved"`, `"declined"` (и др. — как приходит из сервиса) |

Дополнительно в данных присутствуют **`prompt_ids`**, **`created_at`**, пользователи ревью и т.д. — см. сырую строку JSON.

**Инвариант по смыслу разметки (документ сервиса):** единицы в **`binary_mask`** допустимы **только на листьях** дерева; промежуточные узлы в маске — нули. См. **`docs/svg-gt-annotation-service.md`** §1.

### Что GT **не** является в самом JSONL

- **Не** хранится отдельный вырезанный SVG-подграф в строке датасета (подграф **строится при инференсе/превью** из маски: prune → `gt_extracted.svg`).
- **Не** хранится bbox детектора в GT-экспорте (bbox появляются у **предикта** DINO в `predictions.ndjson`).
- **Не** хранится отдельный «рендер эталона» в JSONL; PNG **`gt_extracted.png`** пишутся в **`previews/`** при прогоне (`docs/gt_baseline_metrics.md` §2.3).

### Три представления «одной маски» в протоколе метрик

Жёстко разведены в **`docs/gt_baseline_metrics.md`** §2:

1. **Вектор по узлам** (`binary_mask` ↔ `pred_mask_combined`) — **IoU_leaf, F1, exact**.
2. **OR масок листьев на рабочем растре** — оверлеи, **не** обязательные численные метрики сводки.
3. **Cairo PNG вырезки** `gt_extracted.png` ↔ `extracted_union.png` — **MSE/MAE/PSNR/SSIM** (и ROI-варианты в `eval_gt_baseline_metrics`).

### Особый случай: `one_visible_of_many`

- В датасете: несколько строк с одним **`(svg_id, tuple(prompt_texts))`** и видом **`one_visible_of_many`**, разные **`annotation_id`**, разные маски (например несколько окон под промпт `"window"`).
- В отчёте **`eval_gt_baseline_metrics`**: строки этого вида **не** идут как обычные **`canonical_row`**; считается **`oom_merged`** с GT = **OR** масок членов группы (нужны **`oom_merged_*/meta.json`** от **`scripts/write_all_oom_gt_merged.py`**). См. **`docs/gt_baseline_metrics.md`** §5, §8.

### Пример записей (без полного массива `binary_mask`)

**Пример A — declined (в прогон не входит):** первая строка файла — `dataset (1).jsonl` строка 1:
- `svg_id`: 1, `annotation_id`: 1, `annotation_kind`: `multiple_visible_objects`, `node_count`: 6, маска короткая `[0,0,0,1,0,0]`, `prompt_texts`: `["star"]`, **`review_decision`: `"declined"`**.

**Пример B — типичная строка (kept):** `svg_id`: 278, `annotation_id`: 1053, `annotation_kind`: `multiple_visible_objects`, `node_count`: 11, маска из одиниц на хвосте, `prompt_texts`: `["gifts"]`, `review_decision`: null.

**Пример C — несколько промптов:** `svg_id`: 68, `annotation_id`: 173, `multiple_visible_objects`, `prompt_texts`: `["white clouds", "clouds"]`, `node_count`: 142.

**Пример D — one_visible_of_many:** `svg_id`: 68, `annotation_id`: 216, `prompt_texts`: `["window"]`, `node_count`: 142 (семь единиц в маске — одно «видимое окно» из группы похожих).

---

## 3. Как создавался GT

**Источник правды по процессу:** **`docs/svg-gt-annotation-service.md`** — описание отдельного приложения (фронт/бэк/БД, MinIO, PostgreSQL).

**Характер создания:**

- **Ручная разметка в UI:** дерево элементов SVG, проекция черновика в вектор маски длины **N**, сохранение в БД.
- **Не автоматическая сегментация как источник GT** в этом документе не заявлена; автоматика в репозитории относится к **инференсу бейзлайна** и **LLM**, не к построению эталона.

**Постобработка на стороне датасета:**

- Поле **`review_decision`** и связанные поля ревьюера — человеческая модерация (хотя бы для части строк: в данных **90** строк с `"approved"`, **2181** с `null`, **4** `"declined"` — распределение по **всем** 2275 строкам файла).

**Экспорт в JSONL:** предполагается экспорт из сервиса (конкретная команда экспорта в этом репозитории **не найдена** — есть только потребление файла).

**Нормализация порядка промптов (локально в проекте):** для строк **`one_visible_of_many`** в репозитории выполнялась сортировка пар **`prompt_texts`/`prompt_ids`** для согласования групп с LLM v2 (история в чате). Это **меняет ключ группировки** для OOM-merge относительно «как выгрузили из сервиса». Если статья опирается на исходный экспорт без этой правки — **явно уточнить версию датасета**.

---

## 4. Что является одной benchmark sample (точное определение — зависит от слоя)

### Уровень строки датасета (JSONL)

**Одна строка** = один объект **`(svg_id, annotation_id)`** с одним вектором **`binary_mask`** и списком **`prompt_texts`**.

Это **не** всегда совпадает с «одним запросом к модели»:

- Для **каждого индекса** в `prompt_texts` бейзлайн делает **отдельный** вызов DINO (см. **`docs/baseline_gt_dataset_inference.md`** §2).
- В **OpenRouter samples v2** один промпт из строки с несколькими `prompt_texts` становится **отдельным sample** внутри одного HTTP-ответа на `svg_id` (см. **`_unpacked_openrouter_samples_v2_20260508/README.md`** §«Новая семантика»).

### Уровень офлайн-метрик (`eval_gt_baseline_metrics`)

**Одна строка отчёта** (`canonical_row` или `oom_merged:…`) соответствует ключу из NDJSON, как минимум:

**`(svg_id, annotation_id, prompt_index, detector_id, segmenter_id)`**  
(для LLM-моста — упрощённые записи через `build_llm_prediction_record`, см. **`docs/llm_samples_v2_baseline_eval_bridge.md`**).

Для **`one_visible_of_many`** canonical-строк по членам группы в основной сводке нет — см. §2 документации выше; одна строка **`oom_merged`** на группу × пара детектор/сегментатор при выполнимости условий.

**Итог для статьи:** формулировать «sample» нужно **в двух режимах**: (а) запись датасета; (б) задача оценки после моста/инференса (prompt_index развёрнут).

---

## 5. Есть ли ambiguous cases

Зафиксировано в типах **`annotation_kind`** и в доках:

| Вид | Смысл (для трактовки) |
|-----|------------------------|
| **`exactly_one_visible_object`** | Ожидается один выделенный объект под семейством промптов (несколько текстовых формулировок одной цели). |
| **`multiple_visible_objects`** | Допускается несколько видимых частей/объектов под запрос. |
| **`one_visible_of_many`** | Несколько экземпляров одной семантики на сцене; при оценке группа может сливаться в OR для merged-метрик. |

**Реальные явления в данных:**

- **Несколько валидных объектов одного класса** на одном SVG — кейс **`one_visible_of_many`** (пример: много окон, `svg_id` 68, группа с промптом `"window"` — см. док **`docs/gt_baseline_metrics.md`** §10.4 и строку `annotation_id` 216 выше).
- **Вложенность / группы в SVG** — структура дерева; маска «только на листьях» перекладывает сложность на выбор листьев в UI (**`docs/svg-gt-annotation-service.md`**).
- **Пересечение масок разных аннотаций** на одном файле: **не посчитано** в репозитории автоматически; потенциально возможно при перекрывающихся объектах.
- **Дубликаты семантики** — несколько строк с близкими промптами (`"door"` / `"brown door"` на `svg_id` 68, `annotation_id` 177) — это **разные задачи** (разные `prompt_index`), а не дубликат одной строки.

**Расхождение датасет ↔ LLM v2 для OOM:** при разном порядке `prompt_texts` у членов группы v2 может отдавать merged-sample с `source_annotation_ids`, для которого нет ровно совпадающего `oom_merged` meta на диске — мост логирует warning (обсуждалось в разработке; частично лечится канонизацией порядка промптов).

---

## 6. Статистика benchmark

### По файлу **`dataset (1).jsonl`** (пересчёт)

**Строки и фильтр:**

| Метрика | Значение |
|---------|----------|
| Всего строк | 2275 |
| `review_decision == declined` | 4 |
| Строк после исключения declined | 2271 |
| Уникальных `svg_id` | 268 |

**`annotation_kind` среди kept:**

| kind | count |
|------|-------|
| `exactly_one_visible_object` | 1032 |
| `multiple_visible_objects` | 643 |
| `one_visible_of_many` | 596 |

**`review_decision` (все строки):** `declined`: 4, `approved`: 90, `null`: 2181.

**Сложность SVG (`node_count` N), kept:**

| stat | value |
|------|-------|
| min | 2 |
| max | 1362 |
| mean | ~158.99 |
| median | 100 |

**Размер GT как число единиц в маске (`sum(binary_mask)`), kept:**

| stat | value |
|------|-------|
| min | 1 |
| max | 238 |
| mean | ~7.69 |

**Промпты:**

| stat | value |
|------|-------|
| `len(prompt_texts)` min / max / mean | 1 / 5 / ~1.46 |
| длина строки промпта (символы), по всем промптам kept: min / max / mean / median | 1 / 55 / ~8.42 / 7 |

**Суммарное число пар «строка × промпт»** (если считать каждый `prompt_index` отдельной задачей): **3326**.

### По README OpenRouter samples v2 (другой срез — LLM прогон)

Файл **`_unpacked_openrouter_samples_v2_20260508/README.md`** (цитата):

- Уникальных SVG в cleaned dataset: **268**
- Взято в прогон (raw SVG chars ≤ 32768): **173** SVG
- Пропущено: **95** SVG
- Sample kinds в выбранных SVG: `exactly_one_visible_object`: 728, `multiple_visible_objects`: 621, `one_visible_of_many`: 127

**Неопределённость:** эти числа относятся к **агрегации v2 samples**, не напрямую к строкам **`dataset (1).jsonl`** без проверки соответствия имени файла датасета (`dataset.cleaned.jsonl` vs `dataset (1).jsonl`).

### Что **не** посчитано в репозитории (явно)

- Распределение **типов примитивов** SVG (`path`, `rect`, …), глубина групп `<g>`, доля `defs` / `clipPath` / `mask` — **нет готового отчёта** в просмотренных артефактах.
- **Категории объектов** в смысле таксономии ImageNet — не используются; «категория» = текст **`prompt_texts`**.
- **Попарное пересечение GT-масок** между аннотациями на одном SVG — **не вычислялось** здесь.

---

## 7. Примеры benchmark cases (representative)

Ниже **идентификаторы** для воспроизведения; полный SVG: **`_unpacked_svg_annotation_library/<svg_id>.svg`** (или по манифесту).

| # | svg_id | annotation_id | annotation_kind | node_count | sum(mask) | prompt_texts (кратко) | Замечание |
|---|--------|----------------|-----------------|------------|-----------|-------------------------|-----------|
| 1 | 1 | 1 | multiple_visible_objects | 6 | 1 | `["star"]` | Строка **declined** — демонстрация ревью. |
| 2 | 278 | 1053 | multiple_visible_objects | 11 | 9 | `["gifts"]` | Малое N, простая сцена. |
| 3 | 68 | 173 | multiple_visible_objects | 142 | 1 | `["white clouds","clouds"]` | Два промпта → две задачи для DINO/LLM. |
| 4 | 68 | 177 | exactly_one_visible_object | 142 | 3 | `["door","brown door"]` | Две формулировки одной цели. |
| 5 | 68 | 216 | one_visible_of_many | 142 | 7 | `["window"]` | Классический OOM-кейс (несколько окон в доках). |
| 6 | 34 | 1831 | one_visible_of_many | 109 | (см. файл) | `["belt","strap"]` после канонизации порядка | История рассинхрона с v2 до сортировки `prompt_texts`. |

Дополнительно в документации эталонно разбирается **`svg_id` 68** (`1554025379.svg`): **`docs/gt_baseline_metrics.md`** §10.4.

**Число primitives / особенности сцены в табличном виде** потребуют парсинга соответствующего `.svg` — не включено в автоматический отчёт здесь.

---

## 8. Ограничения benchmark и пайплайна

### Исключения SVG / задач

- **Длина текста SVG > 32768 символов:** такие SVG **не попали** в OpenRouter samples v2 eligible (`README` в архиве v2). Для LLM с тем же лимитом метрик **нет**.
- **`review_decision == declined`:** строка **не участвует** в GT-бейзлайн прогоне по умолчанию.
- **Расхождение `node_count` и парсера:** строка должна пропускаться с WARN (политика **`docs/baseline_gt_dataset_inference.md`** §8) — иначе маска неконсистентна.

### Unsupported / fragile

- **Сопоставление нескольких боксов DINO с одной GT-маской:** основной протокол сводки использует **`pred_mask_combined`**; per-box метрики в сводке не обязательны (`docs/gt_baseline_metrics.md` §3).
- **OOM без `oom_merged_*/meta.json`:** нет строк **`oom_merged`** в eval для этой группы.
- **Мост LLM v2 ↔ NDJSON:** требует точного совпадения множеств `annotation_ids` для OOM-sample и совпадения `prompt_text` с `prompt_texts[0]` группы — см. **`docs/llm_samples_v2_baseline_eval_bridge.md`** и код **`scripts/write_llm_bridge_examples.py`**.
- **Сравнение LLM и DINO по информации:** LLM видит все задачи SVG в одном запросе; DINO — изолированно по промпту (**`docs/baseline_gt_dataset_inference.md`** §4) — **не apples-to-apples** без оговорки.

### Предположения аннотации

- Канонический порядок узлов и равенство **`node_count`** байтовому SVG при прогоне.
- Единицы маски только на **листьях** (спецификация сервиса).
- Для метрик по PNG: общий **`raster_max_side`** (часто 640) и фиксированные пороги prune GT (**0.5**) vs pred (**score_threshold**, часто **0.8**) — см. **`docs/gt_baseline_metrics.md`**.

---

## Полезные пути (скрипты и конфиги)

| Назначение | Путь |
|------------|------|
| Загрузка датасета | `svg_gt_baseline/dataset_io.py` |
| Типы строк | `svg_gt_baseline/types.py` |
| Инференс GT-бейзлайна | `scripts/run_gt_baseline_inference.py`, пакет `svg_gt_baseline/` |
| Метрики | `scripts/eval_gt_baseline_metrics.py` (в т.ч. **`--reaggregate-from`** / **`--align-svg-ids-from`** для подвыборки без повторного PNG-eval) |
| OOM merged превью | `scripts/write_all_oom_gt_merged.py`, `svg_gt_baseline/oom_gt_merge.py` |
| Мост LLM v2 | `scripts/write_llm_bridge_examples.py`, `svg_gt_baseline/prediction_ndjson.py` |
| Сводная таблица метрик (markdown) | `output/metrics_comparison_llm_vs_baseline.md` |
| Документация протоколов | `docs/baseline_gt_dataset_inference.md`, `docs/gt_baseline_metrics.md`, `docs/llm_samples_v2_baseline_eval_bridge.md`, `docs/svg-gt-annotation-service.md` |

---

## Вопросы из `questions_for_cursor.txt` — покрытие

| Пункт | Где ответ |
|-------|-----------|
| 1 Данные | §«Задача 1» |
| 2 GT | §2 |
| 3 Создание GT | §3 |
| 4 Единица sample | §4 |
| 5 Ambiguity | §5 |
| 6 Статистика | §6 |
| 7 Примеры | §7 |
| 8 Ограничения | §8 |

Конец отчёта.
