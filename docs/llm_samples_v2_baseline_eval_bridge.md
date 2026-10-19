# OpenRouter samples-v2 → тот же `eval_gt_baseline_metrics` (зафиксированные решения)

Документ фиксирует **договорённости** по мосту от архива **`svg_segmentation_openrouter_samples_v2_*.zip`** к каталогу прогона с **`predictions.ndjson`** + **`previews/`**, чтобы запускать **`scripts/eval_gt_baseline_metrics.py`** (и при необходимости **`scripts/write_all_oom_gt_merged.py`**) без опоры на метрики из самого LLM-архива.

Инференс LLM и формат v2 описаны в **`_unpacked_openrouter_samples_v2_20260508/README.md`** (или внутри zip). Здесь — только **наши** решения.

---

## 1. Объём: кого считаем, «skipped» SVG

**Решение:** в основной пайплайн попадают **только** SVG из **`eligible_svg_samples_le32768.jsonl`** (те же ~173, для которых в v2 уже есть строки в **`predictions/*.jsonl`**).

**Почему «skipped» не включаем:** для SVG с сырой длиной текста **> 32768** в v2 **нет** HTTP-ответа модели → **нет** масок в `predictions/*.jsonl`. Считать метрики «как у бейзлайна» не на чем, пока не сделан **отдельный** LLM-прогон с другим лимитом/чанкингом. Это не «забыли», а отсутствие входа.

---

## 2. Что значит «стыковка v2 → одна строка `predictions.ndjson`» (конкретно: `svg_id == 68`, не-OOM)

В v2 **одна строка JSONL на файл** = один HTTP-ответ модели на **весь `svg_id`**. Внутри неё:

- **`samples[]`**: по одному объекту на каждую «задачу» (промпт × аннотация или merged-OOM); у каждого есть `sample_id`, `golden_mask`, `source_annotation_ids`, `source_prompt_indices`, …  
- **`predictions[]`**: параллельно по `sample_id` лежит **`prediction`** (тот же порядок листьев, что у `golden_mask`).

У **нас** в **`predictions.ndjson`** каждая строка — это уже **одна** задача в смысле **`eval_gt_baseline_metrics`**: ключ **`(svg_id, annotation_id, prompt_index, detector_id, segmenter_id)`**.

**Возьмём реальный `svg_id == 68`** (Gemma/Qwen v2, распакованный архив). В `samples[]` для одной и той же аннотации **177** лежат **два** нормальных sample’а (оба `exactly_one_visible_object`, `merged_source_count == 1`):

| У них (v2, внутри одной строки JSONL) | Смысл |
|----------------------------------------|--------|
| `sample_id = "a177_p0_2a72c9a4333a"` | «дверь»: `source_annotation_ids = [177]`, `source_prompt_indices = [0]`, `prompt_text` согласован с нулевым промптом строки датасета |
| `sample_id = "a177_p1_df8582da73d9"` | «коричневая дверь»: те же `[177]`, но `source_prompt_indices = [1]` — **вторая** строка промптов в датасете для той же аннотации |

В **`predictions[]`** для выбранной модели находим записи с теми же `sample_id` и снимаем **`prediction`**.

**Что пишем у нас** — **две** строки NDJSON (две задачи, как в датасете: тот же `svg_id`, та же `annotation_id`, разные `prompt_index`):

| Поле | Строка A (промпт 0) | Строка B (промпт 1) |
|------|---------------------|----------------------|
| `svg_id` | 68 | 68 |
| `annotation_id` | 177 | 177 |
| `prompt_index` | 0 | 1 |
| `prompt_text` | как в **dataset** для `(68, 177, 0)` | как в **dataset** для `(68, 177, 1)` |
| `binary_mask` | из **dataset** для той пары (эталон; можно `assert` = `golden_mask` из соответствующего v2-sample) | то же |
| `pred_mask`, `pred_mask_combined` | **`prediction`** из v2 для `a177_p0_…` | **`prediction`** из v2 для `a177_p1_…` |
| `detector_id` / `segmenter_id` | §8 (например `llm` / `gemma-v2`) | то же |

**Формат строки NDJSON:** старый пайплайн **`run_inference_job`** не трогаем — он по-прежнему пишет полную строку (в т.ч. **`dino_box_index`**, пороги, растр, боксы). Для моста v2→eval используем **новый** хелпер **`build_llm_prediction_record`** в **`svg_gt_baseline/prediction_ndjson.py`**: только поля, которые реально читает **`eval_gt_baseline_metrics`** (идентификаторы задачи, **`binary_mask`**, **`pred_mask_combined`**, плюс **`pred_mask`** = тому же вектору). Дедуп строк в **`eval_metrics`** понимает и старый ключ **`dino_box_index`**, и нейтральный **`box_index`**, и отсутствие обоих (одна строка на ключ). Итого: **один v2-sample с `merged_source_count == 1`** ↔ **ровно одна строка NDJSON** в смысле eval.

### Несколько ответов на один и тот же промпт?

**В v2 нет сценария «один текст промпта → несколько независимых completion’ов на выбор».** На один `svg_id` приходит **одна** строка JSONL = **один** ответ API; внутри неё заранее разобрано много **разных задач**, у каждой свой `sample_id` и **ровно один** массив `prediction`.

Что может путать:

- **Разные `prompt_index` у одной `annotation_id`** (как дверь / коричневая дверь у 177) — это **две разные задачи** в датасете, не «два ответа на один промпт».
- **OOM:** семь строк датасета с текстом «window» дают **одну** merged-задачу в v2 и **один** общий `prediction` на группу.
- **Синонимы в v2** (несколько `sample_kind == one_visible_of_many` с разным `prompt_text`, но тем же merged `golden_mask`): в мосте берём **одну** строку по правилу §5, остальные **не дублируем** в NDJSON — снова не «много ответов», а **много формулировок одной задачи**.

Если бы в NDJSON случайно оказались **две строки с одним ключом** `(svg_id, annotation_id, prompt_index, detector_id, segmenter_id)`, `dedupe_canonical_combined` оставит одну — такой дубль нужно ловить при конвертации, а не считать нормой.

---

## 3. Неполные ответы (`ok: false`, битый JSON, …)

**Решение:** любую строку v2 с **`ok: false`** для данной модели **не конвертируем**: **нет** записей в NDJSON и **нет** превью для этого `svg_id` в этом LLM-прогоне; причину фиксируем в **`manifest_skips.jsonl`** или **`run.log`** (как удобнее в скрипте).

**По факту v2:** у Qwen зафиксирован **1** failed SVG (`svg_id == 5` в **`metrics_summary.json`**); у Gemma — **0**. Другие случаи (пустой `predictions`, несовпадение длины маски) — по правилу выше: **пропуск + лог**, не «нулевая маска», чтобы не портить сводки.

---

## 4–5. OOM: одна группа датасета ↔ NDJSON + откуда pred (тот же `svg_id == 68`)

**§4 — как у нас в пайплайне.** Группа **`one_visible_of_many`** в датасете задаётся по **`(svg_id, кортеж prompt_texts)`**. Для **68** есть классическая «семёрка окон» с одним и тем же текстом промпта **`window`**: семь разных **`annotation_id`** (216, 219, 221, 222, 224, 226, 227 — как в датасете). Скрипт **`write_all_oom_gt_merged`** пишет, например, **`previews/68/oom_merged_<slug>/meta.json`** (`kind: one_visible_of_many_gt_merged`), внутри **`annotation_ids`** и merged GT для превью. **Якорь** для eval — **`annotation_id = min`** по группе → **216**, **`prompt_index = 0`** (для такой группы один текстовый ряд). В фазе **`oom_merged`** **`eval_gt_baseline_metrics`** считает **`leaf_metrics(OR(binary_mask всех членов), pred_mask_combined якоря)`** и сравнивает картинки по путям с **`gt_is_merged_group=True`** (merged GT у якоря под `gt_merged_group_extracted.png`).

**Важно для NDJSON:** индекс строится по **каждому** члену группы. Для каждого **`annotation_id`** из `meta.json` должна существовать **своя** строка NDJSON с **индивидуальной** `binary_mask` из датасета. **`pred_mask_combined` для метрики читается только у строки якоря `(216, 0)`**; на практике удобно **продублировать один и тот же** merged `prediction` на всех членах группы (одинаковый вектор), чтобы не путаться при ручном просмотре — численно используется только якорь.

**§5 — как у них в v2 на том же SVG.** В той же строке JSONL для 68 в `samples[]` есть sample  

- `sample_id = "ovm_68_74abd9af4c4d"`, **`sample_kind == one_visible_of_many`**, **`merged_source_count == 7`**,  
- **`source_annotation_ids = [216, 219, 221, 222, 224, 226, 227]`** — тот же состав,  
- **`prompt_text = "window"`**, **`golden_mask`** — уже **OR** по группе (как наш merged GT).

В **`predictions[]`** берём **`prediction`** для этого `sample_id` — это и есть **один** вектор предсказания на всю OOM-группу.

**Стыковка:** для якоря **`(68, 216, prompt_index=0, detector_id, segmenter_id)`** в NDJSON кладём **`pred_mask` / `pred_mask_combined`** = этот **`prediction`**. Правило выбора sample среди нескольких OOM-кандидатов в v2: **`prompt_text` совпадает с `prompt_texts[0]`** из **`meta.json`** группы (у «окон» это как раз `"window"`). Другие v2-samples с тем же merged GT (синонимы промпта) **в отдельные строки eval не размножаем**. Если подходящего sample нет — OOM-строка для модели **пропускается** + лог.

---

## 6. Порог при рендере `extracted_union` из LLM-вектора

У **полных** строк из **`run_inference_job`** в NDJSON есть **`score_threshold`** и др.; у **минимальной** LLM-строки (**`prediction_ndjson.build_llm_prediction_record`**) этих полей может не быть — на eval это не влияет. Для **бинарного** `prediction` (0/1 на лист) после **`leaf_scores_from_binary_mask`** получаются только скоры **0.0** и **1.0**; любой **`τ` с `0 < τ ≤ 1`** в **`prune_from_scores`** оставляет **то же** множество листьев.

**Решение:** рендерить **`extracted_union.*`** той же цепочкой, что и GT: `leaf_scores_from_binary_mask` → `prune_from_scores`. В доке фиксируем **`τ = 0.5`**, как в коде для ветки GT; любое другое **`τ ∈ (0, 1]`** на бинарном векторе **эквивалентно**.

---

## 7. Превью на диске

**Обязательное для `eval_gt_baseline_metrics`:** те же пути превью, что ожидает скрипт (как в уже существующих прогонах с `predictions.ndjson`):

- **`previews/<svg_id>/raster.png`**, **`source.svg`**;  
- GT: **`ann_<id>/gt_extracted.png`**;  
- предикт: **`ann_<id>/p<prompt_index>/<detector_id>/<segmenter_id>/extracted_union.png`**;  
- OOM: **`oom_merged_<slug>/meta.json`**, у якоря в **`.../p0/<detector>/<segmenter>/`** — **`gt_merged_group_extracted.png`** (после **`write_all_oom_gt_merged`**).

**Дополнительно для глаз:** в той же папке **`.../<detector>/<segmenter>/`** пишем **`extracted_box0.svg`** и **`extracted_box0.png`** (при необходимости **`extracted_box{i}`** для **i > 1** — только если позже появится согласованное правило разбиения). В v2 **нет** геометрии боксов → **правило превью:** **`extracted_box0`** = **те же пиксели**, что **`extracted_union`**. Если в NDJSON нужны поля для инструментов/резюме рядом с полным прогоном — можно дописать **`n_boxes`**, **`zero_boxes`**, **`box_index`** (или оставить только минимальную строку из **`prediction_ndjson`** — eval от этого не зависит).

**Не генерируем** никаких файлов превью **вне** перечисленного выше: в источнике только вектор маски, eval лишнего не использует.

---

## 8. Имена «детектора» и «сегментатора»

**Решение:** завести **свои** стабильные строки **`detector_id` / `segmenter_id`**, чтобы каталоги **`.../<detector_id>/<segmenter_id>/`** не пересекались по имени с уже существующими прогонами в том же родительском **`out`**.

**Пример:** **`detector_id` = `llm`**, **`segmenter_id` = `qwen-v2`** или **`gemma-v2`**. Путь: **`.../p0/llm/qwen-v2/extracted_union.png`**.

Главное — **однозначно** и по желанию **без коллизий** с другими прогонами.

---

## 9. `run_config.json` в каталоге LLM-прогона

**Решение:** **не обязателен** как полная копия бейзлайнового конфига. Если пишем — только **минимум для воспроизводимости**: пути к `dataset_jsonl`, `manifest_json`, **`raster_max_side`**, ссылка на исходный **`predictions/*_samples_v2*.jsonl`**, версия/хеш скрипта моста, дата. Поля **`box_threshold` / `text_threshold` / `score_threshold`** для LLM-ветки **не влияют на метрики** (§6); можно опустить или пометить `n/a`.

---

## 10. Проверка перед полным прогоном

**Решение:** перед массовым eval прогоняем конвертер + метрики на **3–4 заранее выбранных** `svg_id` / парах ann+prompt; сверяем векторные IoU/F1 вручную или коротким одноразовым скриптом с **`leaf_metrics`**.

---

## 11. `metrics_summary.json` внутри zip v2

**Решение:** **оставляем** в архиве как часть поставки ML, **в наших сводках не используем**; эталон чисел — только вывод **`eval_gt_baseline_metrics.py`** после моста.

---

## Краткая таблица решений

| Тема | Решение |
|------|--------|
| Skipped SVG (>32768) | Не входят в мост без отдельного LLM-прогона |
| Строка NDJSON из v2 | Один sample с `merged_source_count==1` → одна запись; минимум — **`build_llm_prediction_record`**; полный engine-формат не трогаем |
| `ok: false` | Не конвертируем; лог/скип |
| OOM | Группы как в датасете + `write_all_oom_gt_merged`; pred из одного v2-sample по `prompt_text == prompt_texts[0]` |
| Порог при рендере LLM→extracted | Бинарный вектор: τ∈(0,1] не меняет листья; в доке **0.5** как в коде для GT |
| Превью | Минимум для eval + **`extracted_box0`** = union; без лишних PNG из другого пайплайна (§7) |
| Имена det/seg | Свои id (**`llm` + `qwen-v2` / `gemma-v2`**), без коллизий с другими прогонами |
| Один промпт — много ответов? | Нет: одна строка v2 на SVG; см. подраздел после §2 |
| `run_config` | Опционально, урезанный |
| QA | 3–4 примера до полного прогона |
| Метрики из архива v2 | Не используем |

---

## Следующий шаг реализации

Один скрипт (имя на усмотрение, например **`scripts/bridge_openrouter_samples_v2_to_eval_layout.py`**): вход — путь к распакованному v2, путь к `dataset.jsonl`, `manifest`, эталонные числа `raster_max_side`; выход — **`--out`** с `predictions.ndjson` + минимальными `previews/` + опциональный урезанный `run_config.json`. Затем **`write_all_oom_gt_merged`** (если OOM) и **`eval_gt_baseline_metrics.py`**.

**Готовые примеры для просмотра:** **`scripts/write_llm_bridge_examples.py`** — собирает `predictions.ndjson` через **`build_llm_prediction_record`**, копирует `raster`/`source`/`gt_extracted` из эталонного каталога превью и рисует **`extracted_union`** / **`extracted_box0`**. В репозитории после прогона: **`output/llm_bridge_examples_svg68_gemma`** — полный набор по строке v2 **Gemma** для `svg_id` 68 (у Gemma на этом SVG многие маски в архиве **нулевые** — превью пустые, это содержимое v2, не ошибка моста); **`output/llm_bridge_examples_svg68_tiny`** — три строки NDJSON (**`--only-annotation-ids 177,232`**, без OOM), собрано из **Qwen** v2 и **`llm`/`qwen-v2`**, чтобы на дверях/окне было **видно ненулевой** предикт.
