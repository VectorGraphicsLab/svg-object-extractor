# Метрики GT-бейзлайна (DINO × сегментатор): что с чем сравнивать

Документ фиксирует **протокол оценки поверх уже сохранённых артефактов** (`predictions.ndjson`, дерево `previews/`). Инференс описан в **`docs/baseline_gt_dataset_inference.md`**. **Оффлайн-агрегация по этому протоколу** в репозитории зафиксирована в **`scripts/eval_gt_baseline_metrics.py`** (вектор §2.1 + пиксельные метрики по паре extracted §2.3); см. **§8**. Для групп **`one_visible_of_many`** перед замером нужны каталоги **`previews/<svg_id>/oom_merged_*/meta.json`** — их массово пишет **`scripts/write_all_oom_gt_merged.py`** (без DINO/SAM).

**Scope:** **основные численные метрики** — по вектору **`pred_mask_combined`** ↔ **`binary_mask`** (или merged GT для OOM), см. §2.1 и §5. **Визуальные численные метрики** (если нужны в отчёте поверх картинок) — **по парам Cairo PNG §2.3** (**`gt_extracted.png`** ↔ **`extracted_union.png`**, для OOM — **`gt_merged_group_extracted.png`** ↔ **`extracted_union.png`** якоря): прежде всего **MSE / MAE / SSIM / PSNR** по пикселям рендера (**не** бинарный IoU по маскам на полном растре §2.2 и **не** обязательный «силуэтный» IoU после порога на PNG — см. §2.3). Оверлеи **`gt_mask`** / **`pred_leaf_mask`** — только для глаз §2.2. Записи **`pred_mask`** / **`extracted_box{i}.*`** по каждому боксу DINO **сохраняются для отладки**, но **инстанс-метрики** в сводный отчёт **не входят**, пока нет отдельного правила matching.

**Оформление формул:** только обычный текст и символы Unicode (**∩** пересечение, **∪** объединение, **F₁**, **≥**, **×**). Так предпросмотр Markdown в Cursor/VS Code и на GitHub отображает всё без плагинов KaTeX/MathJax.

---

## 1. Откуда берутся величины

| Источник | Что даёт |
|----------|-----------|
| Строка датасета (JSONL) | Golden **`binary_mask`** длины **`node_count`**, **`annotation_kind`**, **`prompt_texts`**, ключ **`(svg_id, annotation_id)`**. |
| Строка **`predictions.ndjson`** | Копия провенанса строки + **`pred_mask`** (после prune для **конкретного** бокса), **`pred_mask_combined`** (один и тот же combined-prune для всех записей одного вызова DINO×сегментатор на строку), **`dino_box_index`**, **`n_boxes`**, **`zero_boxes`**, **`prompt_index`** / **`prompt_text`**, **`boxes`**, пары **`detector_id`** / **`segmenter_id`**, а также **`raster_width`** / **`raster_height`**, **`raster_max_side`** и пороги **`box_threshold`**, **`text_threshold`**, **`score_threshold`** (дублируют `run_config.json`, но привязаны к строке). Файл **дописывается**; при повторном прогоне запись с тем же ключом **`(annotation_id, prompt_index, detector_id, segmenter_id, dino_box_index)`** не дублируется (**resume**). При **`n_boxes == 0`** одна строка с **`dino_box_index: null`**, **`pred_mask`** и **`pred_mask_combined`** — нулевые векторы. |
| **`previews/<svg_id>/`** | Растр, GT по строке, предикты, для **`one_visible_of_many`** — объединённый GT группы и зеркала рядом с предиктом. |

Порог prune для golden при построении вырезанного GT из бинарной маски в коде фиксирован **0.5** (маска дискретная). Порог prune для предикта — **`score_threshold`** из прогона (по умолчанию **0.8**), см. `run_config.json`.

---

## 2. Три представления одной «маски»

Их путают чаще всего; для метрик важно не смешивать.

### 2.1 Вектор по узлам SVG (длина **N**)

- Индексация согласована с парсером и **`data-local-id`** / `seg_<i>` (см. §8 базового документа).
- **Golden:** `binary_mask` из датасета (копия в NDJSON).
- **Pred:** `pred_mask` (per-box) или **`pred_mask_combined`** (один вектор на весь вызов при **K ≥ 1** боксов; при **`n_boxes == 0`** — нулевой вектор).

**Метрики здесь:** точное совпадение векторов (exact match), **F₁** по позициям с ограничением только на drawable-листья при желании.

**Leaf IoU:** по индексам, где в GT стоят единицы, строим множество **G**; где в Pred — **P**:

- **G** = { i | GT[i] = 1 }
- **P** = { i | Pred[i] = 1 }

```
IoU_leaf = |G ∩ P| / |G ∪ P|
```

(если оба множества пусты — заранее договориться: считать IoU = 1 или пропускать строку).

### 2.2 Булева маска на **рабочем растре** (размер **W × H**, как `raster.png`)

Строится объединением (**OR**) масок листьев из **`LeafMaskCache`**, отобранных единицами вектора §2.1 (функция уровня реализации: `leaf_union_from_seg_indices`).

Файлы превью:

- GT по **одной строке:** визуально — `ann_<id>/gt_mask.png`, оверлей — `gt_overlay.png`.
- Pred в том же мире: **`.../<segmenter_id>/pred_leaf_mask.png`** (серое PNG, то же, что подложено под `pred_leaves_overlay.png`).

**Роль в протоколе:** удобные **оверлеи и ручная диагностика** («что попало на растр», конфликт листьев, шум сегментатора в пикселях запроса). В **сводном отчёте по этому документу не задаём обязательных численных метрик** по этим PNG (ни обязательного пиксельного IoU, ни Dice как второго «главного» столбца рядом с **`IoU_leaf`** из §2.1): рендер §2.2 **другой**, чем Cairo-вырезка §2.3, и легко расходится с векторным счётом без ошибки пайплайна.

*(Если когда-то понадобится экспериментальный пиксельный счёт на **`LeafMaskCache`**, это отдельное приложение к отчёту с явным порогом бинаризации и пометкой «не канон относительно §2.1».)*

### 2.3 Вырезанный подграф SVG → Cairo PNG

Pipeline: вектор скорингов → **`prune_from_scores`** → поддерево → **`write_tree`** → **`rasterize_svg_path_max_side`** с тем же **`raster_max_side`**, что у прогона.

- GT по строке: `ann_<id>/gt_extracted.svg`, `gt_extracted.png`.
- Pred: `.../extracted_union.svg`, **`extracted_union.png`** (combined-ветка).

Это **не тот же рендер**, что §2.2 (Cairo vs OR растров листьев). **Совпадение набора листьев с разметкой** задаётся **один раз** — в §2.1: это буквально те же **`binary_mask`** и **`pred_mask_combined`** из NDJSON (побитово по **`node_count`** позициям). Отдельной «проверки листьев числом» через растровые маски §2.2 мы **не предлагаем** — там только оверлей для глаз. Число по выбранным листьям — **§2.1**. Число по **как выглядит вырезка после Cairo** — отдельный блок ниже (**MSE / MAE / SSIM / PSNR**).

#### Зачем вообще MSE / SSIM / PSNR, если уже есть **`IoU_leaf`** по булевым маскам?

Это **разные вопросы**:

- **§2.1** отвечает на: *какие индексы листьев помечены как «внутри объекта» в GT и что предикт поставил в тех же позициях?* Это **дискретная** ошибка по разметке. Два разных рендера могут иметь **одинаковые** маски по узлам и **разный** вид (толщина, цвет обводки, сглаживание) — тогда **`IoU_leaf` = 1**, а MSE ≠ 0.
- **MSE / MAE / SSIM / PSNR** на паре **extracted** PNG отвечают на: *насколько **пиксельно похожи** два **изображения** после одного и того же пайплайна prune → Cairo при одном **`raster_max_side`**?* Это про **непрерывную** ошибку по яркости/структуре картинки.

**Что нового дают пиксельные метрики по сравнению с булевыми:**

1. **Регрессии рендера и визуальной подачи** — смена версии Cairo, масштаба, стиля SVG, антиалиасинга: **`IoU_leaf`** может не двигаться, **PSNR/SSIM** — двигаться.
2. **Тонкие геометрические отличия подграфа** при том же множестве листьев (редко, но возможно при разном порядке/группировке в SVG) — глаз и MSE это заметят раньше, чем битовое равенство масок.
3. **Связка с человеческим «выглядит так же»** — SSIM ближе к этому, чем сухой IoU по узлам.

**Чего они не дают:**

- Они **не заменяют** §2.1 как критерий «угадали ли набор листьев по датасету»: можно получить приемлемый SSIM при частично неправильном наборе узлов или наоборот — плохой MSE при **`IoU_leaf` = 1** из‑за чисто визуальных отличий.

**Итог:** булева маска — **семантика разметки**; **MSE / SSIM / PSNR** на extracted — **качество совпадения картинки**. Для отчёта о сегментации по GT первым остаётся §2.1; второй столбец по желанию — пиксельные метрики на **extracted**, а не IoU по маскам на **`raster.png`**.

**Рекомендуемый набор визуальных численных метрик по extracted** (все опциональны; зафиксировать в отчёте цветовое пространство — RGB или grayscale — и область усреднения: весь кадр или только пиксели «не фона» по простому правилу, чтобы не забивать метрику белым полем):

1. **Качественно:** пара PNG без формулы.
2. **MSE** и **MAE:** как в предыдущей редакции — усреднение **(GT − Pred)²** или **|GT − Pred|** по выбранным каналам (**float**, **[0, 1]** после деления на 255).
3. **PSNR:** монотонная функция от MSE; при значениях в **[0, 1]** удобно  
   `PSNR = 10 × log₁₀(1 / MSE)` (при MSE = 0 договориться о большой константе или пропуске); при счёте в шкале **0…255** используют  
   `PSNR = 10 × log₁₀(255² / MSE_255)`.
4. **SSIM:** по grayscale или усреднение по каналам; типично окно **11×11**, константы стабилизации как в оригинальной статье Wang et al.

**Не включаем в рекомендуемый набор:** бинарный пиксельный **IoU/Dice** после порога на PNG — это снова искусственная маска с произвольным τ, почти не добавляет к §2.1 по смыслу «листья» и хуже интерпретируется, чем MSE/SSIM/PSNR для вашей цели «визуальное совпадение рендера».

**Не смешивать числа:** совпадение **`binary_mask`** и **`pred_mask_combined`** **не** эквивалентно совпадению PNG по PSNR/SSIM и наоборот — см. блок выше.

Для **`one_visible_of_many`** визуальная пара та же по смыслу: **`gt_merged_group_extracted.png`** (или **`oom_merged_*/gt_merged_extracted.png`**) ↔ **`extracted_union.png`** якорного прогона — см. §5.3.

---

## 3. Много боксов у DINO: слой сопоставления

На одну строку датасета и один **`(prompt_index, detector_id, segmenter_id)`** в NDJSON может быть **несколько** записей с разным **`dino_box_index`** (или одна запись при **`n_boxes == 0`**).

Чтобы получить **один** предикат на одну golden-маску строки, нужно явно выбрать правило (зафиксировать в отчёте):

| Вариант | Смысл |
|---------|--------|
| **A. Только combined** | Сравнивать **`pred_mask_combined`** с **`binary_mask`** строки (или с объединённым GT для OOM — см. §5.3). Одна запись на ключ можно дублировать из любого `dino_box_index`. |
| **B. Argmax IoU по боксам** | Для каждого ключа посчитать IoU (leaf или пиксель) между **`binary_mask`** и каждым **`pred_mask`**; взять бокс с максимумом. |
| **C. Венгерка / matching** | Если golden сам по себе «мультиобъектный» и есть несколько предиктов — сложнее; для текущего бейзлайна реже нужно. |

Рекомендация по умолчанию для простых отчётов: **вариант A** для сводной картины + **B** как диагностика «лучший бокс».

**Инстансы vs общий предикат.** **Инстанс-замеры (отдельный IoU/F₁ по каждому `pred_mask` и каждому боксу) в текущем протоколе не нужны** — в отчёт по этому документу их не включаем. Если DINO вернул **K > 1** боксов, мы всё равно сохраняем **каждый бокс отдельно**: в NDJSON — по строке на **`dino_box_index`** с полем **`pred_mask`**; в превью — **`extracted_box{i}.*`**, маски **`box{i}_*`**. Это для отладки и для **будущего** протокола сопоставления предикт↔экземпляр (matching / Hungarian и т.п.). **Что считаем сейчас по вектору — только combined:** **`pred_mask_combined`**. Превью **`extracted_union.*`** — каноническая пара к GT для **визуальных** численных метрик (§2.3); **`pred_leaf_mask.png`** — для оверлея и диагностики §2.2, **не** как обязательный столбец сводных метрик. Варианты **B/C** в таблице выше — опциональная диагностика («лучший бокс»), не обязательная часть основного протокола.

---

## 4. Артефакты превью (шпаргалка)

Корень прогона: **`--out`** (например `output/gt_baseline_full/`).

```
previews/<svg_id>/
  source.svg
  raster.png
  oom_merged_<slug>/          # только для групп one_visible_of_many
    meta.json
    gt_merged_mask.png
    gt_merged_overlay.png
    gt_merged_extracted.svg
    gt_merged_extracted.png
  ann_<annotation_id>/
    gt_mask.png
    gt_overlay.png
    gt_extracted.svg
    gt_extracted.png
    p<prompt_index>/<detector_id>/<segmenter_id>/
      mask_union.png              # SAM/box-fill в пикселях запроса
      overlay_union.png
      pred_leaf_mask.png           # pred combined в мире листьев на растре
      pred_leaves_overlay.png
      extracted_union.svg / .png   # pred объект (Cairo)
      extracted_box{i}.*           # per-box предикты (SVG+PNG)
      box{i}_mask.png / box{i}_overlay.png
      dino_boxes.png               # боксы DINO на растре (диагностика)
      gt_merged_group_mask.png      # только для строк kind=one_visible_of_many, зеркало группы
      gt_merged_group_overlay.png
      gt_merged_group_extracted.svg / .png
```

Имена **`gt_merged_group_*`** в папке сегментатора совпадают по содержанию с файлами в **`oom_merged_<slug>/`** для соответствующей группы (**одинаковый `svg_id` и список `prompt_texts`**); они кладутся для удобства сравнения рядом с **`extracted_union.*`**.

Отключить запись merged-превью: **`--no-oom-gt-merged`** в CLI.

---

## 5. По каждому `annotation_kind`

В экспорте сервиса встречаются как минимум три вида (строки с **`review_decision == declined`** в прогон не входят — см. базовый документ).

### 5.1 `exactly_one_visible_object`

- **Смысл строки:** один выделенный объект (или одна согласованная группа листьев) под данным промптом; golden — **`binary_mask`** этой строки.
- **Эталон для замера:** **`binary_mask`** (§2.1); визуальные числа при необходимости — только производные §2.3 (**extracted**), не §2.2.
- **Предикат:** после §3 (обычно **`pred_mask_combined`** или лучший бокс).
- **Визуал:** основная сопоставимая пара для численного визуала — **`gt_extracted.png`** ↔ **`extracted_union.png`** (§2.3). **`gt_mask`** / **`pred_leaf_mask`** — по желанию для оверлея (§2.2), без обязательных сводных метрик по ним.

Особый случай: в строке **несколько** `prompt_texts` — каждый **`prompt_index`** это отдельный запрос DINO и отдельный столбец метрик.

### 5.2 `multiple_visible_objects`

- **Смысл строки:** golden задаёт **набор** листьев (часто несколько визуальных частей одной семантики: звёзды, облака и т.д.) как **одну** маску на строку.
- **Эталон:** снова **`binary_mask`** строки; предикт — тот же протокол §3.
- **Визуал:** как в §5.1 (**extracted** для численного визуала при необходимости).
- **Замечание:** если DINO даёт **несколько боксов**, **`pred_mask_combined`** отражает объединённую ветку прунинга по **`combined_mask`** сегментатора — это естественная пара к «одному golden на строку» при варианте **A** из §3.

### 5.3 `one_visible_of_many`

- **Смысл данных:** на одном **`svg_id`** несколько **строк** с **одинаковым** набором **`prompt_texts`** (например одно слово «window»), каждая строка помечает **один экземпляр** из многих похожих объектов на сцене.
- **Группа для объединённого эталона:** все строки с одинаковым **`(svg_id, tuple(prompt_texts))`** и этим **`annotation_kind`**.
- **Объединённый golden (группа):** покомпонентное **OR** всех **`binary_mask`** в группе → **`merged_binary_mask`** (та же длина **N**).

**Что сравнивать с чем (протокол OOM — один предикат якоря на объединённый GT группы):**

1. Выбрать **якорную строку** (например минимальный **`annotation_id`** в группе или явно заданное правило — зафиксировать в отчёте).
2. Инференс для якорной строки даёт **`pred_mask_combined`** (и превью в `ann_<anchor>/p*/…`).

   **NDJSON:** для **каждой** строки группы инференс обычно тоже выполняется — в файле есть отдельные записи с их **`binary_mask`** и своим **`pred_mask_combined`**. В протоколе **§5.3** метрика **merged ↔ предикт** считается **один раз на группу**: берут только пару **`merged_binary_mask`** + **`pred_mask_combined` якоря**. Предикты **неякорных** строк к этой метрике **не подмешивают** (иначе иной смысл «один предикат на объединённый GT», и при micro-агрегации по строкам датасета группа даст лишние точки).

3. **Численно:** тот же **IoU_leaf**, что в §2.1, между векторами **`merged_binary_mask`** и **`pred_mask_combined`**:

   - **G_M** = индексы единиц в `merged_binary_mask`
   - **G_P** = индексы единиц в `pred_mask_combined`

```
IoU_leaf(merged_binary_mask, pred_mask_combined) = |G_M ∩ G_P| / |G_M ∪ G_P|
```

4. **Визуально «объект» (и единственная рекомендуемая база для визуальных численных метрик):** **`gt_merged_group_extracted.png`** (или центральная копия в `oom_merged_<slug>/`) ↔ **`extracted_union.png`** того же якорного прогона и той же тройки **`(prompt_index, detector_id, segmenter_id)`** — см. §2.3.
5. **`gt_merged_group_mask.png`** ↔ **`pred_leaf_mask.png`** — оверлей / диагностика по листьям на растре (§2.2); **в сводные визуальные метрики протокола не включаем**.

На якорном прогоне DINO может снова выдать **несколько боксов** — они сохранены как инстансы (**§3**, **`extracted_box{i}.*`**, строки NDJSON с разными **`dino_box_index`**). **Числовые метрики в §5.3 по умолчанию считаются не по каждому боксу**, а по **`pred_mask_combined`** (один вектор на весь вызов), т.е. «на общем» объединении ветки сегментатора; это сознательно, пока **нет согласованного протокола метрик per-instance**.

**Связь с §8 (скрипт оценки).** Строки датасета с **`annotation_kind`**: **`one_visible_of_many`** **не** эмитятся как **`canonical_row`**: для каждой обнаруженной на диске группы (`oom_merged_*/meta.json`) считается **одна** строка отчёта с **`row_kind`**: **`oom_merged:<slug>`**, **`annotation_kind`**: **`one_visible_of_many_merged`**, GT = OR масок членов группы, Pred = **`pred_mask_combined`** якоря (**минимальный `annotation_id`** в группе). Неякорные строки группы в эту точку **не входят** (нет двойного micro-счёта по строкам JSONL).

Так отделяется GT одной строки (`gt_extracted` в `ann_<id>`) от **группового** эталона OOM (`gt_merged_*`). Отдельный учёт боксов к экземпляру (matching per-instance) пока не задаём — **§3**.

---

## 6. Стратификация отчёта

Сводные цифры (micro/macro по IoU, exact match, доля **`zero_boxes`**) разумно считать **отдельно** по **`annotation_kind`** и отдельно по **`(detector_id, segmenter_id)`**, чтобы не смешивать режимы с разным смыслом golden и разным числом строк на SVG.

В **`scripts/eval_gt_baseline_metrics.py`** последняя строка потока NDJSON — JSON с **`row_kind`**: **`_summary`**. В ней три макро-блока по **одним и тем же** числовым полям построчных записей (средние по выбранным метрикам + **`exact_match_rate`**):

- **`macro_canonical_row`** — только строки **`row_kind`**: **`canonical_row`** (дедуп по ключу **`(svg_id, annotation_id, prompt_index, detector_id, segmenter_id)`**, первая запись по рангу бокса; **`one_visible_of_many`** в канонике **не считается**).
- **`macro_oom_merged`** — только строки, у которых **`row_kind`** начинается с **`oom_merged:`** (одна строка на группу × detector × segmenter при наличии предиктов у всех членов группы для этой пары).
- **`macro_all`** — объединение обоих множеств (удобно как «весь отчёт одним числом», но смешивает разную семантику строки датасета vs группы OOM).

Дополнительно **`macro_by_detector_segmenter`** — те же агрегаты внутри каждой пары **`detector_id|segmenter_id`**.

---

## 7. Связь с LLM-прогоном

У OpenRouter одна модель в одном ответе часто выдаёт маски сразу на все **`annotation_id`** группы. У бейзлайна вызовы **изолированы по строке** — см. **`docs/llm_inference_issues_and_recommendations.md`**. Сравнение отчётов LLM и DINO требует одинакового **описанного здесь** протокола для DINO-стороны (особенно §3 и §5.3).

---

## 8. Реализация (зафиксированный способ замера)

### 8.1 Входы и подготовка OOM

- **Каталог прогона** **`--out`**: обязательно **`predictions.ndjson`** и **`previews/`** (как после инференса из **`docs/baseline_gt_dataset_inference.md`**).
- **Группы `one_visible_of_many`:** оценщик **не** читает заново весь JSONL для списка групп; он **обходит** только существующие **`previews/<svg_id>/oom_merged_*/meta.json`** с **`kind`**: **`one_visible_of_many_gt_merged`**. Если таких каталогов нет, **`oom_groups=0`** и OOM-метрик не будет — до полного eval один раз выполнить **`scripts/write_all_oom_gt_merged.py`** (например **`--out <каталог>`** + **`--from-run-config`**, см. справку скрипта). Запись merged-превью при инференсе можно отключить флагом **`--no-oom-gt-merged`**, но **оценка OOM всё равно опирается** на эти **`meta.json`** после отдельного прогона **`write_all_oom_gt_merged`**.

### 8.2 `scripts/eval_gt_baseline_metrics.py`

**Каноническая фаза (если не передан `--oom-metrics-only`):** из **`predictions.ndjson`** берётся по одной записи на ключ **`(svg_id, annotation_id, prompt_index, detector_id, segmenter_id)`** (дедупликация **`dedupe_canonical_combined`**). Для каждой такой записи, у которой **`annotation_kind` ≠ `one_visible_of_many`**, считаются **`binary_mask`** ↔ **`pred_mask_combined`** и визуальные метрики по паре **`gt_extracted.png`** ↔ **`extracted_union.png`**. В выход попадает **`row_kind`**: **`canonical_row`**.

**Фаза OOM:** для каждой группы из **`discover_oom_groups(previews)`** и каждой пары **`(detector_id, segmenter_id)`**, для которой в NDJSON есть предикт якоря и **всех** членов группы, считается OR их **`binary_mask`**, сравнение с **`pred_mask_combined`** якоря, визуал по **`gt_merged_group_extracted`** / центральной копии (**`gt_is_merged_group=True`** в коде). **`row_kind`** в выводе: префикс **`oom_merged:`** + **полное имя каталога** группы под **`previews/<svg_id>/`** (то есть строка вида **`oom_merged:oom_merged_<hex>`** — дублирующийся префикс в имени, это следствие **`slug = dirname`** в коде).

**Флаги (кратко):** **`--svg-ids`**, **`--smoke-svg-n`** (первые N уникальных **`svg_id`** по порядку строк в **`predictions.ndjson`**; несовместимо с **`--svg-ids`**), **`--limit`** (усечь число **канонических** строк после сортировки; **несовместимо** с **`--oom-metrics-only`**), **`--oom-metrics-only`** (только OOM-строки + сводка), **`--csv`**, **`--summary-json`**, **`--progress-every N`** (прогресс и ETA в **stderr** каждые N шагов; **`0`** — выкл.), **`--roi-gray-eps`**. Флаг **`--oom`** оставлен для совместимости и **не меняет** поведение: OOM считается **всегда**, если на диске есть соответствующие **`meta.json`**.

**Выход:** в **stdout** сначала NDJSON по одной метрике на строку, **последняя** строка — сводка **`_summary`** (см. §6). Строка **`_summary`** в **`--csv` не попадает**. Логи удобно писать в отдельный файл (**stderr**), например **`2> metrics_run.log`**, и при длинных прогонах выставить **`PYTHONUNBUFFERED=1`**, чтобы прогресс не буферизовался.

**Пример полного прогона** (корень репозитория; пути заменить на свой каталог):

```bash
PYTHONUNBUFFERED=1 python scripts/eval_gt_baseline_metrics.py \
  --out output/gt_baseline_full \
  --progress-every 100 \
  --csv output/gt_baseline_full/metrics_per_row_full.csv \
  --summary-json output/gt_baseline_full/metrics_full.summary.json \
  > output/gt_baseline_full/metrics_per_row_full.ndjson \
  2> output/gt_baseline_full/metrics_full_run.log
```

**Только OOM** (после того же **`write_all_oom_gt_merged`**):

```bash
python scripts/eval_gt_baseline_metrics.py --out output/gt_baseline_full --oom-metrics-only \
  --summary-json output/gt_baseline_full/metrics_oom_only.summary.json \
  > output/gt_baseline_full/metrics_oom_only.ndjson
```

Логика построения merged GT и путей превью совпадает с модулем **`svg_gt_baseline.oom_gt_merge`** (ключ группы **`(svg_id, tuple(prompt_texts))`**).

---

## 9. Приложение: примеры счёта на коротких векторах (листья)

Везде длина вектора **N** совпадает с **`node_count`** строки; позиции — те же, что в **`binary_mask`** / **`pred_mask`** / **`pred_mask_combined`**.

**Обозначения**

- **G** = множество индексов, где GT = 1; **P** = где Pred = 1.
- **IoU_leaf** = |G ∩ P| / |G ∪ P| (см. §2.1 про пустые множества).
- **F₁** — по всем **N** позициям, как бинарная классификация «предсказана ли единица в GT» (TP / FP / FN по позициям).
- **Exact** — побитовое равенство двух векторов.

**Pred** ниже — уже выбранный вектор после правил §3 (чаще всего **`pred_mask_combined`** одной записи NDJSON для нужной тройки detector/segmenter/prompt).

### 9.1 `exactly_one_visible_object`

**Смысл:** одна строка датасета — один эталон **`binary_mask`** на один промпт (или несколько промптов подряд — тогда считаем **отдельный счёт на каждый `prompt_index`**).

**Пример А** (N = 6), один промпт:

| | индекс | 0 | 1 | 2 | 3 | 4 | 5 |
|---|-----|---|---|---|---|---|---|
| GT | | 0 | 0 | 1 | 1 | 0 | 0 |
| Pred | | 0 | 0 | 1 | 1 | 0 | 0 |

G = P = {2, 3} → **IoU_leaf = 1**, **F₁ = 1**, **Exact = да**.

**Пример Б** (частичное попадание):

| GT | 0 | 1 | 1 | 1 | 0 | 0 |
| Pred | 0 | 1 | 0 | 1 | 1 | 0 |

G = {1, 2, 3}, P = {1, 3, 4} → |G ∩ P| = 2, |G ∪ P| = 4 → **IoU_leaf = 0.5**; TP=2, FP=1, FN=1 → **F₁ ≈ 0.667**; Exact = нет.

**Пример В** (два промпта в одной строке): для **`prompt_index=0`** и **`prompt_index=1`** берутся **разные** вызовы DINO → **два разных Pred** из NDJSON; golden для каждого индекса промпта — тот же **`binary_mask`** строки **или** если в продукте маска завязана на задачу по-другому, следовать экспорту; в текущем JSONL одна строка — одна **`binary_mask`** на всю строку, несколько **`prompt_texts`** задают несколько запросов к одному и тому же golden-вектору (метрики: два столбца IoU и F₁ при тех же GT и разных Pred).

### 9.2 `multiple_visible_objects`

**Смысл:** по-прежнему **одна** golden-маска **`binary_mask`** на строку, но семантика «несколько видимых частей одной категории» (несколько компонент единиц в одном векторе). **Формулы те же**, что в §9.1: сравнение вектора длины **N** с Pred.

**Пример** (N = 8): golden объединяет два куска (например несколько «звёзд» одним классом):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| GT | 0 | **1** | **1** | 0 | 0 | **1** | **1** | 0 |

G = {1, 2, 5, 6}.

**Pred** «поймал только левый кусок»:

| Pred | 0 | **1** | **1** | 0 | 0 | 0 | 0 | 0 |

P = {1, 2} → |G ∩ P| = 2, |G ∪ P| = 4 → **IoU_leaf = 0.5**; TP=2, FP=0, FN=2 → precision=1, recall=0.5 → **F₁ ≈ 0.667**.

Если Pred заполнит все четыре позиции правильно и не добавит лишних — снова IoU = F₁ = 1. Отличие от `exactly_one` — не в формуле, а в **интерпретации отчёта** (отдельная страта по **`annotation_kind`**).

### 9.3 `one_visible_of_many`

Протокол тот же, что **§5.3**: объединённый GT группы ↔ **`pred_mask_combined`** прогона **якорной** строки. В автоматическом отчёте **`eval_gt_baseline_metrics.py`** эта ветка попадает **только** в строки **`oom_merged:*`**, не в **`canonical_row`** (см. §5.3 конец и §8).

На якорном прогоне DINO может вернуть **K ≥ 1** боксов — в артефактах они сохранены как инстансы (**§3**); в числовом примере ниже **Pred** уже **`pred_mask_combined`** (агрегат по боксам), не отдельный бокс.

На одном **`svg_id`** несколько **строк** с одинаковым **`prompt_texts`**; у каждой своё **`binary_mask`** (обычно почти непересекающиеся единицы — разные экземпляры).

**Шаг 1 — объединённый эталон** (покомпонентное OR по строкам группы с тем же ключом `(svg_id, tuple(prompt_texts))`):

Строка ann\_10: `[1, 1, 0, 0, 0, 0]`  
Строка ann\_11: `[0, 0, 1, 1, 0, 0]`  
Строка ann\_12: `[0, 0, 0, 0, 1, 1]`

**Merged M** (покомпонентное ИЛИ):

| M | **1** | **1** | **1** | **1** | **1** | **1** |

**Шаг 2 — предикт:** один прогон по **якорной** строке (например ann\_10). У реального DINO там могло быть несколько боксов — в таблице ниже **Pred** уже агрегирован как **`pred_mask_combined`** (не отдельный бокс):

| P_anchor (combined) | **1** | **1** | **1** | **1** | 0 | 0 |

G_M = {0, …, 5}, P = {0, 1, 2, 3} → |G_M ∩ P| = 4, |G_M ∪ P| = 6 → **IoU_leaf(M, P) = 4/6 ≈ 0.667**; TP=4, FP=0, FN=2 → precision=1, recall = 4/6 → **F₁ = 0.8**.

**Exact** между M и P здесь обычно **нет** (разная поддержка единиц).

---

## 10. Конкретный пример: что замерить на smoke-прогоне

**Smoke** — после фильтра **`review_decision != declined`** берутся только **первые N уникальных `svg_id`** в порядке следования строк в JSONL (см. **`docs/baseline_gt_dataset_inference.md`**). Пример ниже опирается на уже собранный каталог **`output/gt_baseline/`** в корне репозитория (запуск с **`--smoke --smoke-n 3`** и тем же датасетом: в конфиге указано **`smoke_n`: 3**).

Корень прогона (чтобы подставить свои пути):

`/home/izhanvarsky/svg-segmentator/output/gt_baseline/`

### 10.1 Одна строка `exactly_one_visible_object` (svg_id 5, ann 6, промпт «sun»)

**Контекст.** В **`predictions.ndjson`** первая запись этого прогона — строка датасета **`annotation_id`: 6**, **`svg_id`: 5**, **`annotation_kind`**: `exactly_one_visible_object`, файл **`little-city-2.svg`**, **`node_count`**: 252, один промпт (**`prompt_index`**: 0, **`prompt_text`**: `"sun"`).

**Что сравниваем численно (протокол §3, вариант A).**

1. Найти в **`predictions.ndjson`** объект с **`annotation_id`**: 6, **`prompt_index`**: 0, нужными **`detector_id`** / **`segmenter_id`** (ниже — конкретная папка превью).
2. Взять **`binary_mask`** из этой же строки NDJSON — это **GT** длины **N = 252**.
3. Взять **`pred_mask_combined`** из **любой** записи этого вызова с тем же **`dino_box_index`** или при **`n_boxes`**: 1 — там одна строка с **`dino_box_index`**: 0; **`pred_mask`** и **`pred_mask_combined`** совпадают.
4. Посчитать **IoU_leaf** по §2.1: **G** — индексы единиц **`binary_mask`**, **P** — единиц **`pred_mask_combined`**. **F₁** и **Exact** — как в §9 (по всем **N** позициям).

**Замеры по файлу** `/home/izhanvarsky/svg-segmentator/output/gt_baseline/predictions.ndjson` (запись **`annotation_id`**: 6, **`prompt_index`**: 0, **`gdino-tiny`**, **`box-fill`**, **`dino_box_index`**: 0):

| Величина | Значение |
|----------|----------|
| **N** (`node_count`) | 252 |
| **G** | ровно **2** индекса (две единицы в GT) |
| **P** | **2** индекса, те же что **G** |
| **IoU_leaf** (§2.1) | **1** (= 2/2, совпали множества индексов) |
| **Exact** (побитово `binary_mask` == `pred_mask_combined`) | **да** |
| **F₁** по позициям (TP=2, FP=0, FN=0) | **1** |

Инстансы по боксам для этой строки при **`n_boxes`**: 1 не дают отдельной метрики в основном протоколе — см. **Scope** и §3.

**Где лежат картинки для ручной проверки** (та же строка, **`gdino-tiny`**, **`box-fill`**, **`prompt_index`**: 0):

- Подложка и GT строки:  
  `/home/izhanvarsky/svg-segmentator/output/gt_baseline/previews/5/raster.png`  
  `/home/izhanvarsky/svg-segmentator/output/gt_baseline/previews/5/ann_6/gt_mask.png`  
  `/home/izhanvarsky/svg-segmentator/output/gt_baseline/previews/5/ann_6/gt_overlay.png`
- Вырезанный объект (Cairo):  
  `/home/izhanvarsky/svg-segmentator/output/gt_baseline/previews/5/ann_6/gt_extracted.png`  
  ↔ предикт combined:  
  `/home/izhanvarsky/svg-segmentator/output/gt_baseline/previews/5/ann_6/p0/gdino-tiny/box-fill/extracted_union.png`
- Диагностика детектора / сегментатора в пикселях запроса: **`dino_boxes.png`**, **`mask_union.png`**, **`overlay_union.png`** в той же папке **`.../p0/gdino-tiny/box-fill/`**. Численный визуал по протоколу — пара **`gt_extracted.png`** ↔ **`extracted_union.png`** (§2.3). **`pred_leaf_mask.png`** и **`gt_mask`** — только оверлей §2.2; **`extracted_box0.*`** / **`box0_*`** — отладка бокса.

### 10.2 Несколько боксов на той же строке (тот же ann 6, другой детектор)

Для **`gdino-base`** + **`box-fill`** на этой же строке в NDJSON **`n_boxes`**: 2 — **две** записи с **`dino_box_index`**: 0 и 1, разные **`pred_mask`**, но **одинаковый** **`pred_mask_combined`** на обеих.

**Замер по протоколу:** одна пара векторов **`binary_mask`** ↔ **`pred_mask_combined`** (достаточно **любой** из двух записей NDJSON с тем же **`annotation_id`** / **`prompt_index`** / detector / segmenter — **`pred_mask_combined`** совпадает). Отдельно IoU по **`pred_mask`** каждого бокса **не считаем** в сводном отчёте.

**Замеры по тому же** `predictions.ndjson` (**`gdino-base`**, **`box-fill`**, любая из записей с **`dino_box_index`**: 0 или 1):

| Величина | Значение |
|----------|----------|
| **G** | **2** индекса |
| **P** | **3** индекса (combined подхватил лишний лист от второго бокса) |
| **IoU_leaf** (§2.1) | **2/3 ≈ 0.667** |
| **Exact** | **нет** |
| **F₁** по позициям (TP=2, FP=1, FN=0) | **0.8** |

Для визуальных метрик по §2.3 — пара **`gt_extracted.png`** ↔ **`extracted_union.png`** (рядом в дереве превью). **`pred_leaf_mask.png`** — при необходимости оверлея §2.2. Папка:  
`/home/izhanvarsky/svg-segmentator/output/gt_baseline/previews/5/ann_6/p0/gdino-base/box-fill/`  
(плюс **`extracted_box0.*`**, **`extracted_box1.*`** для двух боксов).

### 10.3 Про `one_visible_of_many` на smoke

В состав **первых N уникальных `svg_id`** (например **`--smoke-svg-n 4`** в **`scripts/eval_gt_baseline_metrics.py`**) может **не попасть** ни одной группы **`one_visible_of_many`** — тогда в превью не будет задействованных кейсов этого вида. Ниже §10.4 — разбор по **`svg_id` 68**; точечные метрики для этого SVG: построчные **`canonical_row`** только для **других** `annotation_kind` на том же SVG; для **`one_visible_of_many`** — только строки **`oom_merged:*`** (см. ниже), плюс сводка **`_summary`** в конце NDJSON.

- **`output/gt_baseline_full/metrics_per_row_svg68_one_of_many.ndjson`**
- **`output/gt_baseline_full/metrics_per_row_svg68_one_of_many.csv`**
- **`output/gt_baseline_full/metrics_svg68_one_of_many.summary.json`**

Команда (корень репозитория, **`--out`** = каталог прогона с **`predictions.ndjson`** и **`previews/`**):

`python scripts/eval_gt_baseline_metrics.py --out output/gt_baseline_full --svg-ids 68 --csv output/gt_baseline_full/metrics_per_row_svg68_one_of_many.csv --summary-json output/gt_baseline_full/metrics_svg68_one_of_many.summary.json > output/gt_baseline_full/metrics_per_row_svg68_one_of_many.ndjson`

Перед запуском eval для полного дерева один раз: **`python scripts/write_all_oom_gt_merged.py --out … --from-run-config`** (или явные **`--dataset`** / **`--manifest`**). В **`scripts/eval_gt_baseline_metrics.py`** строки датасета с **`annotation_kind`**: **`one_visible_of_many`** **никогда** не попадают в **`canonical_row`**: для этого вида GT в отчёте — **merged** (OR по членам группы), метрики только в **`row_kind`**: **`oom_merged:…`** (по одному на группу × det × seg при полном наборе предиктов по членам группы), для чего на диске должны быть **`previews/<svg_id>/oom_merged_*/meta.json`**.

### 10.4 Полный прогон: `one_visible_of_many` (svg_id 68, промпт «window»)

Корень прогона:

`/home/izhanvarsky/svg-segmentator/output/gt_baseline_full/`

**Группа.** Один **`svg_id`**: **68**, файл **`1554025379.svg`**, **`prompt_texts`**: `["window"]`, **`annotation_kind`**: `one_visible_of_many`. Состав группы (эталон объединения) записан в:

`/home/izhanvarsky/svg-segmentator/output/gt_baseline_full/previews/68/oom_merged_44ff7b02c80d/meta.json`

Там **`annotation_ids`**: **216, 219, 221, 222, 224, 226, 227** (**7** строк), **`gt_leaf_nodes_ones_merged`**: **49** — число единиц в **`merged_binary_mask`** (покомпонентное OR семи **`binary_mask`** из датасета).

**Якорь** (правило из §5.3 — минимальный **`annotation_id`** в группе): **216**.

**Что сравниваем численно (§5.3).**

1. Построить **`merged_binary_mask`** длины **N = 142**: OR масок **`binary_mask`** по всем семи **`annotation_id`** (маски покомпонентно по индексам листьев). У проверенного артефакта сумма единиц **49** (как в **`meta.json`**).
2. Взять **`pred_mask_combined`** из **одной** строки **`predictions.ndjson`** для якоря: **`annotation_id`**: **216**, **`prompt_index`**: **0**, **`detector_id`**: **`gdino-tiny`**, **`segmenter_id`**: **`box-fill`** — любая запись с этим ключом, например **`dino_box_index`**: **0** (на якорном прогоне **`n_boxes`**: **8**; для замера нужен только combined, одинаковый на всех восьми строках NDJSON этого вызова).
3. Считать **IoU_leaf** между множествами индексов единиц **M** и **P** как в §2.1 / §5.3. Остальные шесть строк группы в NDJSON **не участвуют** в этой точке метрики (см. §5.3 про двойной счёт).

**Замеры по файлу** `/home/izhanvarsky/svg-segmentator/output/gt_baseline_full/predictions.ndjson` (якорь **216**, **`gdino-tiny`**, **`box-fill`**, **`pred_mask_combined`**):

| Величина | Значение |
|----------|----------|
| **N** (`node_count`) | **142** |
| Число единиц в **merged_binary_mask** (**M**) | **49** |
| Число единиц в **`pred_mask_combined`** (**P**) | **66** |
| Пересечение **M ∩ P** (оба вектора имеют 1) | **49** (по объединённому GT лишних пропусков нет: FN = 0) |
| **IoU_leaf**(**M**, **P**) | **49 / 66 ≈ 0.742** |
| **Exact** (**M** vs **P**) | **нет** |
| **F₁** по позициям (TP=49, FP=17, FN=0) | **≈ 0.852** |

Инстансы (**8** боксов → **`extracted_box0..7`**, разные **`pred_mask`** в NDJSON) в сводную метрику по §5.3 **не входят**.

**Превью для глаз и опциональных визуальных метрик §2.3.**

- Объединённый GT группы (центральная папка), пара для extracted:  
  `/home/izhanvarsky/svg-segmentator/output/gt_baseline_full/previews/68/oom_merged_44ff7b02c80d/gt_merged_extracted.png`  
  ↔ **`extracted_union.png`** якоря (путь ниже). Маска на растре **`gt_merged_mask.png`** — только диагностика §2.2.
- Рядом с предиктом якоря (зеркало extracted/mask):  
  `/home/izhanvarsky/svg-segmentator/output/gt_baseline_full/previews/68/ann_216/p0/gdino-tiny/box-fill/gt_merged_group_extracted.png`  
  ↔ `/home/izhanvarsky/svg-segmentator/output/gt_baseline_full/previews/68/ann_216/p0/gdino-tiny/box-fill/extracted_union.png`  
  Оверлей по листьям при желании: **`gt_merged_group_mask.png`**, **`pred_leaf_mask.png`** в той же папке (§2.2, без обязательных сводных метрик).

Цифры в таблице соответствуют текущему сохранённому **`predictions.ndjson`** / **`meta.json`** в репозитории; после перезапуска инференса их нужно пересчитать тем же способом.
