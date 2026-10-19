# Архив: GT-бейзлайн + документация + README

Снимок для передачи без полного репозитория (без `output/`, `.venv/`, сервисов Docker и тяжёлых данных).

## Состав

| Путь | Назначение |
|------|------------|
| `README.md` | Обзор пайплайна `svg_segmentator` (DINO, SAM, prune) |
| `requirements.txt` | Зависимости Python |
| `svg_gt_baseline/` | Пакет батч-инференса по GT JSONL |
| `svg_segmentator/` | Ядро растра, детекторы, сегментаторы, leaf prune |
| `scripts/run_gt_baseline_inference.py` | CLI запуска бейзлайна |
| `docs/` | Спецификации: `baseline_gt_dataset_inference.md`, `gt_baseline_metrics.md`, `llm_inference_issues_and_recommendations.md` |
| `reference/openrouter_structured_run_README.md` | Копия README из отчёта прогона LLM (OpenRouter structured masks) |

## Быстрый старт (GT-бейзлайн)

```bash
cd svg_segmentator_gt_baseline_bundle   # корень после распаковки архива
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# опционально: .env с HF_TOKEN для Hugging Face

.venv/bin/python scripts/run_gt_baseline_inference.py \
  --dataset "dataset.jsonl" \
  --manifest path/to/svg_archive_manifest.json \
  --smoke --smoke-n 3
```

По умолчанию артефакты в `output/gt_baseline/` относительно корня **репозитория-источника**; при распаковке только бандла задайте явный `--out ./my_run`.

## Документы по смыслу

- **`docs/baseline_gt_dataset_inference.md`** — контракт NDJSON, кэши, превью, связь с LLM (§4).
- **`docs/gt_baseline_metrics.md`** — что с чем мерить по `annotation_kind`, matching при нескольких боксах, OOM и файлы превью.
- **`docs/llm_inference_issues_and_recommendations.md`** — проблемы LLM-инференса и рекомендации по протоколу оценки.

В каталоге `output/` лежат оба варианта упаковки одного содержимого: **`svg_segmentator_gt_baseline_docs_bundle_YYYYMMDD.tar.gz`** и **`svg_segmentator_gt_baseline_docs_bundle_YYYYMMDD.zip`**. Дата — в имени файла.
