"""NDJSON helpers for prediction rows: LLM / v2 bridge (minimal rows) + shared dedupe keys.

Существующий ``run_inference_job`` по-прежнему пишет полные строки с ``dino_box_index`` —
ничего там не ломаем. Мост v2→eval может писать **укороченные** строки через
``build_llm_prediction_record``; ``eval_metrics`` понимает оба варианта через ``row_box_index``.
"""

from __future__ import annotations

from typing import Any

from svg_gt_baseline.types import DatasetAnnotationRow


def row_box_index(d: dict[str, Any]) -> int | None:
    """Индекс инстанса для дедупа: сначала нейтральный ``box_index``, иначе ``dino_box_index``."""
    if "box_index" in d:
        v = d["box_index"]
        return None if v is None else int(v)
    if "dino_box_index" in d:
        v = d["dino_box_index"]
        return None if v is None else int(v)
    return None


def build_llm_prediction_record(
    *,
    row: DatasetAnnotationRow,
    prompt_index: int,
    prompt_text: str,
    detector_id: str,
    segmenter_id: str,
    pred_mask: list[int],
) -> dict[str, Any]:
    """Минимальная строка ``predictions.ndjson`` для LLM: только то, что читает ``eval_gt_baseline_metrics``."""
    pm = list(pred_mask)
    return {
        "svg_id": int(row.svg_id),
        "annotation_id": int(row.annotation_id),
        "annotation_kind": str(row.annotation_kind),
        "prompt_index": int(prompt_index),
        "prompt_text": str(prompt_text),
        "detector_id": str(detector_id),
        "segmenter_id": str(segmenter_id),
        "binary_mask": list(row.binary_mask),
        "pred_mask": pm,
        "pred_mask_combined": pm,
    }
