from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class InferenceSettings:
    """Hyperparameters aligned with docs/baseline_gt_dataset_inference.md."""

    raster_max_side: int = 640
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    score_threshold: float = 0.8
    max_leaves: int | None = None
    seed: int = 42
    leaf_render_dpi: float = 96.0  # unused when output_wh is set for leaf cache


@dataclass
class DatasetAnnotationRow:
    """One line from service export JSONL."""

    svg_id: int
    annotation_id: int
    annotation_kind: str
    object_key: str
    original_filename: str
    node_count: int
    binary_mask: list[int]
    prompt_texts: list[str]
    review_decision: str | None
    raw: dict = field(repr=False)
