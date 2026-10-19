"""GT dataset baseline inference (DINO + segmenters) with disk cache and NDJSON output."""

from __future__ import annotations

__all__ = ["run_inference_job"]


def __getattr__(name: str):
    if name == "run_inference_job":
        from svg_gt_baseline.engine import run_inference_job

        return run_inference_job
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
