"""One detector+segmenter+CLIP bundle for repeated pipeline runs without reloading weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from svg_segmentator.backends.interfaces import DetectionBackend, SegmentationBackend
from svg_segmentator.backends.registry import build_detector, build_segmenter
from svg_segmentator.clip_fitness import load_clip
from svg_segmentator.pipeline import PipelineConfig, PipelineRunSummary, run_pipeline_combo


@dataclass
class EvalPipeline:
    """
    Holds detector, segmenter, and CLIP in memory for many ``run`` calls.

    Create one instance per (detector, segmenter) pair; use :meth:`from_registry`
    for the usual id-based construction.
    """

    detector: DetectionBackend
    segmenter: SegmentationBackend
    device: torch.device
    clip_model_id: str
    _clip: tuple[CLIPProcessor, CLIPModel, torch.device]

    @classmethod
    def from_registry(
        cls,
        *,
        detector_id: str,
        segmenter_id: str,
        device: torch.device,
        clip_model_id: str = "openai/clip-vit-base-patch32",
        gdino_model_override: str | None = None,
        sam_model_override: str | None = None,
    ) -> EvalPipeline:
        detector = build_detector(detector_id, override_model_id=gdino_model_override)
        segmenter = build_segmenter(segmenter_id, override_model_id=sam_model_override)
        clip_bundle = load_clip(clip_model_id, device=device)
        return cls(
            detector=detector,
            segmenter=segmenter,
            device=device,
            clip_model_id=clip_model_id,
            _clip=clip_bundle,
        )

    def run(
        self,
        *,
        svg_path: Path,
        out_dir: Path,
        config: PipelineConfig,
        candidate_render_dir: Path | None = None,
    ) -> PipelineRunSummary:
        if config.clip_model_id != self.clip_model_id:
            raise ValueError(
                "PipelineConfig.clip_model_id must match EvalPipeline.clip_model_id "
                f"({self.clip_model_id!r}), got {config.clip_model_id!r}"
            )
        return run_pipeline_combo(
            svg_path=svg_path,
            out_dir=out_dir,
            detector=self.detector,
            segmenter=self.segmenter,
            config=config,
            device=self.device,
            candidate_render_dir=candidate_render_dir,
            clip=self._clip,
        )
