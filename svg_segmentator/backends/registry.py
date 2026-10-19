from __future__ import annotations

from dataclasses import dataclass

from svg_segmentator.backends.detectors.gdino_hf import GroundingDinoHFDetector
from svg_segmentator.backends.interfaces import DetectionBackend, SegmentationBackend
from svg_segmentator.backends.segmenters.box_fill import BoxFillSegmenter
from svg_segmentator.backends.segmenters.sam_hf import SamHFSegmenter


@dataclass(frozen=True)
class DetectorSpec:
    backend_id: str
    model_id: str


@dataclass(frozen=True)
class SegmenterSpec:
    backend_id: str
    model_id: str | None = None


_DETECTOR_SPECS: dict[str, DetectorSpec] = {
    "gdino-tiny": DetectorSpec(backend_id="gdino-tiny", model_id="IDEA-Research/grounding-dino-tiny"),
    "gdino-base": DetectorSpec(backend_id="gdino-base", model_id="IDEA-Research/grounding-dino-base"),
    # MM Grounding DINO (HF): a lightweight checkpoint that's a strong A/B baseline vs IDEA tiny.
    "mmgdino-tiny-v3det": DetectorSpec(
        backend_id="mmgdino-tiny-v3det",
        model_id="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
    ),
    # IDEA-Research does not publish a HF "grounding-dino-large"; this is MM Grounding DINO Large (same AutoModel API).
    "gdino-large": DetectorSpec(
        backend_id="gdino-large",
        model_id="openmmlab-community/mm_grounding_dino_large_all",
    ),
}

_SEGMENTER_SPECS: dict[str, SegmenterSpec] = {
    "sam-vit-base": SegmenterSpec(backend_id="sam-vit-base", model_id="facebook/sam-vit-base"),
    "box-fill": SegmenterSpec(backend_id="box-fill"),
}


def list_detector_ids() -> list[str]:
    return sorted(_DETECTOR_SPECS.keys())


def list_segmenter_ids() -> list[str]:
    return sorted(_SEGMENTER_SPECS.keys())


def build_detector(detector_id: str, *, override_model_id: str | None = None) -> DetectionBackend:
    spec = _DETECTOR_SPECS.get(detector_id)
    if spec is None:
        raise ValueError(f"Unknown detector: {detector_id}. Available: {', '.join(list_detector_ids())}")
    return GroundingDinoHFDetector(
        model_id=override_model_id or spec.model_id,
        backend_id=detector_id,
    )


def build_segmenter(segmenter_id: str, *, override_model_id: str | None = None) -> SegmentationBackend:
    spec = _SEGMENTER_SPECS.get(segmenter_id)
    if spec is None:
        raise ValueError(f"Unknown segmenter: {segmenter_id}. Available: {', '.join(list_segmenter_ids())}")
    if segmenter_id == "box-fill":
        return BoxFillSegmenter(backend_id="box-fill")
    assert spec.model_id is not None
    return SamHFSegmenter(
        model_id=override_model_id or spec.model_id,
        backend_id=segmenter_id,
    )
