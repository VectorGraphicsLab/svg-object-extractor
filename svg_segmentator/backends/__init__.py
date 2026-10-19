from svg_segmentator.backends.interfaces import DetectionBackend, SegmentationBackend
from svg_segmentator.backends.types import DetectionBox, DetectionResult, RunContext, SegmentationResult

__all__ = [
    "DetectionBackend",
    "SegmentationBackend",
    "DetectionBox",
    "DetectionResult",
    "SegmentationResult",
    "RunContext",
]
