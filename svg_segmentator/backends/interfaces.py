from __future__ import annotations

from abc import ABC, abstractmethod

from svg_segmentator.backends.types import DetectionResult, RunContext, SegmentationResult


class DetectionBackend(ABC):
    backend_id: str

    @abstractmethod
    def detect(self, ctx: RunContext, *, box_threshold: float, text_threshold: float) -> DetectionResult:
        raise NotImplementedError


class SegmentationBackend(ABC):
    backend_id: str

    @abstractmethod
    def segment(self, ctx: RunContext, detections: DetectionResult) -> SegmentationResult:
        raise NotImplementedError
