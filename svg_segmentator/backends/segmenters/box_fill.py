from __future__ import annotations

import numpy as np

from svg_segmentator.backends.interfaces import SegmentationBackend
from svg_segmentator.backends.types import DetectionResult, RunContext, SegmentationResult


class BoxFillSegmenter(SegmentationBackend):
    def __init__(self, backend_id: str = "box-fill"):
        self.backend_id = backend_id

    def segment(self, ctx: RunContext, detections: DetectionResult) -> SegmentationResult:
        w, h = ctx.image_rgb.size
        combined = np.zeros((h, w), dtype=bool)
        masks_out: list[np.ndarray] = []

        for b in detections.boxes:
            xi0 = max(0, min(w, int(round(b.x0))))
            yi0 = max(0, min(h, int(round(b.y0))))
            xi1 = max(0, min(w, int(round(b.x1))))
            yi1 = max(0, min(h, int(round(b.y1))))
            m = np.zeros((h, w), dtype=bool)
            if xi1 > xi0 and yi1 > yi0:
                m[yi0:yi1, xi0:xi1] = True
            combined |= m
            masks_out.append(m)

        return SegmentationResult(masks=masks_out, combined_mask=combined)
