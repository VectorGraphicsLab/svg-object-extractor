from __future__ import annotations

import numpy as np
import torch
from transformers import SamModel, SamProcessor

from svg_segmentator.backends.interfaces import SegmentationBackend
from svg_segmentator.backends.types import DetectionResult, RunContext, SegmentationResult


class SamHFSegmenter(SegmentationBackend):
    def __init__(self, model_id: str, backend_id: str):
        self.model_id = model_id
        self.backend_id = backend_id
        self._processor = None
        self._model = None
        self._loaded_device: torch.device | None = None

    def _ensure_loaded(self, device: torch.device) -> None:
        if self._processor is None:
            self._processor = SamProcessor.from_pretrained(self.model_id)
        if self._model is None or self._loaded_device != device:
            self._model = SamModel.from_pretrained(self.model_id).to(device)
            self._model.eval()
            self._loaded_device = device

    def segment(self, ctx: RunContext, detections: DetectionResult) -> SegmentationResult:
        self._ensure_loaded(ctx.device)
        assert self._processor is not None
        assert self._model is not None

        w, h = ctx.image_rgb.size
        combined = np.zeros((h, w), dtype=bool)
        masks_out: list[np.ndarray] = []

        for b in detections.boxes:
            box = [b.x0, b.y0, b.x1, b.y1]
            inputs = self._processor(
                ctx.image_rgb,
                input_boxes=[[box]],
                return_tensors="pt",
            ).to(ctx.device)
            with torch.no_grad():
                outputs = self._model(**inputs, multimask_output=False)

            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            mask = masks[0, 0].numpy() > 0.0
            combined |= mask
            masks_out.append(mask)

        return SegmentationResult(masks=masks_out, combined_mask=combined)
