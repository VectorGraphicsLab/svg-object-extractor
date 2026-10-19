from __future__ import annotations

import inspect
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from svg_segmentator.backends.interfaces import DetectionBackend
from svg_segmentator.backends.types import DetectionBox, DetectionResult, RunContext


class GroundingDinoHFDetector(DetectionBackend):
    def __init__(self, model_id: str, backend_id: str):
        self.model_id = model_id
        self.backend_id = backend_id
        self._processor = None
        self._model = None
        self._loaded_device: torch.device | None = None

    def _ensure_loaded(self, device: torch.device) -> None:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        if self._model is None or self._loaded_device != device:
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(device)
            self._model.eval()
            self._loaded_device = device

    def detect(self, ctx: RunContext, *, box_threshold: float, text_threshold: float) -> DetectionResult:
        self._ensure_loaded(ctx.device)
        assert self._processor is not None
        assert self._model is not None

        w, h = ctx.image_rgb.size
        target_sizes = [(h, w)]
        inputs = self._processor(images=ctx.image_rgb, text=ctx.text_labels, return_tensors="pt").to(ctx.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        post = self._processor.post_process_grounded_object_detection
        post_kwargs: dict = {
            "outputs": outputs,
            "input_ids": inputs.input_ids,
            "threshold": box_threshold,
            "text_threshold": text_threshold,
            "target_sizes": target_sizes,
        }
        # Newer processors (incl. MM Grounding DINO) accept explicit `text_labels` for postprocessing.
        # Passing the same phrases used at preprocess time avoids ambiguity if `labels` becomes token ids.
        sig = inspect.signature(post)
        if "text_labels" in sig.parameters:
            post_kwargs["text_labels"] = [ctx.text_labels]

        results = post(**post_kwargs)[0]

        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        # Prefer `text_labels` (strings). Touching `labels` can emit FutureWarning on some processor versions
        # because `labels` may become integer token ids.
        raw_labels = results["text_labels"] if "text_labels" in results else results.get("labels", [])

        def _label_for_index(i: int) -> str:
            if i >= len(raw_labels):
                return ""
            v = raw_labels[i]
            # Prefer string labels when the processor provides them.
            if isinstance(v, str):
                return v
            # Some versions return integer label ids; decode via the tokenizer bundled in the processor.
            if isinstance(v, int):
                tok = getattr(self._processor, "tokenizer", None)
                if tok is None:
                    return str(v)
                return tok.decode([v], skip_special_tokens=True).strip()
            # Torch scalar id
            if torch.is_tensor(v) and v.numel() == 1:
                tok = getattr(self._processor, "tokenizer", None)
                if tok is None:
                    return str(int(v.item()))
                return tok.decode([int(v.item())], skip_special_tokens=True).strip()
            return str(v)

        out: list[DetectionBox] = []
        for i, box in enumerate(boxes):
            label = _label_for_index(i)
            out.append(
                DetectionBox(
                    x0=float(box[0]),
                    y0=float(box[1]),
                    x1=float(box[2]),
                    y1=float(box[3]),
                    score=float(scores[i]),
                    label=label,
                )
            )
        return DetectionResult(boxes=out)
