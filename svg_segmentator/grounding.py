from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def query_to_candidate_labels(query: str) -> list[str]:
    q = query.strip().lower()
    if not q:
        return ["object"]
    if q.startswith("a ") or q.startswith("an "):
        return [q]
    return [f"a {q}"]


def run_grounded_sam(
    image: Image.Image,
    text_labels: list[str],
    gdino_processor,
    gdino_model,
    sam_processor,
    sam_model,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
    use_sam: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], list[dict], list[dict]]:
    w, h = image.size
    target_sizes = [(h, w)]

    inputs_dino = gdino_processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_dino = gdino_model(**inputs_dino)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs_dino,
        inputs_dino.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = results["boxes"]
    scores = results["scores"]
    text_lbl = results.get("text_labels", results.get("labels", []))

    boxes_cpu = boxes.cpu()
    scores_cpu = scores.cpu()

    debug = [
        {
            "boxes": boxes_cpu,
            "scores": scores_cpu,
            "text_labels": text_lbl,
        }
    ]

    combined = np.zeros((h, w), dtype=bool)
    per_box_masks: list[np.ndarray] = []
    per_box_meta: list[dict] = []
    if boxes.numel() == 0:
        return combined, per_box_masks, per_box_meta, debug

    for i, box in enumerate(boxes_cpu.tolist()):
        if use_sam:
            inputs_sam = sam_processor(
                image,
                input_boxes=[[box]],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs_sam = sam_model(**inputs_sam, multimask_output=False)

            masks = sam_processor.image_processor.post_process_masks(
                outputs_sam.pred_masks.cpu(),
                inputs_sam["original_sizes"].cpu(),
                inputs_sam["reshaped_input_sizes"].cpu(),
            )[0]
            m = masks[0, 0].numpy() > 0.0
        else:
            x0, y0, x1, y1 = box
            xi0 = max(0, min(w, int(round(x0))))
            yi0 = max(0, min(h, int(round(y0))))
            xi1 = max(0, min(w, int(round(x1))))
            yi1 = max(0, min(h, int(round(y1))))
            m = np.zeros((h, w), dtype=bool)
            if xi1 > xi0 and yi1 > yi0:
                m[yi0:yi1, xi0:xi1] = True
        combined |= m
        per_box_masks.append(m)
        label = ""
        if i < len(text_lbl):
            label = str(text_lbl[i])
        per_box_meta.append(
            {
                "box": box,
                "score": float(scores_cpu[i].item()),
                "text_label": label,
            }
        )

    return combined, per_box_masks, per_box_meta, debug
