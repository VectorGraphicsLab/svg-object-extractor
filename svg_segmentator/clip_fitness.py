"""CLIP image–text similarity (Hugging Face ``openai/clip-vit-base-patch32``)."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_clip(model_id: str = "openai/clip-vit-base-patch32", device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device


@torch.inference_mode()
def clip_image_text_score(
    image_rgb: Image.Image,
    text: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
) -> float:
    """
    Higher is better: diagonal logit from CLIP when one image and one text (scaled cosine).
    """
    if image_rgb.mode != "RGB":
        image_rgb = image_rgb.convert("RGB")
    inputs = processor(text=[text], images=image_rgb, return_tensors="pt", padding=True).to(device)
    out = model(**inputs)
    # shape (1, 1) for single pair
    return float(out.logits_per_image[0, 0].item())
