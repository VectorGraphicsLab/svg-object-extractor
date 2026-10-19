"""PNG previews for GT baseline runs (boxes, masks, overlays)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from svg_segmentator.backends.types import DetectionResult
from svg_segmentator.extract import LeafMaskCache, write_tree


def overlay_bool_mask(
    rgb: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> Image.Image:
    """Alpha-blend a boolean mask onto RGB (same idea as ``scripts/rasterize_and_segment``)."""
    m = np.asarray(mask, dtype=bool)
    base = np.array(rgb).astype(np.float32)
    mo = m.astype(np.float32)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = base * (1 - alpha * mo) + color_arr * (alpha * mo)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def draw_detection_boxes(
    rgb: Image.Image,
    det: DetectionResult,
    *,
    outline: tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
) -> Image.Image:
    """Draw DINO boxes and scores (same spirit as ``pipeline.run_pipeline_combo``)."""
    img = rgb.copy()
    draw = ImageDraw.Draw(img)
    for idx, box in enumerate(det.boxes):
        x0, y0, x1, y1 = float(box.x0), float(box.y0), float(box.x1), float(box.y1)
        draw.rectangle((x0, y0, x1, y1), outline=outline, width=width)
        label_txt = (box.label or "object").strip()
        draw.text((x0 + 4, y0 + 4), f"{idx}:{label_txt} {box.score:.3f}", fill=outline)
    return img


def mask_bool_to_u8_png(mask: np.ndarray) -> Image.Image:
    m = np.asarray(mask, dtype=bool)
    return Image.fromarray((m.astype(np.uint8) * 255))


def leaf_union_from_seg_indices(leaf_cache: LeafMaskCache, seg_flags: list[int]) -> np.ndarray:
    """Union of leaf raster masks where ``seg_flags[idx]==1`` for ``data-local-id`` ``seg_{idx}``."""
    if not leaf_cache.leaf_masks:
        return np.zeros((1, 1), dtype=bool)
    out = np.zeros_like(leaf_cache.leaf_masks[0], dtype=bool)
    for leaf, lm in zip(leaf_cache.leaves, leaf_cache.leaf_masks):
        lid = leaf.get("data-local-id") or ""
        try:
            idx = int(lid.replace("seg_", ""))
        except ValueError:
            continue
        if 0 <= idx < len(seg_flags) and seg_flags[idx]:
            out |= lm.astype(bool)
    return out


def write_png_once(path: Path, img: Image.Image) -> None:
    """Skip if file exists (resume-friendly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        img.save(path)


def write_svg_once(path: Path, root: ET.Element) -> None:
    """Pruned SVG subtree (same semantics as ``pipeline`` ``write_tree``); skip if exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        write_tree(path, root)


def write_preview_meta(path: Path, data: dict[str, Any]) -> None:
    """Human-readable JSON next to previews; overwritten each visit (keeps thresholds in sync)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
