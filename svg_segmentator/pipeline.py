from __future__ import annotations

import csv
import io
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from svg_segmentator.backends.interfaces import DetectionBackend, SegmentationBackend
from svg_segmentator.backends.types import RunContext
from svg_segmentator.clip_fitness import clip_image_text_score, load_clip
from svg_segmentator.extract import (
    build_leaf_mask_cache,
    load_tagged_svg,
    prune_from_scores,
    score_leaf_mask_cache,
    write_tree,
)
from svg_segmentator.grounding import query_to_candidate_labels
from svg_segmentator.render import png_bytes_to_rgba, svg_file_to_png_bytes


@dataclass(frozen=True)
class PipelineConfig:
    query: str
    dpi: float = 96.0
    score_threshold: float = 0.8
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    max_leaves: int | None = None
    clip_model_id: str = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class PipelineRunSummary:
    """Metrics from a finished ``run_pipeline_combo`` (for batch eval / logging)."""

    n_boxes: int
    image_width: int
    image_height: int
    clip_scores: tuple[tuple[str, float], ...]


def _safe_segment(s: str) -> str:
    import re

    s = re.sub(r"[^\w\-.]+", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def run_pipeline_combo(
    *,
    svg_path: Path,
    out_dir: Path,
    detector: DetectionBackend,
    segmenter: SegmentationBackend,
    config: PipelineConfig,
    device: torch.device,
    candidate_render_dir: Path | None = None,
    clip: tuple[CLIPProcessor, CLIPModel, torch.device] | None = None,
) -> PipelineRunSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = svg_path.stem

    png_bytes = svg_file_to_png_bytes(svg_path, dpi=config.dpi)
    shutil.copy2(svg_path, out_dir / f"{stem}_source.svg")
    (out_dir / f"{stem}_source.png").write_bytes(png_bytes)
    (out_dir / f"{stem}_raster.png").write_bytes(png_bytes)
    rgba = png_bytes_to_rgba(png_bytes)
    rgb_work = rgba.convert("RGB")

    text_labels = query_to_candidate_labels(config.query)
    ctx = RunContext(image_rgb=rgb_work, device=device, query=config.query, text_labels=text_labels)

    det_mid = getattr(detector, "model_id", "") or ""
    seg_mid = getattr(segmenter, "model_id", "") or ""
    print(f"Detector: {detector.backend_id}" + (f" ({det_mid})" if det_mid else ""))
    print(f"Segmenter: {segmenter.backend_id}" + (f" ({seg_mid})" if seg_mid else ""))
    print(f"Query: {config.query!r}")
    print(f"Detector text_labels: {text_labels}")
    print(
        "Thresholds: "
        f"box={config.box_threshold:g}, text={config.text_threshold:g}, leaf_prune={config.score_threshold:g}, "
        f"dpi={config.dpi:g}, image={rgb_work.size[0]}x{rgb_work.size[1]}"
    )

    detections = detector.detect(
        ctx,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
    )
    n_boxes = len(detections.boxes)
    print(f"Detector boxes: {n_boxes}")
    if n_boxes == 0:
        print(
            "NOTE: zero detections → empty mask is expected with box-fill; "
            "try lowering --box-threshold, "
            "and/or --text-threshold (Grounding DINO text match), increasing --dpi, "
            "or rephrasing the query (synonyms / more concrete noun)."
        )
    elif n_boxes <= 10:
        for i, b in enumerate(detections.boxes):
            lbl = (b.label or "").strip()
            print(f"  - box[{i}] score={b.score:.3f} label={lbl!r} xyxy=({b.x0:.1f},{b.y0:.1f},{b.x1:.1f},{b.y1:.1f})")

    # Helpful sanity check: models sometimes predict boxes slightly outside the raster bounds.
    # Our segmenters clamp to the image; if clamping is large, overlays can look "wrong".
    iw, ih = rgb_work.size
    for i, b in enumerate(detections.boxes):
        clipped_x0 = max(0.0, min(float(iw), b.x0))
        clipped_y0 = max(0.0, min(float(ih), b.y0))
        clipped_x1 = max(0.0, min(float(iw), b.x1))
        clipped_y1 = max(0.0, min(float(ih), b.y1))
        dx = abs(b.x0 - clipped_x0) + abs(b.x1 - clipped_x1)
        dy = abs(b.y0 - clipped_y0) + abs(b.y1 - clipped_y1)
        if dx > 0.5 or dy > 0.5:
            print(
                f"  WARN box[{i}] clipped to image bounds: raw=({b.x0:.1f},{b.y0:.1f},{b.x1:.1f},{b.y1:.1f}) "
                f"-> ({clipped_x0:.1f},{clipped_y0:.1f},{clipped_x1:.1f},{clipped_y1:.1f}) "
                f"(image {iw}x{ih})"
            )

    seg = segmenter.segment(ctx, detections)

    boxes_preview = rgb_work.copy()
    draw = ImageDraw.Draw(boxes_preview)
    for idx, box in enumerate(detections.boxes):
        draw.rectangle((box.x0, box.y0, box.x1, box.y1), outline=(255, 0, 0), width=3)
        label_txt = (box.label or "object").strip()
        draw.text((box.x0 + 4, box.y0 + 4), f"{idx}:{label_txt} {box.score:.3f}", fill=(255, 0, 0))
    (out_dir / f"{stem}_dino_boxes.png").write_bytes(_img_to_png_bytes(boxes_preview))

    Image.fromarray(seg.combined_mask.astype(np.uint8) * 255).save(out_dir / f"{stem}_query_mask.png")

    root, pmap, leaves, _ = load_tagged_svg(svg_path)
    if config.max_leaves is not None:
        leaves = leaves[: config.max_leaves]
    leaf_cache = build_leaf_mask_cache(
        root=root,
        pmap=pmap,
        leaves=leaves,
        dpi=config.dpi,
        # Keep leaf renders aligned with the original SVG raster (not the SDXL-stylized view).
        rgb_full=rgb_src,
        candidate_render_dir=candidate_render_dir,
    )

    if clip is not None:
        clip_processor, clip_model, clip_device = clip
    else:
        clip_processor, clip_model, clip_device = load_clip(config.clip_model_id, device=device)
    scores = score_leaf_mask_cache(leaf_cache, seg.combined_mask)
    out_root, _ = prune_from_scores(leaf_cache, scores, config.score_threshold)
    out_svg = out_dir / f"{stem}_extracted.svg"
    write_tree(out_svg, out_root)
    extracted_png = svg_file_to_png_bytes(out_svg, dpi=config.dpi)
    extracted_png_path = out_dir / f"{stem}_extracted.png"
    extracted_png_path.write_bytes(extracted_png)
    combined_clip = clip_image_text_score(
        Image.open(io.BytesIO(extracted_png)).convert("RGB"),
        config.query,
        clip_processor,
        clip_model,
        clip_device,
    )

    with (out_dir / f"{stem}_leaf_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data_local_id", "score"])
        writer.writerows(scores)

    clip_rows: list[tuple[str, float]] = [("combined", combined_clip)]
    objects_root = out_dir / "objects"
    for idx, (obj_mask, box) in enumerate(zip(seg.masks, detections.boxes)):
        label_seg = _safe_segment(box.label or "object")
        score_seg = f"{box.score:.3f}"
        obj_name = f"object_{idx:03d}_label-{label_seg}_score-{score_seg}"
        obj_dir = objects_root / obj_name
        obj_dir.mkdir(parents=True, exist_ok=True)
        single_box = rgb_work.copy()
        draw_one = ImageDraw.Draw(single_box)
        draw_one.rectangle((box.x0, box.y0, box.x1, box.y1), outline=(255, 0, 0), width=3)
        label_one = (box.label or "object").strip()
        draw_one.text((box.x0 + 4, box.y0 + 4), f"{idx}:{label_one} {box.score:.3f}", fill=(255, 0, 0))
        (obj_dir / f"{stem}_dino_single_box.png").write_bytes(_img_to_png_bytes(single_box))

        Image.fromarray(obj_mask.astype(np.uint8) * 255).save(obj_dir / f"{stem}_query_mask.png")

        obj_scores = score_leaf_mask_cache(leaf_cache, obj_mask)
        obj_root, _ = prune_from_scores(leaf_cache, obj_scores, config.score_threshold)
        obj_svg = obj_dir / f"{stem}_extracted.svg"
        write_tree(obj_svg, obj_root)
        obj_png_bytes = svg_file_to_png_bytes(obj_svg, dpi=config.dpi)
        (obj_dir / f"{stem}_extracted.png").write_bytes(obj_png_bytes)
        obj_clip = clip_image_text_score(
            Image.open(io.BytesIO(obj_png_bytes)).convert("RGB"),
            config.query,
            clip_processor,
            clip_model,
            clip_device,
        )
        clip_rows.append((obj_name, obj_clip))

        with (obj_dir / f"{stem}_leaf_scores.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["data_local_id", "score"])
            writer.writerows(obj_scores)
        with (obj_dir / f"{stem}_clip_score.txt").open("w", encoding="utf-8") as f:
            f.write(f"{obj_clip:.6f}\n")

    with (out_dir / f"{stem}_clip_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "clip_score"])
        writer.writerows(clip_rows)

    return PipelineRunSummary(
        n_boxes=n_boxes,
        image_width=iw,
        image_height=ih,
        clip_scores=tuple(clip_rows),
    )


def _img_to_png_bytes(img: Image.Image) -> bytes:
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()
