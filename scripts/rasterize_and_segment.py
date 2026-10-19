"""
Rasterize SVG files with CairoSVG, then run Grounding DINO + SAM (HF Transformers).

GroundingDinoProcessor expects `text` as a list of label strings without embedded dots,
e.g. ["a building"] -> merged to "a building." internally.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import cairosvg


def svg_to_png_bytes(svg_path: Path, dpi: float = 96.0) -> bytes:
    """Rasterize SVG to PNG bytes. Uses CairoSVG (vector → raster)."""
    return cairosvg.svg2png(url=str(svg_path), dpi=dpi)


def load_image_png(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def query_to_candidate_labels(query: str) -> list[str]:
    """One phrase for Grounding DINO list-of-labels API (no '.' inside each string)."""
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
) -> tuple[np.ndarray, list[dict]]:
    """
    Returns:
      combined_mask: (H, W) bool, union of SAM masks for all DINO boxes
      debug: one dict per image with boxes/scores/text_labels
    """
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

    debug = [
        {
            "boxes": boxes.cpu(),
            "scores": scores.cpu(),
            "text_labels": text_lbl,
        }
    ]

    combined = np.zeros((h, w), dtype=bool)
    if boxes.numel() == 0:
        return combined, debug

    boxes_xyxy = boxes.cpu().tolist()
    for box in boxes_xyxy:
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
        combined |= m

    return combined, debug


def overlay_mask(rgb: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (255, 64, 64)) -> Image.Image:
    base = np.array(rgb).astype(np.float32)
    mo = mask.astype(np.float32)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = base * (1 - 0.45 * mo) + color_arr * (0.45 * mo)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main() -> int:
    _repo = Path(__file__).resolve().parents[1]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from svg_segmentator.repo_env import load_repo_dotenv

    load_repo_dotenv()

    p = argparse.ArgumentParser(description="SVG → PNG (CairoSVG) → Grounding DINO + SAM mask")
    p.add_argument("--svgs-dir", type=Path, default=Path("svgs"))
    p.add_argument("--out-dir", type=Path, default=Path("output") / "grounded_sam")
    p.add_argument("--query", type=str, default="building", help='Object phrase, e.g. "building" or "a house"')
    p.add_argument("--dpi", type=float, default=96.0, help="Rasterization DPI for CairoSVG")
    p.add_argument("--gdino-model", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam-model", type=str, default="facebook/sam-vit-base")
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--device", type=str, default="", help="cuda / cpu; empty = auto")
    args = p.parse_args()

    if not args.svgs_dir.is_dir():
        print(f"Missing SVG directory: {args.svgs_dir}", file=sys.stderr)
        return 1

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(args.svgs_dir.glob("*.svg"))
    if not svg_files:
        print(f"No .svg files in {args.svgs_dir}", file=sys.stderr)
        return 1

    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        SamModel,
        SamProcessor,
    )

    print("Loading Grounding DINO …")
    gdino_processor = AutoProcessor.from_pretrained(args.gdino_model)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to(device)
    gdino_model.eval()

    print("Loading SAM …")
    sam_processor = SamProcessor.from_pretrained(args.sam_model)
    sam_model = SamModel.from_pretrained(args.sam_model).to(device)
    sam_model.eval()

    text_labels = query_to_candidate_labels(args.query)
    print(f"Text labels for Grounding DINO: {text_labels}")

    for svg_path in svg_files:
        stem = svg_path.stem
        print(f"Processing {svg_path.name} …")
        png_bytes = svg_to_png_bytes(svg_path, dpi=args.dpi)
        raster_path = args.out_dir / f"{stem}_raster.png"
        raster_path.write_bytes(png_bytes)

        image = load_image_png(png_bytes)
        mask, dbg = run_grounded_sam(
            image,
            text_labels=text_labels,
            gdino_processor=gdino_processor,
            gdino_model=gdino_model,
            sam_processor=sam_processor,
            sam_model=sam_model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        n_boxes = dbg[0]["boxes"].shape[0] if dbg and dbg[0]["boxes"].numel() else 0
        print(f"  DINO boxes: {n_boxes}, mask pixels: {int(mask.sum())}")

        mask_u8 = (mask.astype(np.uint8) * 255)
        Image.fromarray(mask_u8).save(args.out_dir / f"{stem}_mask.png")
        overlay_mask(image, mask).save(args.out_dir / f"{stem}_overlay.png")
        print(f"  Wrote {raster_path.name}, {stem}_mask.png, {stem}_overlay.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
