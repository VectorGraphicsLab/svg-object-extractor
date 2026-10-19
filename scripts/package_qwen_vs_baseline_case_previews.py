#!/usr/bin/env python3
"""
Pack raster previews for Qwen-vs-baseline comparison cases (11 total).

Bands:
  A — baseline clearly ahead (weak Qwen IoU)
  B — baseline ahead (stronger Qwen IoU)
  C — Qwen ahead vs max IoU over 4 baseline configs (baseline_max_iou >= 0.15)
  D — both strong (Qwen IoU >= 0.80, baseline max IoU >= 0.80; picks include non-trivial scores)

Contact sheet: plain scene raster | **GT+prompt** (first prompt p0 under crop) | GDINO-base box-fill &
SAM ViT-B paths, plus sheet headers **Qwen 3.6** / **Gemma 4** (full OpenRouter ids in README; dirs `qwen-v2` / `gemma-v2` are pipeline slots).

Produces `output/qwen_vs_baseline_20cases/` (+ `contact_sheet.png`, ZIP).

Run from repo root:
  python3 scripts/package_qwen_vs_baseline_case_previews.py
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT_BASE = REPO / "output" / "qwen_vs_baseline_20cases"
CASES_DIR = OUT_BASE / "cases"

PREV = REPO / "output" / "gt_baseline_full" / "previews"
QWEN_PREV = REPO / "output" / "llm_bridge_full_qwen_eligible" / "previews"
GEMMA_PREV = REPO / "output" / "llm_bridge_full_gemma_eligible" / "previews"
PRED_PATH = REPO / "output" / "llm_bridge_full_qwen_eligible" / "predictions.ndjson"

NCOL = 9

# OpenRouter model ids (see `_unpacked_openrouter_samples_v2_20260508/scripts/export_openrouter_samples_v2.py`).
# `qwen-v2` / `gemma-v2` under `.../llm/` are internal segmenter slot names, not “model v2”.
LLM_QWEN_OPENROUTER_ID = "qwen/qwen3.6-35b-a3b"
LLM_GEMMA_OPENROUTER_ID = "google/gemma-4-26b-a4b-it"
LLM_QWEN_SHEET_LABEL = "Qwen 3.6"
LLM_GEMMA_SHEET_LABEL = "Gemma 4"

# band, band_note, svg_id, annotation_id, prompt_index, best_det, best_seg (max IoU among 4 configs), slug
CASES: list[tuple[str, str, int, int, int, str, str, str]] = [
    ("A", "low IoU Qwen", 171, 240, 0, "gdino-base", "sam-vit-base", "baby_chicks"),
    ("A", "low IoU Qwen", 24, 63, 0, "gdino-base", "sam-vit-base", "leaves"),
    ("A", "low IoU Qwen", 308, 597, 0, "gdino-base", "box-fill", "smiling_face"),
    ("A", "low IoU Qwen", 162, 246, 0, "gdino-base", "box-fill", "cat"),
    ("A", "low IoU Qwen", 146, 910, 2, "gdino-base", "sam-vit-base", "work_report"),
    ("B", "meaningful Qwen", 227, 294, 1, "gdino-base", "box-fill", "star"),
    ("B", "meaningful Qwen", 77, 223, 0, "gdino-base", "box-fill", "woman"),
    ("B", "meaningful Qwen", 47, 74, 0, "gdino-base", "box-fill", "axe"),
    ("B", "meaningful Qwen", 59, 238, 0, "gdino-base", "box-fill", "fire"),
    ("C", "Qwen better than baseline", 221, 129, 0, "gdino-base", "box-fill", "sun"),
    ("C", "Qwen better than baseline", 14, 2430, 0, "gdino-base", "box-fill", "red_arc"),
]

COLUMN_FILES: list[tuple[str, str]] = [
    ("01_original_raster.png", "Raster"),
    ("02_gt_extracted.png", "GT+prompt"),
    ("03_gdino_base_dino_boxes.png", "GDINO-base · boxes"),
    ("04_gdino_base_boxfill_mask_union.png", "GDINO-base · mask"),
    ("05_gdino_base_boxfill_extracted_union.png", "GDINO-base · RGB"),
    ("06_gdino_base_sam_vit_mask_union.png", "SAM ViT-B · mask"),
    ("07_gdino_base_sam_vit_extracted_union.png", "SAM ViT-B · RGB"),
    ("08_qwen_extracted_union.png", LLM_QWEN_SHEET_LABEL),
    ("09_gemma_extracted_union.png", LLM_GEMMA_SHEET_LABEL),
]


def column_paths(svg: int, ann: int, p: int) -> list[Path]:
    ann_dir = PREV / str(svg) / f"ann_{ann}"
    pdir = ann_dir / f"p{p}"
    bf = pdir / "gdino-base" / "box-fill"
    sam = pdir / "gdino-base" / "sam-vit-base"
    return [
        PREV / str(svg) / "raster.png",
        ann_dir / "gt_extracted.png",
        pdir / "gdino-base" / "dino_boxes.png",
        bf / "mask_union.png",
        bf / "extracted_union.png",
        sam / "mask_union.png",
        sam / "extracted_union.png",
        QWEN_PREV / str(svg) / f"ann_{ann}" / f"p{p}" / "llm" / "qwen-v2" / "extracted_union.png",
        GEMMA_PREV / str(svg) / f"ann_{ann}" / f"p{p}" / "llm" / "gemma-v2" / "extracted_union.png",
    ]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        p = Path(fp)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_centered(
    dr: ImageDraw.ImageDraw, cx: int, y: int, text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]
) -> None:
    bbox = dr.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    dr.text((cx - tw // 2, y), text, fill=fill, font=font)


def draw_column_header_vcentered(
    dr: ImageDraw.ImageDraw,
    cx: int,
    hdr_h: int,
    label: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int = 2,
) -> None:
    """One or more lines (`\\n` in label), block vertically centered in header band."""
    lines = label.split("\n")
    heights: list[int] = []
    for ln in lines:
        bb = dr.textbbox((0, 0), ln, font=font)
        heights.append(bb[3] - bb[1])
    block_h = sum(heights) + line_gap * max(0, len(lines) - 1)
    y = max(0, (hdr_h - block_h) // 2)
    for ln, h in zip(lines, heights, strict=True):
        draw_centered(dr, cx, y, ln, font, fill)
        y += h + line_gap


def load_prompts() -> dict[tuple[int, int, int], str]:
    out: dict[tuple[int, int, int], str] = {}
    if not PRED_PATH.is_file():
        return out
    with open(PRED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            r = json.loads(line)
            key = (int(r["svg_id"]), int(r["annotation_id"]), int(r["prompt_index"]))
            out[key] = str(r.get("prompt_text") or "").strip()
    return out


def wrap_lines(
    text: str, font: ImageFont.ImageFont, dr: ImageDraw.ImageDraw, max_w: int
) -> list[str]:
    text = (text or "").strip() or "—"
    words = text.split()
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        bbox = dr.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def cell_gt_with_prompt_below(
    gt_im: Image.Image,
    prompt: str,
    cell_w: int,
    cell_h: int,
    bg: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> Image.Image:
    """GT thumbnail on top; wrapped prompt centered below in the same cell."""
    cell = Image.new("RGB", (cell_w, cell_h), bg)
    dr = ImageDraw.Draw(cell)
    pad = 8
    max_tw = cell_w - 2 * pad
    lines = wrap_lines(prompt, font, dr, max_tw)
    probe = dr.textbbox((0, 0), "Hg", font=font)
    line_h = max(1, probe[3] - probe[1]) + 2
    max_lines = min(12, max(2, int((cell_h * 0.46) // max(1, line_h))))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        ell = " …"
        last = lines[-1].rstrip()
        while (
            last
            and dr.textbbox((0, 0), last + ell, font=font)[2]
            - dr.textbbox((0, 0), last + ell, font=font)[0]
            > max_tw
        ):
            last = last[:-1]
        lines[-1] = (last.rstrip() + ell).strip() or ell.strip()

    text_h = len(lines) * line_h + 12
    text_h = min(int(cell_h * 0.55), max(text_h, 28))
    img_zone_h = max(40, cell_h - text_h - pad * 2)

    t = thumb_fit(gt_im, cell_w - 2 * pad, img_zone_h)
    tw, th = t.size
    cell.paste(t.convert("RGB"), ((cell_w - tw) // 2, pad + (img_zone_h - th) // 2))

    y = pad + img_zone_h + 6
    cx = cell_w // 2
    for ln in lines:
        draw_centered(dr, cx, y, ln, font, (45, 45, 45))
        y += line_h
    return cell


def thumb_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img = img.convert("RGBA")
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def cell_with_centered_image(im: Image.Image, cell_w: int, cell_h: int, bg: tuple[int, int, int]) -> Image.Image:
    inner_pad = 6
    t = thumb_fit(im, cell_w - 2 * inner_pad, cell_h - 2 * inner_pad)
    cell = Image.new("RGB", (cell_w, cell_h), bg)
    tw, th = t.size
    cell.paste(t.convert("RGB"), ((cell_w - tw) // 2, (cell_h - th) // 2))
    return cell


def hconcat(images: list[Image.Image], gap: int, bg: tuple[int, int, int]) -> Image.Image:
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        y = (h - im.height) // 2
        out.paste(im, (x, y))
        x += im.width + gap
    return out


def grid_width(cell_w: int, gap: int, ncol: int) -> int:
    return cell_w * ncol + gap * (ncol - 1)


def main() -> None:
    assert len(COLUMN_FILES) == NCOL

    shutil.rmtree(OUT_BASE, ignore_errors=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    prompts_map = load_prompts()

    cell_w, cell_h = 280, 340
    gap = 10
    hdr_bg = (255, 255, 255)
    cell_bg = (248, 248, 248)
    gw = grid_width(cell_w, gap, NCOL)

    meta: list[dict] = []
    sheet_rows: list[Image.Image] = []

    font_prompt = load_font(30)  # prompt text under GT
    font_header = load_font(24)  # first header row (column titles), slightly smaller

    for i, (band, band_note, svg, ann, p, det, seg, slug) in enumerate(CASES, start=1):
        paths = column_paths(svg, ann, p)
        for name, src in zip([x[0] for x in COLUMN_FILES], paths, strict=True):
            if not src.is_file():
                raise FileNotFoundError(f"missing {name}: {src}")

        case_name = f"{i:02d}_band{band}_svg{svg}_ann{ann}_p{p}_{slug}"
        d = CASES_DIR / case_name
        d.mkdir(parents=True, exist_ok=False)
        for (fname, _), src in zip(COLUMN_FILES, paths, strict=True):
            shutil.copy2(src, d / fname)

        prompt_text = prompts_map.get((svg, ann, p), "")
        # Under GT on the sheet: always annotation's first prompt (p=0), not the row's p.
        prompt_under_gt = prompts_map.get((svg, ann, 0), "") or prompt_text
        (d / "00_prompt.txt").write_text(prompt_text + ("\n" if prompt_text else ""), encoding="utf-8")

        rel_paths = {COLUMN_FILES[j][0]: str(paths[j].relative_to(REPO)) for j in range(len(paths))}
        entry = {
            "index": i,
            "band": band,
            "band_note": band_note,
            "svg_id": svg,
            "annotation_id": ann,
            "prompt_index": p,
            "slug": slug,
            "prompt_text": prompt_text,
            "reference_best_iou_baseline_pair": f"{det}|{seg}",
            "paths_relative_to_repo": rel_paths,
        }
        meta.append(entry)
        with open(d / "case.json", "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        imgs = [Image.open(pth) for pth in paths]
        cells = [
            cell_with_centered_image(imgs[0], cell_w, cell_h, cell_bg),
            cell_gt_with_prompt_below(imgs[1], prompt_under_gt, cell_w, cell_h, cell_bg, font_prompt),
            *[cell_with_centered_image(im, cell_w, cell_h, cell_bg) for im in imgs[2:]],
        ]
        for im in imgs:
            im.close()

        row_img = hconcat(cells, gap, hdr_bg)
        sheet_rows.append(row_img)

    hdr_h = 58
    header = Image.new("RGB", (gw, hdr_h), hdr_bg)
    hdr_dr = ImageDraw.Draw(header)
    for col_i, (_, label) in enumerate(COLUMN_FILES):
        cx = col_i * (cell_w + gap) + cell_w // 2
        draw_column_header_vcentered(hdr_dr, cx, hdr_h, label, font_header, (35, 35, 35))

    margin = 14
    total_h = hdr_h + margin + sum(r.height + 8 for r in sheet_rows) + margin
    sheet = Image.new("RGB", (gw + 2 * margin, total_h), (255, 255, 255))
    y = margin
    sheet.paste(header, (margin, y))
    y += hdr_h + margin
    for row in sheet_rows:
        sheet.paste(row, (margin, y))
        y += row.height + 8

    readme = OUT_BASE / "README.md"
    lines = [
        "# Qwen vs baseline — 11 raster cases",
        "",
        "- **Band A:** baseline ahead, weak Qwen IoU (0 &lt; IoU ≤ 0.12).",
        "- **Band B:** baseline ahead, stronger Qwen IoU (≥ 0.28).",
        "- **Band C:** Qwen IoU **greater than** max baseline IoU over 4 configs, baseline max IoU ≥ 0.15.",
        "- **Band D:** both strong — Qwen IoU ≥ 0.80 **and** baseline max IoU ≥ 0.80; excludes rows where both are exactly 1.0 when possible (see curated list).",
        "",
        f"Each case folder has `00_prompt.txt` (the row's **`prompt_index`**) plus **{NCOL}** PNGs. On **`contact_sheet.png`**, text under GT is the annotation's **first prompt (`prompt_index` 0)** only; column 1 is plain raster.",
        "",
        "1. `01_original_raster.png` — SVG raster.",
        "2. `02_gt_extracted.png` — GT crop.",
        "3. `03_gdino_base_dino_boxes.png` — `gdino-base/dino_boxes.png`.",
        "4. `04_gdino_base_boxfill_mask_union.png` — **gdino-base box-fill** `mask_union.png`.",
        "5. `05_gdino_base_boxfill_extracted_union.png` — **gdino-base box-fill** `extracted_union.png`.",
        "6. `06_gdino_base_sam_vit_mask_union.png` — **SAM** union mask `gdino-base/sam-vit-base/mask_union.png`.",
        "7. `07_gdino_base_sam_vit_extracted_union.png` — **SAM ViT-B** RGB crop `gdino-base/sam-vit-base/extracted_union.png`.",
        f"8. `08_qwen_extracted_union.png` — **Qwen 3.6 35B A3B** (`{LLM_QWEN_OPENROUTER_ID}`), file `llm/qwen-v2/extracted_union.png`.",
        f"9. `09_gemma_extracted_union.png` — **Gemma 4 26B A4B IT** (`{LLM_GEMMA_OPENROUTER_ID}`), file `llm/gemma-v2/extracted_union.png`.",
        "",
        "`reference_best_iou_baseline_pair` in `case.json`: detector|segmenter with **max IoU** among the four baseline configs for that task.",
        "",
        "| # | Band | svg_id | ann | p | Max-IoU baseline pair | slug |",
        "|---|------|--------|-----|---|----------------------|------|",
    ]
    band_desc = {
        "A": "weak Qwen IoU",
        "B": "stronger Qwen IoU",
        "C": "Qwen wins vs max baseline",
        "D": "both strong",
    }
    for i, (band, _, svg, ann, p, det, seg, slug) in enumerate(CASES, start=1):
        lines.append(
            f"| {i} | {band} ({band_desc[band]}) | {svg} | {ann} | {p} | `{det}` \\| `{seg}` | {slug} |"
        )
    lines += ["", f"Bundled folder: `{OUT_BASE.relative_to(REPO)}`", ""]
    readme.write_text("\n".join(lines), encoding="utf-8")

    with open(OUT_BASE / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    sheet_path = OUT_BASE / "contact_sheet.png"
    sheet.save(sheet_path, optimize=True)

    zip_path = REPO / "output" / "qwen_vs_baseline_20cases.zip"
    if zip_path.is_file():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in OUT_BASE.rglob("*"):
            if path.is_file():
                arc = path.relative_to(OUT_BASE.parent)
                zf.write(path, arc.as_posix())

    print(f"Wrote {OUT_BASE}")
    print(f"Contact sheet: {sheet_path}")
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
