"""Merged GT previews for ``one_visible_of_many`` groups (OR over leaf binary masks)."""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from svg_gt_baseline.disk_cache import sha256_hex
from svg_gt_baseline.masks import leaf_scores_from_binary_mask, node_count_from_root
from svg_gt_baseline.svg_size import rasterize_svg_path_max_side
from svg_gt_baseline.types import DatasetAnnotationRow, InferenceSettings
from svg_gt_baseline.viz import (
    leaf_union_from_seg_indices,
    mask_bool_to_u8_png,
    overlay_bool_mask,
    write_png_once,
    write_svg_once,
)
from svg_segmentator.extract import build_leaf_mask_cache, load_tagged_svg, prune_from_scores

LOG = logging.getLogger("svg_gt_baseline")

OOM_KIND = "one_visible_of_many"

# Same threshold as ``svg_gt_baseline.engine._GT_PRUNE_SCORE_THRESHOLD``.
_GT_PRUNE_SCORE_THRESHOLD = 0.5

# Must match ``det_ids`` / ``seg_ids`` in ``svg_gt_baseline.engine.run_inference_job``.
OOM_SEGMENTER_MIRROR_DET_IDS: tuple[str, ...] = ("gdino-tiny", "gdino-base")
OOM_SEGMENTER_MIRROR_SEG_IDS: tuple[str, ...] = ("box-fill", "sam-vit-base")


def group_rows_one_visible_of_many(
    rows: list[DatasetAnnotationRow],
) -> dict[tuple[int, tuple[str, ...]], list[DatasetAnnotationRow]]:
    groups: dict[tuple[int, tuple[str, ...]], list[DatasetAnnotationRow]] = {}
    for r in rows:
        if r.annotation_kind != OOM_KIND:
            continue
        key = (r.svg_id, tuple(r.prompt_texts))
        groups.setdefault(key, []).append(r)
    return groups


def merge_binary_masks_or(masks: list[list[int]]) -> list[int]:
    if not masks:
        return []
    n = len(masks[0])
    for m in masks:
        if len(m) != n:
            raise ValueError(f"mask length mismatch: expected {n}, got {len(m)}")
    out = [0] * n
    for m in masks:
        for i, v in enumerate(m):
            if v:
                out[i] = 1
    return out


def _oom_group_dir_slug(prompt_texts: tuple[str, ...]) -> str:
    h = hashlib.sha256("\n".join(prompt_texts).encode("utf-8")).hexdigest()[:12]
    return h


def _write_extracted_svg_raster_png_once(svg_path: Path, png_path: Path, raster_max_side: int) -> None:
    """Cairo raster of extracted SVG (same as per-row ``gt_extracted.png`` pipeline)."""
    if png_path.is_file():
        return
    png_b, _ = rasterize_svg_path_max_side(svg_path, raster_max_side)
    write_png_once(png_path, Image.open(io.BytesIO(png_b)).convert("RGB"))


def mirror_oom_gt_into_segmenter_previews(
    *,
    out_dir: Path,
    svg_id: int,
    members: list[DatasetAnnotationRow],
    prev_dir: Path,
    det_ids: tuple[str, ...] = OOM_SEGMENTER_MIRROR_DET_IDS,
    seg_ids: tuple[str, ...] = OOM_SEGMENTER_MIRROR_SEG_IDS,
) -> int:
    """
    Copy merged GT PNGs next to ``extracted_union.png`` under each
    ``previews/<svg_id>/ann_<id>/p<i>/<det>/<seg>/`` so masks are directly comparable
    (same work raster size as ``raster.png``).
    """
    mask_src = prev_dir / "gt_merged_mask.png"
    overlay_src = prev_dir / "gt_merged_overlay.png"
    ext_svg_src = prev_dir / "gt_merged_extracted.svg"
    ext_png_src = prev_dir / "gt_merged_extracted.png"
    if not mask_src.is_file() or not overlay_src.is_file():
        LOG.warning("OOM mirror skip svg_id=%s (missing merged PNGs under %s)", svg_id, prev_dir)
        return 0
    mask_bytes = mask_src.read_bytes()
    overlay_bytes = overlay_src.read_bytes()
    ext_svg_bytes = ext_svg_src.read_bytes() if ext_svg_src.is_file() else None
    ext_png_bytes = ext_png_src.read_bytes() if ext_png_src.is_file() else None
    if ext_svg_bytes is None or ext_png_bytes is None:
        LOG.warning(
            "OOM mirror svg_id=%s: merged extracted GT missing under %s (only masks mirrored)",
            svg_id,
            prev_dir,
        )
    n_paths = 0
    for r in members:
        for pi in range(len(r.prompt_texts)):
            for det in det_ids:
                for seg in seg_ids:
                    dest = out_dir / "previews" / str(svg_id) / f"ann_{r.annotation_id}" / f"p{pi}" / det / seg
                    if not dest.is_dir():
                        continue
                    (dest / "gt_merged_group_mask.png").write_bytes(mask_bytes)
                    (dest / "gt_merged_group_overlay.png").write_bytes(overlay_bytes)
                    if ext_svg_bytes is not None:
                        (dest / "gt_merged_group_extracted.svg").write_bytes(ext_svg_bytes)
                    if ext_png_bytes is not None:
                        (dest / "gt_merged_group_extracted.png").write_bytes(ext_png_bytes)
                    n_paths += 1
    LOG.info(
        "OOM merged GT mirrored into segmenter dirs svg_id=%s dest_slots=%d "
        "(gt_merged_group_* next to extracted_union; PNG/SVG object = gt_merged_group_extracted.*)",
        svg_id,
        n_paths,
    )
    return n_paths


def write_oom_gt_merged_previews(
    *,
    rows: list[DatasetAnnotationRow],
    out_dir: Any,
    svg_struct_cache: dict[int, tuple[Any, Any, Any, Any, int]],
    rgb_raster_cache: dict[int, tuple[bytes, tuple[int, int]]],
    settings: InferenceSettings,
) -> int:
    """
    For each ``one_visible_of_many`` group (same ``svg_id`` and ``prompt_texts``),
    OR dataset binary masks and write raster previews under
    ``previews/<svg_id>/oom_merged_<slug>/``.
    """
    out_dir = Path(out_dir)
    groups = group_rows_one_visible_of_many(rows)
    n_written = 0
    for (svg_id, prompt_texts), members in sorted(groups.items()):
        if svg_id not in svg_struct_cache:
            LOG.warning("OOM merged GT skip svg_id=%s (svg not loaded)", svg_id)
            continue
        if svg_id not in rgb_raster_cache:
            LOG.warning("OOM merged GT skip svg_id=%s (no raster cache)", svg_id)
            continue

        _svg_path, root, pmap, leaves, n_parsed = svg_struct_cache[svg_id]
        bad = [r for r in members if r.node_count != n_parsed]
        if bad:
            LOG.warning(
                "OOM merged GT skip svg_id=%s (node_count mismatch vs parsed ann_ids=%s)",
                svg_id,
                [r.annotation_id for r in bad],
            )
            continue

        png_bytes, (rw, rh) = rgb_raster_cache[svg_id]
        rgb_work = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        leaf_cache = build_leaf_mask_cache(
            root,
            pmap,
            leaves,
            settings.leaf_render_dpi,
            rgb_work,
            None,
            output_wh=(rw, rh),
        )
        merged = merge_binary_masks_or([r.binary_mask for r in members])
        gt_r = leaf_union_from_seg_indices(leaf_cache, merged)
        slug = _oom_group_dir_slug(prompt_texts)
        prev_dir = out_dir / "previews" / str(svg_id) / f"oom_merged_{slug}"
        prev_dir.mkdir(parents=True, exist_ok=True)
        write_png_once(prev_dir / "gt_merged_mask.png", mask_bool_to_u8_png(gt_r))
        write_png_once(
            prev_dir / "gt_merged_overlay.png",
            overlay_bool_mask(rgb_work, gt_r, color=(64, 220, 64), alpha=0.45),
        )

        gt_scores = leaf_scores_from_binary_mask(leaf_cache, merged)
        out_merged, _ = prune_from_scores(leaf_cache, gt_scores, _GT_PRUNE_SCORE_THRESHOLD)
        merged_ext_svg = prev_dir / "gt_merged_extracted.svg"
        merged_ext_png = prev_dir / "gt_merged_extracted.png"
        write_svg_once(merged_ext_svg, out_merged)
        _write_extracted_svg_raster_png_once(merged_ext_svg, merged_ext_png, settings.raster_max_side)

        ann_ids = sorted({r.annotation_id for r in members})
        meta = {
            "kind": "one_visible_of_many_gt_merged",
            "svg_id": svg_id,
            "annotation_kind": OOM_KIND,
            "prompt_texts": list(prompt_texts),
            "annotation_ids": ann_ids,
            "n_members": len(members),
            "gt_leaf_nodes_ones_merged": int(sum(merged)),
            "previews": {
                "gt_merged_mask": "gt_merged_mask.png",
                "gt_merged_overlay": "gt_merged_overlay.png",
                "gt_merged_extracted_svg": "gt_merged_extracted.svg",
                "gt_merged_extracted_png": "gt_merged_extracted.png",
                "mirrored_next_to_extracted_union": {
                    "gt_merged_group_mask": "gt_merged_group_mask.png",
                    "gt_merged_group_overlay": "gt_merged_group_overlay.png",
                    "gt_merged_group_extracted_svg": "gt_merged_group_extracted.svg",
                    "gt_merged_group_extracted_png": "gt_merged_group_extracted.png",
                    "relative_path_template": "previews/<svg_id>/ann_<annotation_id>/p<prompt_index>/<detector_id>/<segmenter_id>/",
                },
            },
        }
        (prev_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        mirror_oom_gt_into_segmenter_previews(
            out_dir=out_dir, svg_id=svg_id, members=members, prev_dir=prev_dir
        )
        n_written += 1
        LOG.info(
            "OOM merged GT svg_id=%s slug=%s annotation_ids=%s merged_leaf_ones=%d",
            svg_id,
            slug,
            ann_ids,
            meta["gt_leaf_nodes_ones_merged"],
        )
    return n_written


def warm_svg_and_raster_caches(
    *,
    svg_ids: list[int],
    id_to_path: dict[int, Path],
    out_dir: Path,
    settings: InferenceSettings,
) -> tuple[dict[int, tuple[Any, Any, Any, Any, int]], dict[int, tuple[bytes, tuple[int, int]]]]:
    """Заполнить кэши структуры SVG и растра (как в engine) для списка ``svg_id``."""
    svg_struct_cache: dict[int, tuple[Any, Any, Any, Any, int]] = {}
    rgb_raster_cache: dict[int, tuple[bytes, tuple[int, int]]] = {}
    out_dir = Path(out_dir)
    raster_dir = out_dir / "cache" / "raster"
    raster_dir.mkdir(parents=True, exist_ok=True)

    for svg_id in sorted(set(svg_ids)):
        svg_path = id_to_path.get(svg_id)
        if svg_path is None or not svg_path.is_file():
            LOG.warning("warm caches skip svg_id=%s (missing svg path)", svg_id)
            continue
        root, pmap, leaves, _ = load_tagged_svg(svg_path)
        n_parsed = node_count_from_root(root)
        svg_struct_cache[svg_id] = (svg_path, root, pmap, leaves, n_parsed)

        svg_bytes = svg_path.read_bytes()
        hkey = sha256_hex(svg_bytes)
        rp = raster_dir / f"{hkey}_m{settings.raster_max_side}.png"
        if rp.is_file():
            png_bytes = rp.read_bytes()
            im = Image.open(io.BytesIO(png_bytes))
            wh = im.size
        else:
            png_bytes, wh = rasterize_svg_path_max_side(svg_path, settings.raster_max_side)
            rp.write_bytes(png_bytes)
        rgb_raster_cache[svg_id] = (png_bytes, (int(wh[0]), int(wh[1])))

    return svg_struct_cache, rgb_raster_cache


def write_oom_gt_merged_all_groups(
    *,
    rows: list[DatasetAnnotationRow],
    out_dir: Path,
    id_to_path: dict[int, Path],
    settings: InferenceSettings,
) -> int:
    """
    Записать merged GT для **всех** групп ``one_visible_of_many`` в ``rows`` (без инференса).
    Требуются только датасет, манифест и существующий ``out_dir`` (кэш растра подхватится).
    """
    groups = group_rows_one_visible_of_many(rows)
    if not groups:
        LOG.info("OOM merge-all: no one_visible_of_many rows after grouping")
        return 0
    svg_ids = sorted({key[0] for key in groups.keys()})
    LOG.info("OOM merge-all: %d groups across %d svg_ids — warming caches …", len(groups), len(svg_ids))
    svg_struct_cache, rgb_raster_cache = warm_svg_and_raster_caches(
        svg_ids=svg_ids,
        id_to_path=id_to_path,
        out_dir=out_dir,
        settings=settings,
    )
    return write_oom_gt_merged_previews(
        rows=rows,
        out_dir=out_dir,
        svg_struct_cache=svg_struct_cache,
        rgb_raster_cache=rgb_raster_cache,
        settings=settings,
    )
