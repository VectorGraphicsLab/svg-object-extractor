from __future__ import annotations

import io
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from svg_gt_baseline.dataset_io import filter_rows, load_dataset_rows, load_manifest_paths, smoke_svg_ids
from svg_gt_baseline.disk_cache import (
    dino_cache_key_parts,
    read_or_run_dino_json,
    sha256_hex,
)
from svg_gt_baseline.masks import leaf_scores_from_binary_mask, node_count_from_root, pred_mask_from_leaf_ids
from svg_gt_baseline.svg_size import rasterize_svg_path_max_side
from svg_gt_baseline.types import DatasetAnnotationRow, InferenceSettings
from svg_gt_baseline.viz import (
    draw_detection_boxes,
    leaf_union_from_seg_indices,
    mask_bool_to_u8_png,
    overlay_bool_mask,
    write_png_once,
    write_preview_meta,
    write_svg_once,
)
from svg_segmentator.backends.registry import build_detector, build_segmenter
from svg_segmentator.backends.types import DetectionResult, RunContext, SegmentationResult
from svg_segmentator.extract import build_leaf_mask_cache, load_tagged_svg, prune_from_scores, score_leaf_mask_cache
from svg_segmentator.grounding import query_to_candidate_labels
from svg_segmentator.render import png_bytes_to_rgba

LOG = logging.getLogger("svg_gt_baseline")

# Threshold for turning binary GT leaf labels into prune scores (1.0 vs 0.0).
_GT_PRUNE_SCORE_THRESHOLD = 0.5


def _prompt_snippet(text: str, max_len: int = 80) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _write_svg_raster_png_once(svg_path: Path, png_path: Path, raster_max_side: int) -> None:
    """Cairo raster of SVG on disk (same ``raster_max_side`` as previews); skip PNG if exists."""
    if png_path.is_file():
        return
    png_b, _ = rasterize_svg_path_max_side(svg_path, raster_max_side)
    write_png_once(png_path, Image.open(io.BytesIO(png_b)).convert("RGB"))


def set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _record_key(
    annotation_id: int,
    prompt_index: int,
    detector_id: str,
    segmenter_id: str,
    dino_box_index: int | None,
) -> tuple[Any, ...]:
    return (annotation_id, prompt_index, detector_id, segmenter_id, dino_box_index)


def _load_done_keys(ndjson_path: Path) -> set[tuple[Any, ...]]:
    done: set[tuple[Any, ...]] = set()
    if not ndjson_path.is_file():
        return done
    with ndjson_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            done.add(
                _record_key(
                    int(d["annotation_id"]),
                    int(d["prompt_index"]),
                    str(d["detector_id"]),
                    str(d["segmenter_id"]),
                    int(d["dino_box_index"]) if d.get("dino_box_index") is not None else None,
                )
            )
    return done


def _write_record(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def run_inference_job(
    *,
    dataset_jsonl: Path,
    manifest_json: Path,
    out_dir: Path,
    settings: InferenceSettings,
    smoke: bool,
    smoke_n: int = 3,
    cuda_device: int = 0,
    write_oom_gt_merged: bool = True,
    only_svg_ids: frozenset[int] | None = None,
) -> None:
    """
    Run baseline inference; append NDJSON to ``out_dir/predictions.ndjson``.

    Intermediate caches under ``out_dir/cache/``.
    """
    set_global_seed(settings.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset_jsonl": str(dataset_jsonl),
                "manifest_json": str(manifest_json),
                "smoke": smoke,
                "smoke_n": smoke_n,
                "cuda_device": cuda_device,
                "write_oom_gt_merged": write_oom_gt_merged,
                "only_svg_ids": sorted(only_svg_ids) if only_svg_ids is not None else None,
                "settings": {
                    "raster_max_side": settings.raster_max_side,
                    "box_threshold": settings.box_threshold,
                    "text_threshold": settings.text_threshold,
                    "score_threshold": settings.score_threshold,
                    "seed": settings.seed,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cache_dir = out_dir / "cache"
    raster_dir = cache_dir / "raster"
    dino_dir = cache_dir / "dino"
    raster_dir.mkdir(parents=True, exist_ok=True)
    dino_dir.mkdir(parents=True, exist_ok=True)

    ndjson_path = out_dir / "predictions.ndjson"
    skips_path = out_dir / "manifest_skips.jsonl"
    done = _load_done_keys(ndjson_path)

    rows_all = load_dataset_rows(dataset_jsonl)
    rows = filter_rows(rows_all)
    if smoke:
        allowed = smoke_svg_ids(rows, smoke_n)
        rows = [r for r in rows if r.svg_id in allowed]
        LOG.info("Smoke mode: %d svg_ids -> %d rows", len(allowed), len(rows))

    if only_svg_ids is not None:
        rows = [r for r in rows if r.svg_id in only_svg_ids]
        LOG.info("only_svg_ids: keep %d rows (%d ids requested)", len(rows), len(only_svg_ids))

    id_to_path = load_manifest_paths(manifest_json)

    svg_struct_cache: dict[int, tuple[Path, Any, Any, Any, int]] = {}
    rgb_raster_cache: dict[int, tuple[bytes, tuple[int, int]]] = {}

    detectors: dict[str, Any] = {}
    segmenters: dict[str, Any] = {}

    def get_detector(det_id: str):
        if det_id not in detectors:
            LOG.info("Loading detector backend_id=%s …", det_id)
            detectors[det_id] = build_detector(det_id)
            LOG.info("Detector ready backend_id=%s", det_id)
        return detectors[det_id]

    def get_segmenter(seg_id: str):
        if seg_id not in segmenters:
            LOG.info("Loading segmenter backend_id=%s …", seg_id)
            segmenters[seg_id] = build_segmenter(seg_id)
            LOG.info("Segmenter ready backend_id=%s", seg_id)
        return segmenters[seg_id]

    det_ids = ("gdino-tiny", "gdino-base")
    seg_ids = ("box-fill", "sam-vit-base")

    memo_seg: dict[
        tuple[int, str, str, str],
        tuple[list[int], list[list[int]], SegmentationResult],
    ] = {}

    svg_preview_base: set[int] = set()

    n_rows_ok = 0
    n_predictions_written = 0
    n_predictions_skip_done = 0
    n_dino_cache_hit = 0
    n_dino_cache_miss = 0
    n_seg_memo_hit = 0
    n_seg_memo_miss = 0

    LOG.info(
        "Job start out_dir=%s device=%s rows=%d resume_done_keys=%d smoke=%s",
        out_dir,
        device,
        len(rows),
        len(done),
        smoke,
    )

    for row_i, row in enumerate(rows, start=1):
        svg_path = id_to_path.get(row.svg_id)
        if svg_path is None or not svg_path.is_file():
            msg = f"SKIP missing svg file svg_id={row.svg_id} path={svg_path}"
            LOG.warning(msg)
            _write_record(skips_path, {"reason": "missing_svg", "svg_id": row.svg_id, "annotation_id": row.annotation_id})
            continue

        if row.svg_id not in svg_struct_cache:
            root, pmap, leaves, _ = load_tagged_svg(svg_path)
            n = node_count_from_root(root)
            svg_struct_cache[row.svg_id] = (svg_path, root, pmap, leaves, n)

        _path, root, pmap, leaves, n_parsed = svg_struct_cache[row.svg_id]
        if n_parsed != row.node_count:
            msg = (
                f"SKIP node_count mismatch svg_id={row.svg_id} annotation_id={row.annotation_id} "
                f"dataset={row.node_count} parsed={n_parsed} file={svg_path}"
            )
            LOG.warning(msg)
            print(msg, file=sys.stderr)
            _write_record(
                skips_path,
                {
                    "reason": "node_count_mismatch",
                    "svg_id": row.svg_id,
                    "annotation_id": row.annotation_id,
                    "dataset_node_count": row.node_count,
                    "parsed_node_count": n_parsed,
                    "object_key": row.object_key,
                },
            )
            continue

        svg_bytes = svg_path.read_bytes()
        svg_bytes_hash = sha256_hex(svg_bytes)

        raster_reused = row.svg_id in rgb_raster_cache
        if row.svg_id not in rgb_raster_cache:
            hkey = svg_bytes_hash
            rp = raster_dir / f"{hkey}_m{settings.raster_max_side}.png"
            if rp.is_file():
                png_bytes = rp.read_bytes()
                im = Image.open(io.BytesIO(png_bytes))
                wh = im.size
                LOG.info("Raster cache hit svg_id=%s file=%s", row.svg_id, rp.name)
            else:
                LOG.info(
                    "Rasterizing svg_id=%s max_side=%d (Cairo) …",
                    row.svg_id,
                    settings.raster_max_side,
                )
                png_bytes, wh = rasterize_svg_path_max_side(svg_path, settings.raster_max_side)
                rp.write_bytes(png_bytes)
                LOG.info("Raster ok svg_id=%s -> %s (%dx%d)", row.svg_id, rp.name, wh[0], wh[1])
            rgb_raster_cache[row.svg_id] = (png_bytes, wh)

        png_bytes, (rw, rh) = rgb_raster_cache[row.svg_id]
        if raster_reused:
            LOG.info("Raster reuse svg_id=%s (%dx%d)", row.svg_id, rw, rh)
        rgba = png_bytes_to_rgba(png_bytes)
        rgb_work = rgba.convert("RGB")
        output_wh = (rw, rh)

        n_rows_ok += 1
        LOG.info(
            "Row %d/%d svg_id=%s annotation_id=%s prompts=%d work_raster=%dx%d",
            row_i,
            len(rows),
            row.svg_id,
            row.annotation_id,
            len(row.prompt_texts),
            rw,
            rh,
        )

        leaf_cache = build_leaf_mask_cache(
            root,
            pmap,
            leaves,
            settings.leaf_render_dpi,
            rgb_work,
            None,
            output_wh=output_wh,
        )

        svg_prev_dir = out_dir / "previews" / str(row.svg_id)
        svg_prev_dir.mkdir(parents=True, exist_ok=True)
        ann_prev_dir = svg_prev_dir / f"ann_{row.annotation_id}"
        if row.svg_id not in svg_preview_base:
            shutil.copy2(svg_path, svg_prev_dir / "source.svg")
            write_png_once(svg_prev_dir / "raster.png", rgb_work)
            write_preview_meta(
                svg_prev_dir / "meta.json",
                {
                    "kind": "svg_raster",
                    "svg_id": row.svg_id,
                    "source_svg": str(svg_path),
                    "previews": {"source": "source.svg", "raster": "raster.png", "annotations": "ann_<annotation_id>/"},
                },
            )
            svg_preview_base.add(row.svg_id)

        gt_mask_path = ann_prev_dir / "gt_mask.png"
        if not gt_mask_path.is_file():
            gt_r = leaf_union_from_seg_indices(leaf_cache, row.binary_mask)
            write_png_once(gt_mask_path, mask_bool_to_u8_png(gt_r))
            write_png_once(
                ann_prev_dir / "gt_overlay.png",
                overlay_bool_mask(rgb_work, gt_r, color=(64, 220, 64), alpha=0.45),
            )

        write_preview_meta(
            ann_prev_dir / "meta.json",
            {
                "kind": "annotation",
                "svg_id": row.svg_id,
                "annotation_id": row.annotation_id,
                "annotation_kind": row.annotation_kind,
                "object_key": row.object_key,
                "original_filename": row.original_filename,
                "node_count": row.node_count,
                "review_decision": row.review_decision,
                "prompt_texts": row.prompt_texts,
                "gt_leaf_nodes_ones": int(sum(row.binary_mask)),
                "raster_width": rw,
                "raster_height": rh,
                "raster_max_side": settings.raster_max_side,
                "leaf_render_dpi": settings.leaf_render_dpi,
                "score_threshold": settings.score_threshold,
                "previews": {
                    "gt_mask": "gt_mask.png",
                    "gt_overlay": "gt_overlay.png",
                    "gt_extracted_svg": "gt_extracted.svg",
                    "gt_extracted_png": "gt_extracted.png",
                    "subfolders": "p<prompt_index>/<detector_id>/… — один промпт + детектор; внутри подпапки сегментаторов.",
                },
            },
        )

        gt_scores = leaf_scores_from_binary_mask(leaf_cache, row.binary_mask)
        out_gt, _ = prune_from_scores(leaf_cache, gt_scores, _GT_PRUNE_SCORE_THRESHOLD)
        gt_ext_svg = ann_prev_dir / "gt_extracted.svg"
        write_svg_once(gt_ext_svg, out_gt)
        _write_svg_raster_png_once(gt_ext_svg, ann_prev_dir / "gt_extracted.png", settings.raster_max_side)

        for prompt_index, prompt_text in enumerate(row.prompt_texts):
            for detector_id in det_ids:
                dkey = dino_cache_key_parts(
                    svg_hash=svg_bytes_hash,
                    detector_id=detector_id,
                    box_threshold=settings.box_threshold,
                    text_threshold=settings.text_threshold,
                    raster_max_side=settings.raster_max_side,
                    prompt_text=prompt_text,
                )
                dpath = dino_dir / f"{dkey}.json"

                detector = get_detector(detector_id)

                def run_dino() -> DetectionResult:
                    labels = query_to_candidate_labels(prompt_text)
                    ctx = RunContext(
                        image_rgb=rgb_work,
                        device=device,
                        query=prompt_text,
                        text_labels=labels,
                    )
                    return detector.detect(
                        ctx,
                        box_threshold=settings.box_threshold,
                        text_threshold=settings.text_threshold,
                    )

                dino_cached = dpath.is_file()
                det_result = read_or_run_dino_json(dpath, run_dino)
                if dino_cached:
                    n_dino_cache_hit += 1
                else:
                    n_dino_cache_miss += 1
                LOG.info(
                    "DINO %s svg_id=%s ann=%s p=%s det=%s boxes=%d prompt=%s",
                    "cache_hit" if dino_cached else "cache_miss",
                    row.svg_id,
                    row.annotation_id,
                    prompt_index,
                    detector_id,
                    len(det_result.boxes),
                    _prompt_snippet(prompt_text),
                )

                prompt_det_dir = ann_prev_dir / f"p{prompt_index}" / detector_id
                write_png_once(prompt_det_dir / "dino_boxes.png", draw_detection_boxes(rgb_work, det_result))

                labels_for_prompt = query_to_candidate_labels(prompt_text)
                write_preview_meta(
                    prompt_det_dir / "meta.json",
                    {
                        "kind": "prompt_detector",
                        "svg_id": row.svg_id,
                        "annotation_id": row.annotation_id,
                        "prompt_index": prompt_index,
                        "prompt_text": prompt_text,
                        "text_labels": labels_for_prompt,
                        "detector_id": detector_id,
                        "box_threshold": settings.box_threshold,
                        "text_threshold": settings.text_threshold,
                        "raster_width": rw,
                        "raster_height": rh,
                        "raster_max_side": settings.raster_max_side,
                        "dino_cache_json": str(dpath.relative_to(out_dir)),
                        "detections": {
                            "n_boxes": len(det_result.boxes),
                            "boxes": _serialize_boxes(det_result),
                        },
                        "previews": {
                            "dino_boxes": "dino_boxes.png",
                            "segmenters": "Подпапки по имени segmenter_id (box-fill, sam-vit-base, …).",
                        },
                    },
                )

                for segmenter_id in seg_ids:
                    # Segmenter memo: один вывод на (svg_id, prompt_text, det, seg) без annotation_id —
                    # для одной картинки и одного промпта предикт совпадает между строками датасета (OOM и др.).
                    mk = (row.svg_id, prompt_text, detector_id, segmenter_id)
                    seg_memo_cached = mk in memo_seg
                    if mk not in memo_seg:
                        n_seg_memo_miss += 1
                        segmenter = get_segmenter(segmenter_id)
                        labels = query_to_candidate_labels(prompt_text)
                        ctx = RunContext(
                            image_rgb=rgb_work,
                            device=device,
                            query=prompt_text,
                            text_labels=labels,
                        )
                        LOG.info(
                            "Segment forward svg_id=%s ann=%s p=%s det=%s seg=%s …",
                            row.svg_id,
                            row.annotation_id,
                            prompt_index,
                            detector_id,
                            segmenter_id,
                        )
                        seg = segmenter.segment(ctx, det_result)
                        n_boxes_run = len(det_result.boxes)
                        if n_boxes_run == 0:
                            z = [0] * row.node_count
                            memo_seg[mk] = (z, [], seg)
                        else:
                            scores_c = score_leaf_mask_cache(leaf_cache, seg.combined_mask)
                            _, lids_c = prune_from_scores(leaf_cache, scores_c, settings.score_threshold)
                            pred_c = pred_mask_from_leaf_ids(root, lids_c)
                            per_box_masks: list[list[int]] = []
                            for obj_mask in seg.masks:
                                scores_b = score_leaf_mask_cache(leaf_cache, obj_mask)
                                _, lids_b = prune_from_scores(leaf_cache, scores_b, settings.score_threshold)
                                per_box_masks.append(pred_mask_from_leaf_ids(root, lids_b))
                            memo_seg[mk] = (pred_c, per_box_masks, seg)
                    else:
                        n_seg_memo_hit += 1

                    pred_combined, per_box_preds, seg = memo_seg[mk]

                    seg_prev_dir = prompt_det_dir / segmenter_id
                    write_png_once(seg_prev_dir / "mask_union.png", mask_bool_to_u8_png(seg.combined_mask))
                    write_png_once(
                        seg_prev_dir / "overlay_union.png",
                        overlay_bool_mask(rgb_work, seg.combined_mask, color=(255, 64, 64), alpha=0.45),
                    )
                    pred_leaf_r = leaf_union_from_seg_indices(leaf_cache, pred_combined)
                    write_png_once(seg_prev_dir / "pred_leaf_mask.png", mask_bool_to_u8_png(pred_leaf_r))
                    write_png_once(
                        seg_prev_dir / "pred_leaves_overlay.png",
                        overlay_bool_mask(rgb_work, pred_leaf_r, color=(80, 200, 255), alpha=0.5),
                    )

                    scores_union = score_leaf_mask_cache(leaf_cache, seg.combined_mask)
                    out_union, _ = prune_from_scores(leaf_cache, scores_union, settings.score_threshold)
                    union_svg = seg_prev_dir / "extracted_union.svg"
                    write_svg_once(union_svg, out_union)
                    _write_svg_raster_png_once(union_svg, seg_prev_dir / "extracted_union.png", settings.raster_max_side)

                    n_boxes = len(det_result.boxes)
                    all_boxes = _serialize_boxes(det_result)

                    per_box_rows: list[dict[str, Any]] = []
                    for bi, obj_mask in enumerate(seg.masks):
                        scores_b = score_leaf_mask_cache(leaf_cache, obj_mask)
                        out_box, _ = prune_from_scores(leaf_cache, scores_b, settings.score_threshold)
                        box_svg = seg_prev_dir / f"extracted_box{bi}.svg"
                        write_svg_once(box_svg, out_box)
                        _write_svg_raster_png_once(
                            box_svg, seg_prev_dir / f"extracted_box{bi}.png", settings.raster_max_side
                        )
                        write_png_once(seg_prev_dir / f"box{bi}_mask.png", mask_bool_to_u8_png(obj_mask))
                        write_png_once(
                            seg_prev_dir / f"box{bi}_overlay.png",
                            overlay_bool_mask(rgb_work, obj_mask, color=(255, 140, 40), alpha=0.45),
                        )
                        m = np.asarray(obj_mask, dtype=bool)
                        pm = per_box_preds[bi] if bi < len(per_box_preds) else []
                        per_box_rows.append(
                            {
                                "box_index": bi,
                                "segmentation_mask_true_pixels": int(m.sum()),
                                "pred_leaf_nodes_ones_after_prune": int(sum(pm)),
                            }
                        )

                    write_preview_meta(
                        seg_prev_dir / "meta.json",
                        {
                            "kind": "segmenter_run",
                            "svg_id": row.svg_id,
                            "annotation_id": row.annotation_id,
                            "prompt_index": prompt_index,
                            "prompt_text": prompt_text,
                            "detector_id": detector_id,
                            "segmenter_id": segmenter_id,
                            "text_labels": labels_for_prompt,
                            "box_threshold": settings.box_threshold,
                            "text_threshold": settings.text_threshold,
                            "score_threshold": settings.score_threshold,
                            "raster_width": rw,
                            "raster_height": rh,
                            "n_boxes": n_boxes,
                            "combined_mask_true_pixels": int(np.asarray(seg.combined_mask, dtype=bool).sum()),
                            "pred_leaf_nodes_ones_combined_after_prune": int(sum(pred_combined)),
                            "gt_leaf_nodes_ones": int(sum(row.binary_mask)),
                            "per_box": per_box_rows,
                            "previews": {
                                "mask_union": "mask_union.png",
                                "overlay_union": "overlay_union.png",
                                "pred_leaf_mask": "pred_leaf_mask.png",
                                "pred_leaves_overlay": "pred_leaves_overlay.png",
                                "extracted_union_svg": "extracted_union.svg",
                                "extracted_union_png": "extracted_union.png",
                                "extracted_per_box_svg": "extracted_box{i}.svg",
                                "extracted_per_box_png": "extracted_box{i}.png",
                                "per_box_mask": "box{i}_mask.png",
                                "per_box_overlay": "box{i}_overlay.png",
                            },
                        },
                    )

                    ndjson_new = 0
                    ndjson_skip = 0
                    if n_boxes == 0:
                        rk = _record_key(row.annotation_id, prompt_index, detector_id, segmenter_id, None)
                        if rk in done:
                            ndjson_skip += 1
                        else:
                            rec = _build_record(
                                row=row,
                                prompt_index=prompt_index,
                                prompt_text=prompt_text,
                                detector_id=detector_id,
                                segmenter_id=segmenter_id,
                                dino_box_index=None,
                                n_boxes=0,
                                boxes_json=all_boxes,
                                pred_mask=pred_combined,
                                pred_mask_combined=pred_combined,
                                zero_boxes=True,
                                raster_wh=output_wh,
                                settings=settings,
                            )
                            _write_record(ndjson_path, rec)
                            done.add(rk)
                            ndjson_new += 1
                    else:
                        for bi in range(n_boxes):
                            rk = _record_key(row.annotation_id, prompt_index, detector_id, segmenter_id, bi)
                            if rk in done:
                                ndjson_skip += 1
                                continue
                            rec = _build_record(
                                row=row,
                                prompt_index=prompt_index,
                                prompt_text=prompt_text,
                                detector_id=detector_id,
                                segmenter_id=segmenter_id,
                                dino_box_index=bi,
                                n_boxes=n_boxes,
                                boxes_json=all_boxes,
                                pred_mask=per_box_preds[bi],
                                pred_mask_combined=pred_combined,
                                zero_boxes=False,
                                raster_wh=output_wh,
                                settings=settings,
                            )
                            _write_record(ndjson_path, rec)
                            done.add(rk)
                            ndjson_new += 1

                    n_predictions_written += ndjson_new
                    n_predictions_skip_done += ndjson_skip
                    union_px = int(np.asarray(seg.combined_mask, dtype=bool).sum())
                    LOG.info(
                        "Step svg_id=%s ann=%s p=%s det=%s seg=%s seg_memo=%s dino_boxes=%d "
                        "union_px=%d pred_nodes=%d ndjson_new=%d ndjson_skip_resume=%d",
                        row.svg_id,
                        row.annotation_id,
                        prompt_index,
                        detector_id,
                        segmenter_id,
                        "hit" if seg_memo_cached else "miss",
                        n_boxes,
                        union_px,
                        int(sum(pred_combined)),
                        ndjson_new,
                        ndjson_skip,
                    )

    if write_oom_gt_merged:
        from svg_gt_baseline.oom_gt_merge import write_oom_gt_merged_previews

        n_oom = write_oom_gt_merged_previews(
            rows=rows,
            out_dir=out_dir,
            svg_struct_cache=svg_struct_cache,
            rgb_raster_cache=rgb_raster_cache,
            settings=settings,
        )
        LOG.info("OOM merged GT preview groups written=%d", n_oom)

    ndjson_lines = 0
    if ndjson_path.is_file():
        ndjson_lines = sum(1 for line in ndjson_path.open(encoding="utf-8") if line.strip())
    LOG.info(
        "Job done rows_processed=%d predictions_written=%d predictions_skip_resume=%d "
        "dino_cache_hit=%d dino_cache_miss=%d seg_memo_hit=%d seg_memo_miss=%d predictions.ndjson_lines=%d path=%s",
        n_rows_ok,
        n_predictions_written,
        n_predictions_skip_done,
        n_dino_cache_hit,
        n_dino_cache_miss,
        n_seg_memo_hit,
        n_seg_memo_miss,
        ndjson_lines,
        ndjson_path,
    )


def _serialize_boxes(det: DetectionResult) -> list[dict[str, Any]]:
    return [
        {
            "x0": b.x0,
            "y0": b.y0,
            "x1": b.x1,
            "y1": b.y1,
            "score": b.score,
            "label": b.label,
        }
        for b in det.boxes
    ]


def _build_record(
    *,
    row: DatasetAnnotationRow,
    prompt_index: int,
    prompt_text: str,
    detector_id: str,
    segmenter_id: str,
    dino_box_index: int | None,
    n_boxes: int,
    boxes_json: list[dict[str, Any]],
    pred_mask: list[int],
    pred_mask_combined: list[int],
    zero_boxes: bool,
    raster_wh: tuple[int, int],
    settings: InferenceSettings,
) -> dict[str, Any]:
    return {
        "svg_id": row.svg_id,
        "annotation_id": row.annotation_id,
        "annotation_kind": row.annotation_kind,
        "object_key": row.object_key,
        "original_filename": row.original_filename,
        "node_count": row.node_count,
        "binary_mask": row.binary_mask,
        "prompt_index": prompt_index,
        "prompt_text": prompt_text,
        "detector_id": detector_id,
        "segmenter_id": segmenter_id,
        "dino_box_index": dino_box_index,
        "n_boxes": n_boxes,
        "boxes": boxes_json,
        "zero_boxes": zero_boxes,
        "pred_mask": pred_mask,
        "pred_mask_combined": pred_mask_combined,
        "raster_max_side": settings.raster_max_side,
        "raster_width": raster_wh[0],
        "raster_height": raster_wh[1],
        "box_threshold": settings.box_threshold,
        "text_threshold": settings.text_threshold,
        "score_threshold": settings.score_threshold,
    }
