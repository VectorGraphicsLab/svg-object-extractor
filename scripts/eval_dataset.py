"""Batch-evaluate the extraction pipeline on annotations.csv (one run per SVG tag)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from svg_segmentator.repo_env import load_repo_dotenv

load_repo_dotenv()

from svg_segmentator.backends.registry import list_detector_ids, list_segmenter_ids
from svg_segmentator.eval_pipeline import EvalPipeline
from svg_segmentator.pipeline import PipelineConfig, PipelineRunSummary


def _safe_segment(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _run_dir_name(stem: str, tag: str, max_tag_len: int = 80) -> str:
    seg = _safe_segment(tag)
    if len(seg) > max_tag_len:
        seg = seg[:max_tag_len]
    return f"{stem}__{seg}"


def _read_resume_summary(svg_stem: str, out_dir: Path) -> PipelineRunSummary | None:
    clip_path = out_dir / f"{svg_stem}_clip_scores.csv"
    if not clip_path.is_file():
        return None
    clip_rows: list[tuple[str, float]] = []
    with clip_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "target" not in row or "clip_score" not in row:
                return None
            clip_rows.append((row["target"], float(row["clip_score"])))
    if not clip_rows:
        return None
    raster = out_dir / f"{svg_stem}_raster.png"
    if not raster.is_file():
        return None
    with Image.open(raster) as im:
        w, h = im.size
    n_obj = sum(1 for t, _ in clip_rows if t != "combined")
    return PipelineRunSummary(
        n_boxes=n_obj,
        image_width=w,
        image_height=h,
        clip_scores=tuple(clip_rows),
    )


def _clip_combined(summary: PipelineRunSummary) -> float | None:
    for name, score in summary.clip_scores:
        if name == "combined":
            return score
    return None


def _load_tags(cell: str) -> list[str]:
    cell = cell.strip()
    if not cell:
        return []
    data = json.loads(cell)
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for x in data:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


RESULT_FIELDNAMES = [
    "dataset_rel_path",
    "tag",
    "detector",
    "segmenter",
    "detector_model_id",
    "segmenter_model_id",
    "dpi",
    "box_threshold",
    "text_threshold",
    "score_threshold",
    "max_leaves",
    "clip_model_id",
    "image_width",
    "image_height",
    "n_boxes",
    "clip_combined",
    "clip_scores_json",
    "run_out_dir",
    "status",
    "error",
]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run pipeline once per (SVG, tag) from annotations.csv; write one wide results CSV."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=_REPO / "data" / "dataset_annotations",
        help="Directory with annotations.csv and SVG paths as in the CSV",
    )
    p.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="CSV path (default: <dataset-dir>/annotations.csv)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=_REPO / "output" / "dataset_eval",
        help="Root directory for per-run outputs and results CSV",
    )
    p.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Path for aggregated CSV (default: <out-root>/eval_results__<detector>__<segmenter>.csv)",
    )
    p.add_argument("--detector", type=str, default="gdino-tiny")
    p.add_argument("--segmenter", type=str, default="box-fill")
    p.add_argument("--gdino-model", type=str, default="", help="Override HF model id for Grounding DINO detectors")
    p.add_argument("--sam-model", type=str, default="", help="Override HF model id for SAM segmenter")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--dpi", type=float, default=96.0)
    p.add_argument("--score-threshold", type=float, default=0.8)
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--max-leaves", type=int, default=0)
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--candidate-render-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="Max number of (svg,tag) runs (0 = no limit)")
    p.add_argument(
        "--max-svgs",
        type=int,
        default=0,
        help="Stop after this many annotation rows with non-empty tags (0 = no limit). Each row is one source SVG.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If run output already has clip_scores.csv, reuse metrics and skip inference",
    )
    args = p.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    ann_path = args.annotations.resolve() if args.annotations else (dataset_dir / "annotations.csv")
    if not ann_path.is_file():
        print(f"Missing annotations: {ann_path}", file=sys.stderr)
        return 1

    if args.detector not in list_detector_ids():
        print(f"Unknown detector {args.detector!r}. Available: {', '.join(list_detector_ids())}", file=sys.stderr)
        return 2
    if args.segmenter not in list_segmenter_ids():
        print(f"Unknown segmenter {args.segmenter!r}. Available: {', '.join(list_segmenter_ids())}", file=sys.stderr)
        return 2

    out_root = args.out_root.resolve()
    combo_dir = out_root / f"{args.detector}__{args.segmenter}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    results_csv = (
        args.results_csv.resolve()
        if args.results_csv
        else (out_root / f"eval_results__{args.detector}__{args.segmenter}.csv")
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    max_leaves = args.max_leaves if args.max_leaves > 0 else None

    pipeline = EvalPipeline.from_registry(
        detector_id=args.detector,
        segmenter_id=args.segmenter,
        device=device,
        clip_model_id=args.clip_model,
        gdino_model_override=args.gdino_model or None,
        sam_model_override=args.sam_model or None,
    )
    det_model = getattr(pipeline.detector, "model_id", "") or ""
    seg_model = getattr(pipeline.segmenter, "model_id", "") or ""

    runs_done = 0
    rows_written = 0
    skipped_empty_tags = 0
    svgs_done = 0
    tag_runs_success = 0
    tag_runs_zero_boxes = 0
    tag_runs_positive_boxes = 0
    svgs_no_detection_all_tags = 0

    with results_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        out_f.flush()

        with ann_path.open(newline="", encoding="utf-8") as ann_f:
            reader = csv.DictReader(ann_f)
            if reader.fieldnames is None or "path" not in reader.fieldnames or "tags" not in reader.fieldnames:
                print("annotations.csv must have columns: path, tags", file=sys.stderr)
                return 3

            for row in reader:
                rel = (row.get("path") or "").strip()
                if not rel:
                    continue
                tags = _load_tags(row.get("tags") or "[]")
                if not tags:
                    skipped_empty_tags += 1
                    continue

                svg_path = (dataset_dir / rel).resolve()
                if not svg_path.is_file():
                    writer.writerow(
                        {
                            "dataset_rel_path": rel,
                            "tag": "",
                            "detector": args.detector,
                            "segmenter": args.segmenter,
                            "detector_model_id": det_model,
                            "segmenter_model_id": seg_model,
                            "dpi": args.dpi,
                            "box_threshold": args.box_threshold,
                            "text_threshold": args.text_threshold,
                            "score_threshold": args.score_threshold,
                            "max_leaves": args.max_leaves,
                            "clip_model_id": args.clip_model,
                            "image_width": "",
                            "image_height": "",
                            "n_boxes": "",
                            "clip_combined": "",
                            "clip_scores_json": "",
                            "run_out_dir": "",
                            "status": "missing_svg",
                            "error": f"not a file: {svg_path}",
                        }
                    )
                    rows_written += 1
                    out_f.flush()
                    svgs_done += 1
                    if args.max_svgs and svgs_done >= args.max_svgs:
                        break
                    continue

                stem = svg_path.stem
                svg_any_success = False
                svg_any_positive_boxes = False
                hit_limit_mid_svg = False
                for tag in tags:
                    if args.limit and runs_done >= args.limit:
                        hit_limit_mid_svg = True
                        break

                    run_dir = combo_dir / _run_dir_name(stem, tag)
                    base_row = {
                        "dataset_rel_path": rel,
                        "tag": tag,
                        "detector": args.detector,
                        "segmenter": args.segmenter,
                        "detector_model_id": det_model,
                        "segmenter_model_id": seg_model,
                        "dpi": args.dpi,
                        "box_threshold": args.box_threshold,
                        "text_threshold": args.text_threshold,
                        "score_threshold": args.score_threshold,
                        "max_leaves": args.max_leaves,
                        "clip_model_id": args.clip_model,
                    }

                    summary: PipelineRunSummary | None = None
                    err = ""
                    status = "ok"

                    if args.resume:
                        summary = _read_resume_summary(stem, run_dir)

                    try:
                        if summary is None:
                            cand = None
                            if args.candidate_render_dir is not None:
                                cand = run_dir / args.candidate_render_dir
                            summary = pipeline.run(
                                svg_path=svg_path,
                                out_dir=run_dir,
                                config=PipelineConfig(
                                    query=tag,
                                    dpi=args.dpi,
                                    score_threshold=args.score_threshold,
                                    box_threshold=args.box_threshold,
                                    text_threshold=args.text_threshold,
                                    max_leaves=max_leaves,
                                    clip_model_id=args.clip_model,
                                ),
                                candidate_render_dir=cand,
                            )
                    except Exception as ex:
                        status = "error"
                        err = f"{type(ex).__name__}: {ex}"
                        writer.writerow(
                            {
                                **base_row,
                                "image_width": "",
                                "image_height": "",
                                "n_boxes": "",
                                "clip_combined": "",
                                "clip_scores_json": "",
                                "run_out_dir": str(run_dir.relative_to(_REPO)),
                                "status": status,
                                "error": err,
                            }
                        )
                        rows_written += 1
                        out_f.flush()
                        runs_done += 1
                        continue

                    svg_any_success = True
                    if summary.n_boxes > 0:
                        svg_any_positive_boxes = True
                    tag_runs_success += 1
                    if summary.n_boxes == 0:
                        tag_runs_zero_boxes += 1
                    else:
                        tag_runs_positive_boxes += 1

                    clip_json = json.dumps(
                        [{"target": t, "clip_score": s} for t, s in summary.clip_scores],
                        ensure_ascii=False,
                    )
                    writer.writerow(
                        {
                            **base_row,
                            "image_width": summary.image_width,
                            "image_height": summary.image_height,
                            "n_boxes": summary.n_boxes,
                            "clip_combined": _clip_combined(summary),
                            "clip_scores_json": clip_json,
                            "run_out_dir": str(run_dir.relative_to(_REPO)),
                            "status": status,
                            "error": "",
                        }
                    )
                    rows_written += 1
                    out_f.flush()
                    runs_done += 1

                if svg_any_success and not svg_any_positive_boxes and not hit_limit_mid_svg:
                    svgs_no_detection_all_tags += 1

                svgs_done += 1
                if args.max_svgs and svgs_done >= args.max_svgs:
                    break
                if args.limit and runs_done >= args.limit:
                    break

    summary_path = results_csv.parent / f"eval_summary__{args.detector}__{args.segmenter}.json"
    summary_payload = {
        "detector": args.detector,
        "segmenter": args.segmenter,
        "results_csv": str(results_csv.relative_to(_REPO)),
        "combo_output_dir": str(combo_dir.relative_to(_REPO)),
        "csv_rows_written": rows_written,
        "tag_runs_completed_ok": tag_runs_success,
        "tag_runs_with_zero_boxes": tag_runs_zero_boxes,
        "tag_runs_with_positive_boxes": tag_runs_positive_boxes,
        "annotation_rows_with_tags_processed": svgs_done,
        "annotation_rows_skipped_empty_tags": skipped_empty_tags,
        "svgs_no_detection_on_any_tag": svgs_no_detection_all_tags,
        "svgs_no_detection_note": (
            "Count of SVGs that had at least one successful tag run and zero detector boxes on every such run."
        ),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {rows_written} rows to {results_csv}")
    print(
        f"Runs completed (attempted): {runs_done}; annotation rows with tags processed: {svgs_done}; "
        f"skipped rows with empty tags: {skipped_empty_tags}"
    )
    print(
        f"Tag runs (ok): {tag_runs_success}; with zero boxes: {tag_runs_zero_boxes}; with ≥1 box: {tag_runs_positive_boxes}"
    )
    print(
        f"SVGs with no detection on any tag (all successful runs had 0 boxes): {svgs_no_detection_all_tags}"
    )
    print(f"Summary JSON: {summary_path}")
    print(f"Outputs under {combo_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
