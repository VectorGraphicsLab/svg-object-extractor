#!/usr/bin/env python3
"""Aggregate metrics from eval_dataset.py output CSVs (eval_results__*.csv)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pvariance, variance

_REPO = Path(__file__).resolve().parents[1]


def _rel_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _clip_values(rows: list[float]) -> tuple[float, float, float, float, float]:
    """Returns mean, median, sample_variance, population_variance, count."""
    if not rows:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    m = mean(rows)
    med = median(rows)
    n = len(rows)
    sv = variance(rows) if n > 1 else 0.0
    pv = pvariance(rows) if n > 0 else float("nan")
    return (m, med, sv, pv, n)


def _max_leaf_score(leaf_csv: Path) -> float | None:
    if not leaf_csv.is_file():
        return None
    mx = float("-inf")
    with leaf_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "score" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                mx = max(mx, float(row["score"]))
            except (KeyError, ValueError):
                continue
    return mx if mx > float("-inf") else None


def _extracted_has_ink_png(png_path: Path) -> bool | None:
    """True if any non-transparent pixel; False if all transparent; None if unreadable."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not png_path.is_file():
        return None
    try:
        with Image.open(png_path) as im:
            rgba = im.convert("RGBA")
            extrema = rgba.getextrema()
            if extrema is None or len(extrema) < 4:
                return None
            alpha_min, alpha_max = extrema[3]
            return alpha_max > 0
    except OSError:
        return None


def summarize_one(
    csv_path: Path,
    repo_root: Path,
    *,
    scan_prune_empty: bool,
    scan_extracted_alpha: bool,
    scan_clip_exclude_transparent_png: bool,
) -> dict:
    clips_all_ok: list[float] = []
    clips_boxes_gt0: list[float] = []
    clips_ok_extracted_visible: list[float] = []

    ok_runs = 0
    runs_n_boxes_0 = 0
    runs_ok_n_boxes_gt_0 = 0
    error_runs = 0
    missing_svg_runs = 0
    other_status: dict[str, int] = defaultdict(int)

    svg_seen: set[str] = set()
    svg_any_positive_boxes: set[str] = set()

    prune_empty = 0
    prune_nonempty = 0
    missing_leaf_csv = 0
    missing_stem_or_run_dir = 0
    alpha_empty = 0
    alpha_ink = 0
    alpha_unknown = 0

    clip_ok_png_transparent = 0
    clip_ok_png_missing_or_unreadable = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = (row.get("dataset_rel_path") or "").strip()
            st = (row.get("status") or "").strip()
            cc = (row.get("clip_combined") or "").strip()
            nb_raw = (row.get("n_boxes") or "").strip()

            if st == "missing_svg":
                missing_svg_runs += 1
                continue
            if st == "error":
                error_runs += 1
                continue
            if st != "ok":
                other_status[st or "?"] += 1
                continue

            ok_runs += 1
            if rel:
                svg_seen.add(rel)

            try:
                n_boxes = int(nb_raw)
            except ValueError:
                n_boxes = -1

            if n_boxes == 0:
                runs_n_boxes_0 += 1
            elif n_boxes > 0:
                runs_ok_n_boxes_gt_0 += 1
                if rel:
                    svg_any_positive_boxes.add(rel)

            if cc:
                v = float(cc)
                clips_all_ok.append(v)
                if n_boxes > 0:
                    clips_boxes_gt0.append(v)

            run_dir_raw = (row.get("run_out_dir") or "").strip()
            stem = Path(rel).stem if rel else ""
            run_dir = repo_root / run_dir_raw if run_dir_raw else repo_root

            if scan_clip_exclude_transparent_png and cc:
                if not stem or not run_dir_raw:
                    clip_ok_png_missing_or_unreadable += 1
                else:
                    ink = _extracted_has_ink_png(run_dir / f"{stem}_extracted.png")
                    if ink is True:
                        clips_ok_extracted_visible.append(float(cc))
                    elif ink is False:
                        clip_ok_png_transparent += 1
                    else:
                        clip_ok_png_missing_or_unreadable += 1

            if not scan_prune_empty and not scan_extracted_alpha:
                continue
            if n_boxes <= 0:
                continue

            if not stem or not run_dir_raw:
                missing_stem_or_run_dir += 1
                continue

            try:
                thresh = float(row.get("score_threshold", "0.8"))
            except ValueError:
                thresh = 0.8

            if scan_prune_empty:
                leaf_path = run_dir / f"{stem}_leaf_scores.csv"
                mx = _max_leaf_score(leaf_path)
                if mx is None:
                    missing_leaf_csv += 1
                elif mx < thresh:
                    prune_empty += 1
                else:
                    prune_nonempty += 1

            if scan_extracted_alpha:
                png_path = run_dir / f"{stem}_extracted.png"
                ink = _extracted_has_ink_png(png_path)
                if ink is True:
                    alpha_ink += 1
                elif ink is False:
                    alpha_empty += 1
                else:
                    alpha_unknown += 1

    svgs_never_positive = len(svg_seen - svg_any_positive_boxes)

    m1, med1, sv1, pv1, n1 = _clip_values(clips_all_ok)
    m2, med2, sv2, pv2, n2 = _clip_values(clips_boxes_gt0)
    m3, med3, sv3, pv3, n3 = _clip_values(clips_ok_extracted_visible)

    return {
        "results_csv": _rel_to_repo(csv_path.resolve(), repo_root),
        "ok_runs": ok_runs,
        "runs_n_boxes_0": runs_n_boxes_0,
        "runs_missing_svg": missing_svg_runs,
        "runs_error": error_runs,
        "runs_other_status": dict(other_status),
        "unique_svgs_in_ok_runs": len(svg_seen),
        "unique_svgs_never_n_boxes_gt_0": svgs_never_positive,
        "clip_combined_all_ok_runs": {
            "n": int(n1),
            "mean": m1,
            "median": med1,
            "sample_variance": sv1,
            "population_variance": pv1,
        },
        "clip_combined_n_boxes_gt_0": {
            "n": int(n2),
            "mean": m2,
            "median": med2,
            "sample_variance": sv2,
            "population_variance": pv2,
        },
        "clip_combined_ok_non_transparent_extracted_png": {
            "enabled": scan_clip_exclude_transparent_png,
            "note": (
                "CLIP stats only for status=ok rows with clip_combined set, where *_extracted.png "
                "has at least one non-transparent pixel (fully transparent runs excluded from mean)."
            ),
            "n": int(n3) if scan_clip_exclude_transparent_png else None,
            "mean": m3 if scan_clip_exclude_transparent_png else None,
            "median": med3 if scan_clip_exclude_transparent_png else None,
            "sample_variance": sv3 if scan_clip_exclude_transparent_png else None,
            "population_variance": pv3 if scan_clip_exclude_transparent_png else None,
            "ok_runs_with_clip_but_png_fully_transparent": clip_ok_png_transparent
            if scan_clip_exclude_transparent_png
            else None,
            "ok_runs_with_clip_but_png_missing_or_unreadable": clip_ok_png_missing_or_unreadable
            if scan_clip_exclude_transparent_png
            else None,
        },
        "prune_scan": {
            "enabled": scan_prune_empty,
            "runs_ok_with_n_boxes_gt_0": runs_ok_n_boxes_gt_0,
            "runs_no_leaf_kept_max_score_below_threshold": prune_empty if scan_prune_empty else None,
            "runs_at_least_one_leaf_kept": prune_nonempty if scan_prune_empty else None,
            "runs_missing_leaf_scores_csv": missing_leaf_csv if scan_prune_empty else None,
            "runs_missing_stem_or_run_out_dir": missing_stem_or_run_dir if scan_prune_empty else None,
        },
        "extracted_png_alpha_scan": {
            "enabled": scan_extracted_alpha,
            "runs_with_visible_pixels": alpha_ink,
            "runs_fully_transparent": alpha_empty,
            "runs_unreadable_or_missing": alpha_unknown,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Summarize eval_results CSVs: CLIP mean/variance, box counts, optional prune/alpha checks."
    )
    p.add_argument(
        "csvs",
        nargs="*",
        type=Path,
        help="Paths to eval_results*.csv (default: all output/dataset_eval_full_*/eval_results*.csv under repo)",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO,
        help="Repository root for resolving run_out_dir (default: repo containing this script)",
    )
    p.add_argument(
        "--prune-empty",
        action="store_true",
        help="Scan *_leaf_scores.csv for n_boxes>0 runs: count rows where max(score) < score_threshold",
    )
    p.add_argument(
        "--extracted-alpha",
        action="store_true",
        help="For n_boxes>0 runs, check whether *_extracted.png has any non-transparent pixels (needs Pillow)",
    )
    p.add_argument(
        "--clip-exclude-transparent-png",
        action="store_true",
        help=(
            "For every ok run with clip_combined, inspect *_extracted.png; report CLIP mean/var "
            "excluding runs where the PNG is fully transparent (needs Pillow)."
        ),
    )
    p.add_argument("--json", action="store_true", help="Print one JSON array to stdout")
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    if args.csvs:
        paths = [c.resolve() for c in args.csvs]
    else:
        paths = sorted(repo_root.glob("output/dataset_eval_full_*/eval_results*.csv"))

    if not paths:
        print("No CSV files found. Pass paths explicitly.", file=sys.stderr)
        return 2

    out: list[dict] = []
    for csv_path in paths:
        if not csv_path.is_file():
            print(f"Skip (not a file): {csv_path}", file=sys.stderr)
            continue
        out.append(
            summarize_one(
                csv_path,
                repo_root,
                scan_prune_empty=args.prune_empty,
                scan_extracted_alpha=args.extracted_alpha,
                scan_clip_exclude_transparent_png=args.clip_exclude_transparent_png,
            )
        )

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    for block in out:
        print("=" * 72)
        for k, v in block.items():
            if isinstance(v, dict):
                print(f"{k}:")
                if not v:
                    print("  (none)")
                else:
                    for sk, sv in v.items():
                        print(f"  {sk}: {sv}")
            else:
                print(f"{k}: {v}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
