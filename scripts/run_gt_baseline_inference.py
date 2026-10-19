#!/usr/bin/env python3
"""CLI: GT baseline inference (DINO × SAM / box-fill) with disk cache."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svg_segmentator.repo_env import load_repo_dotenv

load_repo_dotenv()

from svg_gt_baseline.engine import run_inference_job
from svg_gt_baseline.types import InferenceSettings


def main() -> int:
    ap = argparse.ArgumentParser(description="GT dataset baseline inference (see docs/baseline_gt_dataset_inference.md)")
    ap.add_argument("--dataset", type=Path, required=True, help="Export JSONL (e.g. dataset.jsonl)")
    ap.add_argument("--manifest", type=Path, required=True, help="svg_archive_manifest.json next to SVG files")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory (cache + predictions). Default: {_REPO_ROOT / 'output' / 'gt_baseline'}",
    )
    ap.add_argument("--smoke", action="store_true", help="Only first N unique svg_id after filters")
    ap.add_argument("--smoke-n", type=int, default=3, metavar="N")
    ap.add_argument(
        "--only-svg-ids",
        type=str,
        default=None,
        metavar="IDS",
        help="Comma-separated svg_id list (e.g. 68,14). Applied after filters; overrides breadth — use for OOM merged GT smoke without large --smoke-n",
    )
    ap.add_argument("--raster-max-side", type=int, default=640)
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--score-threshold", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index if GPU available")
    ap.add_argument(
        "--no-oom-gt-merged",
        action="store_true",
        help="Skip writing merged GT previews for one_visible_of_many groups (see previews/<svg_id>/oom_merged_*)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = (_REPO_ROOT / "output" / "gt_baseline").resolve() if args.out is None else args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root_log = logging.getLogger()
    root_log.handlers.clear()
    root_log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    fmt_console = logging.Formatter("%(levelname)s %(message)s")
    fmt_file = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt_console)
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8", mode="a")
    fh.setFormatter(fmt_file)
    root_log.addHandler(sh)
    root_log.addHandler(fh)
    logging.info("Logging also to %s", out_dir / "run.log")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        logging.info("Hugging Face Hub token is set (HF_TOKEN / HUGGING_FACE_HUB_TOKEN).")
    else:
        logging.warning(
            "No Hugging Face Hub token in environment; add HF_TOKEN to repo .env or export it "
            "(see README). Unauthenticated Hub requests are rate-limited."
        )

    settings = InferenceSettings(
        raster_max_side=args.raster_max_side,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        score_threshold=args.score_threshold,
        seed=args.seed,
    )

    only_ids: frozenset[int] | None = None
    if args.only_svg_ids:
        only_ids = frozenset(int(x.strip()) for x in args.only_svg_ids.split(",") if x.strip())

    run_inference_job(
        dataset_jsonl=args.dataset.resolve(),
        manifest_json=args.manifest.resolve(),
        out_dir=out_dir,
        settings=settings,
        smoke=args.smoke,
        smoke_n=args.smoke_n,
        cuda_device=args.cuda_device,
        write_oom_gt_merged=not args.no_oom_gt_merged,
        only_svg_ids=only_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
