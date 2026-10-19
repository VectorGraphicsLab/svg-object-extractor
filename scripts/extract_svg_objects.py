"""Extract SVG objects with pluggable detector/segmenter backends."""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from svg_segmentator.repo_env import load_repo_dotenv

load_repo_dotenv()

from svg_segmentator.backends.registry import list_detector_ids, list_segmenter_ids
from svg_segmentator.eval_pipeline import EvalPipeline
from svg_segmentator.pipeline import PipelineConfig


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)

    @property
    def encoding(self) -> str:
        for s in self._streams:
            enc = getattr(s, "encoding", None)
            if enc:
                return enc
        return "utf-8"


def _safe_segment(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def default_output_dir(svg_path: Path, args: argparse.Namespace) -> Path:
    stem = svg_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sq = _safe_segment(args.query)
    return Path("output") / "_".join([stem, f"q-{sq}", "matrix", ts])


def _resolve_backend_selections(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    detectors = _parse_csv_list(args.detectors)
    segmenters = _parse_csv_list(args.segmenters)

    if args.detector:
        detectors = [args.detector]
    if args.segmenter:
        segmenters = [args.segmenter]

    if not detectors:
        detectors = ["gdino-tiny"]
    if not segmenters:
        segmenters = ["sam-vit-base" if args.mask_source == "sam" else "box-fill"]
    return detectors, segmenters


def _user_requested_matrix(args: argparse.Namespace) -> bool:
    return bool(args.detector or args.detectors or args.segmenter or args.segmenters)


def _iter_combos(detectors: list[str], segmenters: list[str], matrix_mode: str) -> list[tuple[str, str]]:
    if matrix_mode == "pairwise":
        if len(detectors) != len(segmenters):
            raise ValueError("pairwise mode requires equal number of detectors and segmenters")
        return list(zip(detectors, segmenters))
    return [(d, s) for d in detectors for s in segmenters]


def main() -> int:
    p = argparse.ArgumentParser(description="Extract SVG elements with pluggable detector/segmenter backends")
    p.add_argument("svg", type=Path, help="Input .svg path")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--query", type=str, default="building")
    p.add_argument("--dpi", type=float, default=96.0)
    p.add_argument("--score-threshold", type=float, default=0.8)
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--max-leaves", type=int, default=0)
    p.add_argument("--candidate-render-dir", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", type=str, default="run.log")
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")

    # New backend selection flags.
    p.add_argument("--detector", type=str, default="")
    p.add_argument("--detectors", type=str, default="")
    p.add_argument("--segmenter", type=str, default="")
    p.add_argument("--segmenters", type=str, default="")
    p.add_argument("--matrix-mode", type=str, choices=["pairwise", "full"], default="full")

    # Backward-compatibility flags.
    p.add_argument("--gdino-model", type=str, default="")
    p.add_argument("--sam-model", type=str, default="")
    p.add_argument("--mask-source", type=str, choices=["sam", "dino-box"], default="sam")
    args = p.parse_args()

    if not args.svg.is_file():
        print(f"Not a file: {args.svg}", file=sys.stderr)
        return 1

    detectors, segmenters = _resolve_backend_selections(args)
    for d in detectors:
        if d not in list_detector_ids():
            print(f"Unknown detector '{d}'. Available: {', '.join(list_detector_ids())}", file=sys.stderr)
            return 2
    for s in segmenters:
        if s not in list_segmenter_ids():
            print(f"Unknown segmenter '{s}'. Available: {', '.join(list_segmenter_ids())}", file=sys.stderr)
            return 2

    if args.out_dir is None:
        args.out_dir = default_output_dir(args.svg, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = args.out_dir / args.log_file
        log_fh = log_path.open("w", encoding="utf-8")
        sys.stdout = _TeeStream(sys.stdout, log_fh)
        sys.stderr = _TeeStream(sys.stderr, log_fh)
        print(f"Log file: {log_path}")

    if args.verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        os.environ.setdefault("HF_HUB_VERBOSITY", "debug")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Detectors: {detectors}")
    print(f"Segmenters: {segmenters}")

    max_leaves = args.max_leaves if args.max_leaves > 0 else None
    combos = _iter_combos(detectors, segmenters, args.matrix_mode)
    explicit_matrix = _user_requested_matrix(args)
    for det_id, seg_id in combos:
        if len(combos) == 1 and not explicit_matrix:
            combo_dir = args.out_dir
        else:
            combo_dir = args.out_dir / f"{det_id}__{seg_id}"
            combo_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running combo {det_id} + {seg_id} -> {combo_dir}")
        pipeline = EvalPipeline.from_registry(
            detector_id=det_id,
            segmenter_id=seg_id,
            device=device,
            clip_model_id=args.clip_model,
            gdino_model_override=args.gdino_model or None,
            sam_model_override=args.sam_model or None,
        )
        combo_cand_dir = None
        if args.candidate_render_dir is not None:
            combo_cand_dir = combo_dir / args.candidate_render_dir
        pipeline.run(
            svg_path=args.svg,
            out_dir=combo_dir,
            config=PipelineConfig(
                query=args.query,
                dpi=args.dpi,
                score_threshold=args.score_threshold,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                max_leaves=max_leaves,
                clip_model_id=args.clip_model,
            ),
            candidate_render_dir=combo_cand_dir,
        )

    print("All combinations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
