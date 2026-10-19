"""
CLIP-guided genetic refinement of leaf selection (starts from SAM mask + directed coverage baseline).
"""

from __future__ import annotations

import argparse
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

from svg_segmentator.evolution import EvolutionConfig, run_clip_evolution


def _safe_segment(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def default_out_dir(svg: Path, args: argparse.Namespace) -> Path:
    stem = svg.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sq = _safe_segment(args.query)
    return Path("output") / "_".join(
        [stem, f"q-{sq}", "evoclip", f"pop{args.population}", f"gen{args.generations}", ts]
    )


def main() -> int:
    p = argparse.ArgumentParser(description="CLIP evolution over drawable leaves (SAM baseline)")
    p.add_argument("svg", type=Path)
    p.add_argument("--query", type=str, default="building")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dpi", type=float, default=96.0)
    p.add_argument("--score-threshold", type=float, default=0.8)
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--gdino-model", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam-model", type=str, default="facebook/sam-vit-base")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--population", type=int, default=25)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--mutation-flips", type=int, default=3)
    p.add_argument("--crossover-or-prob", type=float, default=0.5)
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.svg.is_file():
        print(f"Not a file: {args.svg}", file=sys.stderr)
        return 1

    out = args.out_dir or default_out_dir(args.svg, args)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

    gdino_processor = AutoProcessor.from_pretrained(args.gdino_model)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to(device)
    gdino_model.eval()
    sam_processor = SamProcessor.from_pretrained(args.sam_model)
    sam_model = SamModel.from_pretrained(args.sam_model).to(device)
    sam_model.eval()

    evo = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        mutation_flips=args.mutation_flips,
        crossover_or_prob=args.crossover_or_prob,
        clip_model_id=args.clip_model,
        seed=args.seed,
    )

    best, log = run_clip_evolution(
        args.svg,
        args.query,
        dpi=args.dpi,
        score_threshold=args.score_threshold,
        gdino_processor=gdino_processor,
        gdino_model=gdino_model,
        sam_processor=sam_processor,
        sam_model=sam_model,
        device=device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        evo_config=evo,
        out_dir=out,
    )
    print(f"Done. Best n_leaves={len(best)} last_clip={log[-1][1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
