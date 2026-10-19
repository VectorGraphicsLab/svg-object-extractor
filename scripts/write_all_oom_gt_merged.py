#!/usr/bin/env python3
"""Записать previews/<svg_id>/oom_merged_*/ для всех групп one_visible_of_many (без DINO/SAM)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svg_gt_baseline.dataset_io import filter_rows, load_dataset_rows, load_manifest_paths
from svg_gt_baseline.oom_gt_merge import write_oom_gt_merged_all_groups
from svg_gt_baseline.types import InferenceSettings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Каталог прогона (cache/, previews/)")
    ap.add_argument("--dataset", type=Path, default=None, help="JSONL датасета (иначе из --out/run_config.json)")
    ap.add_argument("--manifest", type=Path, default=None, help="svg_archive_manifest.json (иначе из run_config.json)")
    ap.add_argument("--from-run-config", action="store_true", help="Брать dataset/manifest из --out/run_config.json")
    ap.add_argument("--raster-max-side", type=int, default=None, help="Переопределить raster_max_side (иначе из run_config или 640)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = args.out.resolve()
    cfg_path = out_dir / "run_config.json"
    dataset_path = args.dataset
    manifest_path = args.manifest
    if args.from_run_config or (dataset_path is None or manifest_path is None):
        if not cfg_path.is_file():
            ap.error(f"missing {cfg_path} (pass --dataset and --manifest explicitly)")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if dataset_path is None:
            dataset_path = Path(cfg["dataset_jsonl"])
        if manifest_path is None:
            manifest_path = Path(cfg["manifest_json"])
        st = cfg.get("settings") or {}
        raster_max = args.raster_max_side if args.raster_max_side is not None else int(st.get("raster_max_side", 640))
        box_t = float(st.get("box_threshold", 0.35))
        text_t = float(st.get("text_threshold", 0.25))
        score_t = float(st.get("score_threshold", 0.8))
        seed = int(st.get("seed", 42))
    else:
        raster_max = args.raster_max_side if args.raster_max_side is not None else 640
        box_t, text_t, score_t, seed = 0.35, 0.25, 0.8, 42

    dataset_path = dataset_path.resolve()
    manifest_path = manifest_path.resolve()
    if not dataset_path.is_file():
        print(f"missing dataset {dataset_path}", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"missing manifest {manifest_path}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rows = filter_rows(load_dataset_rows(dataset_path))
    id_to_path = load_manifest_paths(manifest_path)
    settings = InferenceSettings(
        raster_max_side=raster_max,
        box_threshold=box_t,
        text_threshold=text_t,
        score_threshold=score_t,
        seed=seed,
    )
    n = write_oom_gt_merged_all_groups(rows=rows, out_dir=out_dir, id_to_path=id_to_path, settings=settings)
    logging.info("write_all_oom_gt_merged done groups_written=%d", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
