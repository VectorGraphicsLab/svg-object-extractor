from __future__ import annotations

import json
from pathlib import Path

from svg_gt_baseline.types import DatasetAnnotationRow


def load_manifest_paths(manifest_path: Path) -> dict[int, Path]:
    """Map svg_id -> absolute path to SVG file next to manifest."""
    base = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[int, Path] = {}
    for row in data:
        sid = int(row["svg_id"])
        name = row["filename_in_archive"]
        out[sid] = (base / name).resolve()
    return out


def load_dataset_rows(path: Path) -> list[DatasetAnnotationRow]:
    rows: list[DatasetAnnotationRow] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                DatasetAnnotationRow(
                    svg_id=int(d["svg_id"]),
                    annotation_id=int(d["annotation_id"]),
                    annotation_kind=str(d.get("annotation_kind", "")),
                    object_key=str(d.get("object_key", "")),
                    original_filename=str(d.get("original_filename", "")),
                    node_count=int(d["node_count"]),
                    binary_mask=list(d["binary_mask"]),
                    prompt_texts=list(d["prompt_texts"]),
                    review_decision=d.get("review_decision"),
                    raw=d,
                )
            )
    return rows


def filter_rows(rows: list[DatasetAnnotationRow]) -> list[DatasetAnnotationRow]:
    return [r for r in rows if r.review_decision != "declined"]


def smoke_svg_ids(rows: list[DatasetAnnotationRow], n: int) -> frozenset[int]:
    seen: list[int] = []
    for r in rows:
        if r.svg_id not in seen:
            seen.append(r.svg_id)
        if len(seen) >= n:
            break
    return frozenset(seen)
