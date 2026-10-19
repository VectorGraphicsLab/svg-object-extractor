from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from svg_segmentator.backends.types import DetectionBox, DetectionResult


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dino_cache_key_parts(
    *,
    svg_hash: str,
    detector_id: str,
    box_threshold: float,
    text_threshold: float,
    raster_max_side: int,
    prompt_text: str,
) -> str:
    payload = "|".join(
        [
            svg_hash,
            detector_id,
            f"{box_threshold:.6f}",
            f"{text_threshold:.6f}",
            str(raster_max_side),
            prompt_text,
        ]
    )
    return sha256_hex(payload.encode("utf-8"))


def detection_result_to_jsonable(det: DetectionResult) -> dict[str, Any]:
    return {
        "boxes": [
            {
                "x0": b.x0,
                "y0": b.y0,
                "x1": b.x1,
                "y1": b.y1,
                "score": b.score,
                "label": b.label,
            }
            for b in det.boxes
        ],
    }


def detection_result_from_dict(d: dict[str, Any]) -> DetectionResult:
    boxes: list[DetectionBox] = []
    for b in d.get("boxes", []):
        boxes.append(
            DetectionBox(
                x0=float(b["x0"]),
                y0=float(b["y0"]),
                x1=float(b["x1"]),
                y1=float(b["y1"]),
                score=float(b["score"]),
                label=str(b.get("label", "")),
            )
        )
    return DetectionResult(boxes=boxes)


def read_or_run_dino_json(
    cache_path: Path,
    runner: Callable[[], DetectionResult],
) -> DetectionResult:
    if cache_path.is_file():
        d = json.loads(cache_path.read_text(encoding="utf-8"))
        return detection_result_from_dict(d)
    det = runner()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(detection_result_to_jsonable(det), ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache_path)
    return det
