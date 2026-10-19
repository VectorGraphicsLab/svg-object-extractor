"""Offline metrics on ``predictions.ndjson`` + extracted PNG previews (see docs/gt_baseline_metrics.md)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from svg_gt_baseline.prediction_ndjson import row_box_index


def leaf_metrics(gt: list[int], pr: list[int]) -> dict[str, float | bool]:
    """IoU on positive indices, micro F1 on all positions, exact vector equality."""
    if len(gt) != len(pr):
        raise ValueError(f"length mismatch {len(gt)} vs {len(pr)}")
    gpos = {i for i, v in enumerate(gt) if v}
    ppos = {i for i, v in enumerate(pr) if v}
    inter = len(gpos & ppos)
    union = len(gpos | ppos)
    if union == 0:
        iou = 1.0
    else:
        iou = inter / union

    tp = sum(1 for a, b in zip(gt, pr, strict=True) if a and b)
    fp = sum(1 for a, b in zip(gt, pr, strict=True) if (not a) and b)
    fn = sum(1 for a, b in zip(gt, pr, strict=True) if a and (not b))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "iou_leaf": float(iou),
        "f1": float(f1),
        "exact": bool(gt == pr),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def merge_binary_masks_or(masks: Iterable[list[int]]) -> list[int]:
    masks = list(masks)
    if not masks:
        return []
    n = len(masks[0])
    out = [0] * n
    for m in masks:
        if len(m) != n:
            raise ValueError("mask length mismatch in OR merge")
        for i, v in enumerate(m):
            if v:
                out[i] = 1
    return out


def _record_primary_key(d: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return (
        int(d["svg_id"]),
        int(d["annotation_id"]),
        int(d["prompt_index"]),
        str(d["detector_id"]),
        str(d["segmenter_id"]),
    )


def _box_rank(d: dict[str, Any]) -> tuple[int, int]:
    """Sort key: None (zero boxes) last among singles; then by box index."""
    idx = row_box_index(d)
    if idx is None:
        return (1, 0)
    return (0, int(idx))


def dedupe_canonical_combined(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One NDJSON dict per (svg_id, annotation_id, prompt_index, detector_id, segmenter_id)."""
    buckets: dict[tuple[int, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for d in lines:
        buckets[_record_primary_key(d)].append(d)
    out: list[dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        rows = sorted(buckets[key], key=_box_rank)
        out.append(rows[0])
    return out


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def discover_oom_groups(previews_root: Path) -> list[dict[str, Any]]:
    """Parse ``previews/<svg_id>/oom_merged_<slug>/meta.json`` files."""
    groups: list[dict[str, Any]] = []
    if not previews_root.is_dir():
        return groups
    for svg_dir in sorted(previews_root.iterdir(), key=lambda p: p.name):
        if not svg_dir.is_dir() or not svg_dir.name.isdigit():
            continue
        svg_id = int(svg_dir.name)
        for sub in svg_dir.glob("oom_merged_*/meta.json"):
            meta = json.loads(sub.read_text(encoding="utf-8"))
            if meta.get("kind") != "one_visible_of_many_gt_merged":
                continue
            groups.append(
                {
                    "svg_id": svg_id,
                    "slug": sub.parent.name,
                    "annotation_ids": [int(x) for x in meta["annotation_ids"]],
                    "prompt_texts": list(meta.get("prompt_texts", [])),
                    "meta_path": sub,
                }
            )
    groups.sort(key=lambda g: (g["svg_id"], g["slug"]))
    return groups


def png_to_float01_rgb(path: Path) -> np.ndarray:
    """HWC float32 in [0, 1]."""
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def inspect_png_rgb(path: Path) -> dict[str, Any]:
    """Corner pixels and RGB stats (uint8) — чтобы проверить «чёрный ли фон» на extracted."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]

    def px(y: int, x: int) -> list[int]:
        return [int(arr[y, x, c]) for c in range(3)]

    corners = {
        "top_left": px(0, 0),
        "top_right": px(0, w - 1),
        "bottom_left": px(h - 1, 0),
        "bottom_right": px(h - 1, w - 1),
    }
    return {
        "path": str(path),
        "shape_hw": [h, w],
        "rgb_min": [int(arr[..., c].min()) for c in range(3)],
        "rgb_max": [int(arr[..., c].max()) for c in range(3)],
        "rgb_mean": [float(arr[..., c].mean()) for c in range(3)],
        "corners_rgb": corners,
    }


def pixel_metrics_mse_mae_psnr(gt: np.ndarray, pr: np.ndarray) -> dict[str, float]:
    if gt.shape != pr.shape:
        raise ValueError(f"shape mismatch {gt.shape} vs {pr.shape}")
    diff = gt - pr
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    # Avoid inf in JSON/CSV; при полном совпадении картинок MSE≈0
    if mse <= 1e-16:
        psnr = 100.0
    else:
        psnr = float(10.0 * math.log10(1.0 / mse))
    return {"mse": mse, "mae": mae, "psnr": psnr}


def rgb_to_gray_luminance(x: np.ndarray) -> np.ndarray:
    """x HWC float [0,1] -> HW gray."""
    return 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]


def ssim_gray_numpy(a: np.ndarray, b: np.ndarray, *, data_range: float = 1.0, win_size: int = 11) -> float:
    """Single-band SSIM (Wang et al.), Gaussian-less box filter variant via uniform_filter."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    if win_size % 2 == 0:
        raise ValueError("win_size must be odd")
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    mu_a = uniform_filter(a, size=win_size, mode="reflect")
    mu_b = uniform_filter(b, size=win_size, mode="reflect")
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = uniform_filter(a * a, size=win_size, mode="reflect") - mu_a2
    sigma_b2 = uniform_filter(b * b, size=win_size, mode="reflect") - mu_b2
    sigma_ab = uniform_filter(a * b, size=win_size, mode="reflect") - mu_ab
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


def _ssim_gray_auto_window(a: np.ndarray, b: np.ndarray, *, data_range: float = 1.0) -> float:
    h, w = a.shape
    m = min(h, w)
    win = min(11, m if m % 2 == 1 else m - 1)
    if win < 3:
        return float("nan")
    return ssim_gray_numpy(a, b, data_range=data_range, win_size=win)


def roi_union_mask(gray_gt: np.ndarray, gray_pr: np.ndarray, *, eps: float) -> np.ndarray:
    """Пиксели, где хотя бы один кадр «не фон» по luminance > eps (float01)."""
    return (gray_gt > eps) | (gray_pr > eps)


def visual_metrics_full_and_roi(
    gt: np.ndarray,
    pr: np.ndarray,
    *,
    roi_gray_eps: float = 2.0 / 255.0,
) -> dict[str, float]:
    """
    Полный кадр + ROI = union(яркость GT > eps или Pred > eps).
    MSE/MAE по RGB усредняются только по пикселям ROI; SSIM по grayscale на tight-crop ROI.
    """
    if gt.shape != pr.shape:
        raise ValueError(f"shape mismatch {gt.shape} vs {pr.shape}")
    g = rgb_to_gray_luminance(gt)
    p = rgb_to_gray_luminance(pr)
    full = pixel_metrics_mse_mae_psnr(gt, pr)
    out: dict[str, float] = {
        "mse_full": full["mse"],
        "mae_full": full["mae"],
        "psnr_full": full["psnr"],
        "ssim_full": ssim_gray_numpy(g, p, data_range=1.0),
    }

    mask = roi_union_mask(g, p, eps=roi_gray_eps)
    n = int(mask.sum())
    h, w = mask.shape
    out["roi_pixels"] = float(n)
    out["roi_coverage"] = float(n / (h * w)) if (h * w) else 0.0

    if n == 0:
        out["mse_roi"] = float("nan")
        out["mae_roi"] = float("nan")
        out["psnr_roi"] = float("nan")
        out["ssim_roi"] = float("nan")
        return out

    diff = gt - pr
    m = mask[..., np.newaxis]
    # Как mse_full / mae_full: среднее по всем скалярам (H×W×3), только по пикселям из ROI
    mse_roi = float(np.sum(diff**2 * m) / (n * 3))
    mae_roi = float(np.sum(np.abs(diff) * m) / (n * 3))
    out["mae_roi"] = mae_roi
    if mse_roi <= 1e-16:
        out["psnr_roi"] = 100.0
    else:
        out["psnr_roi"] = float(10.0 * math.log10(1.0 / mse_roi))
    out["mse_roi"] = mse_roi

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    gc = g[y0:y1, x0:x1]
    pc = p[y0:y1, x0:x1]
    out["ssim_roi"] = _ssim_gray_auto_window(gc, pc, data_range=1.0)
    return out


def extracted_pair_paths(
    out_dir: Path,
    *,
    svg_id: int,
    annotation_id: int,
    prompt_index: int,
    detector_id: str,
    segmenter_id: str,
    gt_is_merged_group: bool,
) -> tuple[Path, Path]:
    seg_dir = (
        out_dir
        / "previews"
        / str(svg_id)
        / f"ann_{annotation_id}"
        / f"p{prompt_index}"
        / detector_id
        / segmenter_id
    )
    pred_p = seg_dir / "extracted_union.png"
    if gt_is_merged_group:
        gt_p = seg_dir / "gt_merged_group_extracted.png"
    else:
        gt_p = out_dir / "previews" / str(svg_id) / f"ann_{annotation_id}" / "gt_extracted.png"
    return gt_p, pred_p


def safe_visual_metrics(
    gt_path: Path,
    pred_path: Path,
    *,
    roi_gray_eps: float = 2.0 / 255.0,
) -> dict[str, float] | None:
    if not gt_path.is_file() or not pred_path.is_file():
        return None
    try:
        gt = png_to_float01_rgb(gt_path)
        pr = png_to_float01_rgb(pred_path)
        return visual_metrics_full_and_roi(gt, pr, roi_gray_eps=roi_gray_eps)
    except (OSError, ValueError):
        return None


def ndjson_index_first(lines: list[dict[str, Any]]) -> dict[tuple[int, int, int, str, str], dict[str, Any]]:
    """First row per primary key (same logic as dedupe)."""
    return { _record_primary_key(d): d for d in dedupe_canonical_combined(lines) }
