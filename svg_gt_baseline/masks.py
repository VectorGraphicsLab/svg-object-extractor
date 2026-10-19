from __future__ import annotations

import xml.etree.ElementTree as ET

from svg_segmentator.extract import DRAWABLE_LOCAL, LeafMaskCache


def node_count_from_root(root: ET.Element) -> int:
    return sum(1 for _ in root.iter())


def pred_mask_from_leaf_ids(root: ET.Element, selected_lids: frozenset[str]) -> list[int]:
    """
    Full-length 0/1 vector in ``enumerate(root.iter())`` order (``seg_i`` indices).

    Only drawable tags may be 1 (matches GT service invariant).
    """
    n = node_count_from_root(root)
    out = [0] * n
    for el in root.iter():
        lid = el.get("data-local-id")
        if not lid or lid not in selected_lids:
            continue
        local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if local not in DRAWABLE_LOCAL:
            continue
        try:
            idx = int(lid.replace("seg_", ""))
        except ValueError:
            continue
        if 0 <= idx < n:
            out[idx] = 1
    return out


def leaf_scores_from_binary_mask(leaf_cache: LeafMaskCache, binary_mask: list[int]) -> list[tuple[str, float]]:
    """
    Per-leaf scores for ``prune_from_scores``: 1.0 if ``binary_mask[seg_i]==1`` else 0.0.

    Order matches ``leaf_cache.leaves`` (required by ``prune_from_scores``).
    """
    out: list[tuple[str, float]] = []
    for leaf in leaf_cache.leaves:
        lid = leaf.get("data-local-id") or ""
        try:
            idx = int(lid.replace("seg_", ""))
        except ValueError:
            out.append((lid, 0.0))
            continue
        bit = binary_mask[idx] if 0 <= idx < len(binary_mask) else 0
        out.append((lid, 1.0 if bit else 0.0))
    return out
