"""Build pruned full-frame SVG from a set of leaf ``data-local-id`` (ancestors auto-included)."""

from __future__ import annotations

import copy
from pathlib import Path
from xml.etree import ElementTree as ET

from svg_segmentator.extract import collect_keep_ids, prune_tree_keep_ids, tree_to_svg_bytes, write_tree
from svg_segmentator.render import svg_bytes_to_png_bytes


def subset_png_bytes(
    root_template: ET.Element,
    pmap: dict[ET.Element, ET.Element | None],
    leaf_by_lid: dict[str, ET.Element],
    active_leaf_lids: frozenset[str],
    dpi: float,
) -> bytes:
    """Deep-copy document, keep only ``active_leaf_lids`` plus ancestors; rasterize with CairoSVG."""
    selected = [leaf_by_lid[lid] for lid in active_leaf_lids if lid in leaf_by_lid]
    keep = collect_keep_ids(selected, pmap)
    doc = copy.deepcopy(root_template)
    prune_tree_keep_ids(doc, keep)
    return svg_bytes_to_png_bytes(tree_to_svg_bytes(doc), dpi)


def write_subset_svg(
    out_path: Path,
    root_template: ET.Element,
    pmap: dict[ET.Element, ET.Element | None],
    leaf_by_lid: dict[str, ET.Element],
    active_leaf_lids: frozenset[str],
) -> None:
    selected = [leaf_by_lid[lid] for lid in active_leaf_lids if lid in leaf_by_lid]
    keep = collect_keep_ids(selected, pmap)
    doc = copy.deepcopy(root_template)
    prune_tree_keep_ids(doc, keep)
    write_tree(out_path, doc)
