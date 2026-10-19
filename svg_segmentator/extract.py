from __future__ import annotations

import copy
import io
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from svg_segmentator.render import (
    alpha_mask_bool,
    svg_bytes_to_png_bytes,
    svg_bytes_to_png_sized,
    svg_file_to_png_bytes,
)

SVG_NS = "http://www.w3.org/2000/svg"


def _tag(local: str) -> str:
    return f"{{{SVG_NS}}}{local}"


DRAWABLE_LOCAL = frozenset(
    {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
)


def assign_data_local_ids(root: ET.Element) -> None:
    for i, el in enumerate(root.iter()):
        el.set("data-local-id", f"seg_{i}")


def load_tagged_svg(svg_path: Path) -> tuple[ET.Element, dict[ET.Element, ET.Element | None], list[ET.Element], dict[str, ET.Element]]:
    """Parse SVG, assign ``data-local-id`` to every node, return root, parent map, drawable leaves, leaf id map."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    if root.tag != _tag("svg"):
        raise ValueError("Expected single root <svg>")
    assign_data_local_ids(root)
    pmap = parent_map(root)
    leaves = enumerate_drawable_leaves(root)
    leaf_by_lid: dict[str, ET.Element] = {}
    for leaf in leaves:
        lid = leaf.get("data-local-id")
        if lid:
            leaf_by_lid[lid] = leaf
    return root, pmap, leaves, leaf_by_lid


def parent_map(root: ET.Element) -> dict[ET.Element, ET.Element | None]:
    pmap: dict[ET.Element, ET.Element | None] = {root: None}
    for p in root.iter():
        for c in list(p):
            pmap[c] = p
    return pmap


def defs_blocks_for_candidate_render(root_svg: ET.Element) -> list[ET.Element]:
    """
    Full defs for per-candidate test renders: every <defs> that is a direct child of root <svg>,
    in document order (some files have several). If none, use the first <defs> anywhere in the tree.
    """
    blocks = [c for c in root_svg if c.tag == _tag("defs")]
    if blocks:
        return blocks
    for el in root_svg.iter():
        if el.tag == _tag("defs"):
            return [el]
    return []


def enumerate_drawable_leaves(root_svg: ET.Element) -> list[ET.Element]:
    out: list[ET.Element] = []
    for el in root_svg.iter():
        local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if local in DRAWABLE_LOCAL:
            out.append(el)
    return out


def chain_to_root(elem: ET.Element, root: ET.Element, pmap: dict[ET.Element, ET.Element | None]) -> list[ET.Element]:
    chain: list[ET.Element] = []
    cur: ET.Element | None = elem
    while cur is not None:
        chain.append(cur)
        if cur is root:
            break
        cur = pmap.get(cur)
    chain.reverse()
    return chain


def build_branch_svg_tree(root_svg: ET.Element, chain: list[ET.Element]) -> ET.Element:
    """Minimal SVG: same root attrs, full defs subtree(s), single branch clone down to leaf."""
    new_root = ET.Element(chain[0].tag, dict(chain[0].attrib))
    for defs_el in defs_blocks_for_candidate_render(root_svg):
        new_root.append(copy.deepcopy(defs_el))

    parent_new = new_root
    for i in range(1, len(chain)):
        old = chain[i]
        if i == len(chain) - 1:
            parent_new.append(copy.deepcopy(old))
        else:
            stub = ET.Element(old.tag, dict(old.attrib))
            parent_new.append(stub)
            parent_new = stub
    return new_root


def tree_to_svg_bytes(root: ET.Element) -> bytes:
    # Avoid default_namespace=... (breaks xlink / mixed qualified names on write).
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    buf = io.BytesIO()
    tree = ET.ElementTree(root)
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


def directed_coverage_score(c_mask: np.ndarray, t_mask: np.ndarray) -> float:
    """|C ∩ T| / |C|; 0 if |C|==0."""
    c_area = int(c_mask.sum())
    if c_area == 0:
        return 0.0
    inter = int((c_mask & t_mask).sum())
    return inter / c_area


@dataclass
class LeafMaskCache:
    root: ET.Element
    pmap: dict[ET.Element, ET.Element | None]
    leaves: list[ET.Element]
    leaf_ids: list[str]
    leaf_masks: list[np.ndarray]


def collect_keep_ids(selected_leaves: list[ET.Element], pmap: dict[ET.Element, ET.Element | None]) -> set[str]:
    keep: set[str] = set()
    for leaf in selected_leaves:
        cur: ET.Element | None = leaf
        while cur is not None:
            lid = cur.get("data-local-id")
            if lid:
                keep.add(lid)
            cur = pmap.get(cur)
    return keep


def prune_tree_keep_ids(root_svg: ET.Element, keep_ids: set[str]) -> None:
    """In-place: remove nodes not in keep_ids; skip <defs> subtree entirely (unchanged)."""

    def post_prune(elem: ET.Element) -> None:
        if elem.tag == _tag("defs"):
            return
        for child in list(elem):
            if child.tag == _tag("defs"):
                continue
            post_prune(child)
            lid = child.get("data-local-id")
            if lid not in keep_ids:
                elem.remove(child)

    post_prune(root_svg)


def build_leaf_mask_cache(
    root: ET.Element,
    pmap: dict[ET.Element, ET.Element | None],
    leaves: list[ET.Element],
    dpi: float,
    rgb_full: Image.Image,
    candidate_render_dir: Path | None,
    *,
    output_wh: tuple[int, int] | None = None,
) -> LeafMaskCache:
    h, w = rgb_full.size[1], rgb_full.size[0]
    if candidate_render_dir is not None:
        candidate_render_dir.mkdir(parents=True, exist_ok=True)
    leaf_ids: list[str] = []
    leaf_masks: list[np.ndarray] = []
    for leaf_index, leaf in enumerate(leaves):
        chain = chain_to_root(leaf, root, pmap)
        branch_root = build_branch_svg_tree(root, chain)
        svg_bytes = tree_to_svg_bytes(branch_root)
        if output_wh is not None:
            ow, oh = output_wh
            png = svg_bytes_to_png_sized(svg_bytes, output_width=ow, output_height=oh)
        else:
            png = svg_bytes_to_png_bytes(svg_bytes, dpi=dpi)
        rgba = Image.open(io.BytesIO(png)).convert("RGBA")
        if rgba.size != (w, h):
            raise RuntimeError(
                f"Candidate render size {rgba.size} != full render {w}x{h}; check SVG width/height/viewBox"
            )
        c_mask = alpha_mask_bool(rgba)
        lid = leaf.get("data-local-id", "")
        leaf_ids.append(lid)
        leaf_masks.append(c_mask)
        if candidate_render_dir is not None:
            tag_local = leaf.tag.split("}")[-1] if "}" in leaf.tag else leaf.tag
            stem = f"{leaf_index:04d}_{lid}_{tag_local}".replace("/", "_").replace("\\", "_")
            (candidate_render_dir / f"{stem}.png").write_bytes(png)
            (candidate_render_dir / f"{stem}.svg").write_bytes(svg_bytes)
    return LeafMaskCache(root=root, pmap=pmap, leaves=leaves, leaf_ids=leaf_ids, leaf_masks=leaf_masks)


def score_leaf_mask_cache(cache: LeafMaskCache, target_mask: np.ndarray) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for lid, c_mask in zip(cache.leaf_ids, cache.leaf_masks):
        sc = directed_coverage_score(c_mask, target_mask)
        scores.append((lid, sc))
    return scores


def prune_from_scores(cache: LeafMaskCache, scores: list[tuple[str, float]], score_threshold: float) -> tuple[ET.Element, frozenset[str]]:
    selected: list[ET.Element] = []
    for leaf, (_lid, sc) in zip(cache.leaves, scores):
        if sc >= score_threshold:
            selected.append(leaf)
    keep = collect_keep_ids(selected, cache.pmap)
    out_root = copy.deepcopy(cache.root)
    prune_tree_keep_ids(out_root, keep)
    baseline = frozenset(leaf.get("data-local-id", "") for leaf in selected if leaf.get("data-local-id"))
    return out_root, baseline


def run_leaf_mask_scoring(
    root: ET.Element,
    pmap: dict[ET.Element, ET.Element | None],
    leaves: list[ET.Element],
    target_mask: np.ndarray,
    dpi: float,
    rgb_full: Image.Image,
    candidate_render_dir: Path | None,
    *,
    output_wh: tuple[int, int] | None = None,
) -> list[tuple[str, float]]:
    h, w = target_mask.shape
    if rgb_full.size != (w, h):
        raise ValueError(f"Mask shape {(h, w)} vs image {rgb_full.size}")
    cache = build_leaf_mask_cache(
        root, pmap, leaves, dpi, rgb_full, candidate_render_dir, output_wh=output_wh
    )
    return score_leaf_mask_cache(cache, target_mask)


def extract_by_mask(
    svg_path: Path,
    target_mask: np.ndarray,
    *,
    dpi: float,
    score_threshold: float,
    rgb_full: Image.Image,
    max_leaves: int | None = None,
    candidate_render_dir: Path | None = None,
    output_wh: tuple[int, int] | None = None,
) -> tuple[ET.Element, list[tuple[str, float]], frozenset[str]]:
    """
    Returns pruned XML root, leaf mask scores, and baseline selected **leaf** ids (no ancestor ids).

    If ``candidate_render_dir`` is set, each leaf's mini-SVG (branch + defs) is written as ``.svg`` and
    its Cairo render as ``.png`` with names ``{order:04d}_{data-local-id}.*``.
    """
    root, pmap, leaves, _leaf_by_lid = load_tagged_svg(svg_path)
    if max_leaves is not None:
        leaves = leaves[:max_leaves]
    cache = build_leaf_mask_cache(
        root, pmap, leaves, dpi, rgb_full, candidate_render_dir, output_wh=output_wh
    )
    scores = score_leaf_mask_cache(cache, target_mask)
    out_root, baseline = prune_from_scores(cache, scores, score_threshold)
    return out_root, scores, baseline


def write_tree(path: Path, root: ET.Element) -> None:
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def full_pipeline_raster_masks(
    svg_path: Path,
    dpi: float,
) -> tuple[bytes, Image.Image, np.ndarray]:
    """Full-document PNG bytes, RGB image, bool mask from alpha (nonzero = ink)."""
    png_bytes = svg_file_to_png_bytes(svg_path, dpi=dpi)
    rgba = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    rgb = rgba.convert("RGB")
    mask = alpha_mask_bool(rgba)
    return png_bytes, rgb, mask
