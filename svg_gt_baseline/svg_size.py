"""Infer nominal SVG pixel dimensions for Cairo output_width/height (single final rasterize)."""

from __future__ import annotations

import io
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from svg_segmentator.render import svg_bytes_to_png_bytes, svg_bytes_to_png_sized


def _parse_length_px(s: str) -> float | None:
    s = s.strip()
    if not s or s.endswith("%"):
        return None
    m = re.match(r"^([\d.]+)\s*(px)?$", s, re.I)
    if m:
        return float(m.group(1))
    m = re.match(r"^([\d.]+)\s*pt$", s, re.I)
    if m:
        return float(m.group(1)) * (96.0 / 72.0)
    m = re.match(r"^([\d.]+)\s*mm$", s, re.I)
    if m:
        return float(m.group(1)) * (96.0 / 25.4)
    m = re.match(r"^([\d.]+)\s*cm$", s, re.I)
    if m:
        return float(m.group(1)) * (960.0 / 25.4)
    m = re.match(r"^([\d.]+)\s*in$", s, re.I)
    if m:
        return float(m.group(1)) * 96.0
    try:
        return float(s)
    except ValueError:
        return None


def _viewbox_wh(s: str | None) -> tuple[float, float] | None:
    if not s:
        return None
    parts = s.replace(",", " ").split()
    if len(parts) < 4:
        return None
    try:
        w = float(parts[2])
        h = float(parts[3])
        if w > 0 and h > 0:
            return w, h
    except (ValueError, IndexError):
        return None
    return None


def infer_raster_dimensions(svg_bytes: bytes) -> tuple[int, int]:
    """
    Best-effort intrinsic width/height in pixels at 96dpi semantics.

    Falls back to a one-time Cairo render at dpi=96 and reading PNG size
    (still Cairo-only for the probe; caller then uses sized output for final).
    """
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        return _probe_size(svg_bytes)

    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag.lower() != "svg":
        return _probe_size(svg_bytes)

    vb = root.get("viewBox") or root.get("viewbox")
    vb_wh = _viewbox_wh(vb)
    w_attr = _parse_length_px(root.get("width") or "")
    h_attr = _parse_length_px(root.get("height") or "")

    if vb_wh is not None and w_attr is not None and h_attr is not None:
        return max(1, int(round(w_attr))), max(1, int(round(h_attr)))
    if vb_wh is not None:
        w0, h0 = vb_wh
        return max(1, int(round(w0))), max(1, int(round(h0)))
    if w_attr is not None and h_attr is not None:
        return max(1, int(round(w_attr))), max(1, int(round(h_attr)))

    return _probe_size(svg_bytes)


def _probe_size(svg_bytes: bytes) -> tuple[int, int]:
    png = svg_bytes_to_png_bytes(svg_bytes, dpi=96.0)
    im = Image.open(io.BytesIO(png))
    return im.size


def compute_output_wh(svg_bytes: bytes, max_side: int) -> tuple[int, int]:
    w0, h0 = infer_raster_dimensions(svg_bytes)
    m = max(w0, h0)
    if m <= 0:
        return max_side, max_side
    scale = max_side / m
    ow = max(1, int(round(w0 * scale)))
    oh = max(1, int(round(h0 * scale)))
    return ow, oh


def rasterize_svg_path_max_side(svg_path: Path, max_side: int) -> tuple[bytes, tuple[int, int]]:
    """Single Cairo rasterize at exact output size (max(w,h) ≈ max_side)."""
    svg_bytes = svg_path.read_bytes()
    ow, oh = compute_output_wh(svg_bytes, max_side)
    png = svg_bytes_to_png_sized(svg_bytes, output_width=ow, output_height=oh)
    return png, (ow, oh)
