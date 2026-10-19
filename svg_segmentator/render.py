from __future__ import annotations

import io
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
from PIL import Image


def svg_file_to_png_bytes(svg_path: Path, dpi: float = 96.0) -> bytes:
    return cairosvg.svg2png(url=str(svg_path), dpi=dpi)


def svg_bytes_to_png_bytes(svg_bytes: bytes, dpi: float = 96.0) -> bytes:
    return cairosvg.svg2png(bytestring=svg_bytes, dpi=dpi)


def svg_bytes_to_png_sized(svg_bytes: bytes, *, output_width: int, output_height: int) -> bytes:
    """Rasterize SVG to an exact pixel size (Cairo scales; no PIL resize)."""
    return cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=int(output_width),
        output_height=int(output_height),
    )


def svg_file_to_png_sized(svg_path: Path, *, output_width: int, output_height: int) -> bytes:
    return cairosvg.svg2png(
        url=str(svg_path),
        output_width=int(output_width),
        output_height=int(output_height),
    )


def png_bytes_to_rgba(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def alpha_mask_bool(rgba: Image.Image):
    import numpy as np

    a = np.array(rgba)[:, :, 3]
    return a > 0
