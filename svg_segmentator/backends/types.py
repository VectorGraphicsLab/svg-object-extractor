from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class DetectionBox:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    label: str


@dataclass(frozen=True)
class DetectionResult:
    boxes: list[DetectionBox]


@dataclass(frozen=True)
class SegmentationResult:
    masks: list[np.ndarray]
    combined_mask: np.ndarray


@dataclass(frozen=True)
class RunContext:
    image_rgb: Image.Image
    device: torch.device
    query: str
    text_labels: list[str]
