"""Shared paths and validated inference settings."""

import os
from dataclasses import dataclass
from pathlib import Path


def workspace() -> Path:
    return Path(os.environ.get("VISION_WORKSPACE", ".")).resolve()


@dataclass(frozen=True)
class InferenceConfig:
    model: str = "models/yolo11n.pt"
    confidence: float = 0.3
    iou: float = 0.5
    image_size: int = 640
    device: str = "cpu"
    tracking: bool = True
    tracker_type: str = "ilp"
    max_frames: int | None = None
    classes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.tracker_type not in ("ilp", "bytetrack"):
            raise ValueError("tracker_type must be 'ilp' or 'bytetrack'.")
        if not 0 < self.confidence <= 1 or not 0 < self.iou <= 1:
            raise ValueError("Confidence and IoU must be in (0, 1].")
        if not 32 <= self.image_size <= 1920 or self.image_size % 32:
            raise ValueError("Image size must be a multiple of 32 between 32 and 1920.")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("Maximum frames must be positive.")
        if self.classes is not None and (not self.classes or min(self.classes) < 0):
            raise ValueError("Class filters must contain nonnegative class IDs.")
