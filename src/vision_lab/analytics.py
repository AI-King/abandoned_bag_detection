"""Metrics distinguish repeated observations from unique tracker IDs."""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Analytics:
    frames: int = 0
    frames_with_detections: int = 0
    detections: int = 0
    confidence_sum: float = 0.0
    peak_objects: int = 0
    class_counts: Counter = field(default_factory=Counter)
    track_ids: set[int] = field(default_factory=set)

    def update(self, detections: list[dict]) -> dict:
        count = len(detections)
        self.frames += 1
        self.frames_with_detections += int(count > 0)
        self.detections += count
        self.peak_objects = max(self.peak_objects, count)
        confidences = [d["confidence"] for d in detections]
        self.confidence_sum += sum(confidences)
        self.class_counts.update(d["class_name"] for d in detections)
        self.track_ids.update(d["track_id"] for d in detections if d["track_id"] is not None)
        return {
            "detections": count,
            "mean_confidence": sum(confidences) / count if count else 0.0,
        }

    def summary(self, fps: float, elapsed: float, tracking: bool) -> dict:
        return {
            "frames_processed": self.frames,
            "duration_seconds": self.frames / fps,
            "processing_seconds": elapsed,
            "processing_fps": self.frames / elapsed if elapsed else 0.0,
            "total_detections": self.detections,
            "unique_tracks": len(self.track_ids) if tracking else None,
            "peak_objects_per_frame": self.peak_objects,
            "mean_confidence": self.confidence_sum / self.detections if self.detections else 0.0,
            "frames_with_detections": self.frames_with_detections,
            "class_counts": dict(self.class_counts.most_common()),
        }
