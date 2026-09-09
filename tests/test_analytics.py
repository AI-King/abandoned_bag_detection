import pytest

from vision_lab.analytics import Analytics
from vision_lab.config import InferenceConfig


def test_observations_are_not_unique_objects_and_empty_frames_count():
    analytics = Analytics()
    detection = {"confidence": 0.8, "class_name": "suitcase", "track_id": 3}
    analytics.update([detection])
    analytics.update([])
    analytics.update([detection])
    summary = analytics.summary(25, 1, True)
    assert summary["total_detections"] == 2
    assert summary["unique_tracks"] == 1
    assert summary["frames_processed"] == 3
    assert summary["frames_with_detections"] == 2
    assert summary["mean_confidence"] == pytest.approx(0.8)
    assert summary["duration_seconds"] == pytest.approx(0.12)


def test_empty_video_has_finite_metrics():
    summary = Analytics().summary(25, 0, False)
    assert summary["mean_confidence"] == 0
    assert summary["processing_fps"] == 0
    assert summary["unique_tracks"] is None


@pytest.mark.parametrize(
    "kwargs",
    [{"confidence": 0}, {"iou": 2}, {"image_size": 333}, {"max_frames": -1}, {"classes": ()}],
)
def test_invalid_inference_config(kwargs):
    with pytest.raises(ValueError):
        InferenceConfig(**kwargs)
