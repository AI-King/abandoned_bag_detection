"""Exercise actual encoding, CSV analytics, MLflow storage, and failure handling."""

import json
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
import pytest
from ultralytics.engine.results import Results

from vision_lab.config import InferenceConfig
from vision_lab.pipeline import process_video
from vision_lab.unattended import MonitorConfig


@pytest.fixture
def media_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
    source = tmp_path / "test.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(50):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    model = tmp_path / "fake.pt"
    model.write_bytes(b"test-model-checksum")
    return source, model


class FakeModel:
    names = {0: "person", 28: "suitcase"}

    def __init__(self, _):
        pass

    def track(self, frame, **kwargs):
        boxes = np.array([[90, 60, 120, 90, 7, 0.9, 28]], dtype=np.float32)
        return [Results(frame, "test", self.names, boxes=boxes)]


def test_full_pipeline_logs_playable_video_and_alerts(media_workspace):
    source, model = media_workspace
    output = process_video(
        source,
        InferenceConfig(model=str(model)),
        MonitorConfig(alert_seconds=2, stationary_seconds=1),
        model_factory=FakeModel,
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["alert_count"] == 1
    assert summary["unique_tracks"] == 1
    assert summary["frames_processed"] == 50
    events = json.loads((output / "events.json").read_text())
    assert events[0]["timestamp_seconds"] == pytest.approx(3, abs=0.1)
    capture = cv2.VideoCapture(str(output / "annotated.mp4"))
    assert capture.read()[0]
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 50
    capture.release()
    run = mlflow.get_run(summary["mlflow_run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["alert_count"] == 1
    assert len(pd.read_csv(output / "detections.csv")) == 50
    assert (output / "COMPLETE").exists()


def test_failure_records_failed_run_and_no_completed_video(media_workspace):
    source, model = media_workspace

    def broken_model(_):
        raise RuntimeError("Deliberate detector failure")

    with pytest.raises(RuntimeError, match="Deliberate"):
        process_video(source, InferenceConfig(model=str(model)), model_factory=broken_model)
    outputs = list((source.parent / "outputs").iterdir())
    assert len(outputs) == 1
    assert (outputs[0] / "FAILED.txt").exists()
    assert not (outputs[0] / "COMPLETE").exists()
    assert not (outputs[0] / "annotated_raw.mp4").exists()


def test_bad_video_rejected_before_model_loading(tmp_path):
    video = tmp_path / "bad.mp4"
    video.write_text("not a video")
    with pytest.raises(ValueError, match="decode"):
        process_video(video, InferenceConfig())


def test_frame_limit_marks_partial_analysis(media_workspace):
    source, model = media_workspace
    output = process_video(
        source, InferenceConfig(model=str(model), max_frames=10), model_factory=FakeModel
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["truncated"] is True
    assert summary["frames_processed"] == 10


@pytest.mark.integration
@pytest.mark.skipif(not Path("models/yolo11n.pt").exists(), reason="Run vision prepare first")
def test_real_yolo_video(tmp_path):
    output = process_video(
        Path("data/videos/LeftBag.mp4"),
        InferenceConfig(max_frames=5),
        MonitorConfig(),
        output_root=tmp_path,
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["frames_processed"] == 5
    assert summary["total_detections"] > 0


@pytest.mark.integration
@pytest.mark.skipif(not Path("models/yolo11n.pt").exists(), reason="Run vision prepare first")
def test_real_yolo_five_minute_timer(tmp_path, monkeypatch):
    from vision_lab.fixtures import make_timer_fixture

    source = make_timer_fixture(Path.cwd())
    monkeypatch.setenv("VISION_WORKSPACE", str(tmp_path))
    output = process_video(source, InferenceConfig(), MonitorConfig())
    events = json.loads((output / "events.json").read_text())
    alerts = [event for event in events if event["event"] == "alert"]
    assert len(alerts) == 1
    assert alerts[0]["timestamp_seconds"] == 303
    assert alerts[0]["unattended_seconds"] == 300
