"""Streaming inference with bounded frame memory and auditable output artifacts."""

import csv
import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import cv2
import mlflow

from vision_lab.analytics import Analytics
from vision_lab.config import InferenceConfig, workspace
from vision_lab.ilp_tracker import ILPTracker
from vision_lab.media import to_browser_video
from vision_lab.provenance import sha256
from vision_lab.tracking import tracked_run
from vision_lab.unattended import BAG_NAMES, BagMonitor, MonitorConfig

DETECTION_FIELDS = [
    "frame",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "confidence",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
]
BAG_FIELDS = [
    "timestamp_seconds",
    "track_id",
    "class_name",
    "status",
    "unattended_seconds",
    "person_nearby",
    "alert_id",
]


def load_model(path: str):
    from ultralytics import YOLO

    if not Path(path).is_file():
        raise FileNotFoundError(f"Model not found: {path}. Run 'vision prepare' first.")
    return YOLO(path)


def result_detections(result) -> list[dict]:
    boxes = result.boxes.cpu().numpy() if result.boxes is not None else None
    if boxes is None or len(boxes) == 0:
        return []
    ids = boxes.id.astype(int).tolist() if boxes.id is not None else [None] * len(boxes)
    return [
        {
            "class_id": int(cls),
            "class_name": result.names[int(cls)],
            "confidence": float(conf),
            "track_id": track_id,
            "bbox": box,
        }
        for box, cls, conf, track_id in zip(
            boxes.xyxy.tolist(),
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            ids,
            strict=True,
        )
    ]


def annotate(frame, detections, bag_rows, timestamp: float):
    states = {row["track_id"]: row for row in bag_rows}
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        state = states.get(detection["track_id"])
        color = (175, 210, 70)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        if detection["track_id"] is not None:
            label += f" #{detection['track_id']}"
        if state:
            color = (75, 85, 250) if state["status"] == "alert" else (60, 195, 255)
            label += f" | {state['status']} {state['unattended_seconds']:.0f}s"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (max(0, x1), max(35, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 25), (23, 16, 11), -1)
    cv2.putText(
        frame,
        f"VISION LAB | video {timestamp:.1f}s",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (234, 240, 248),
        1,
        cv2.LINE_AA,
    )
    return frame


def process_video(
    source: Path,
    config: InferenceConfig,
    monitor_config: MonitorConfig | None = None,
    output_root: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    model_factory: Callable = load_model,
) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if monitor_config and not config.tracking:
        raise ValueError("Unattended luggage monitoring requires object tracking.")
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or not width or not height:
        capture.release()
        raise ValueError("Cannot decode video or its frame rate. Convert it to a valid MP4 first.")
    if width * height > 3840 * 2160:
        capture.release()
        raise ValueError("Video resolution exceeds 4K; resize the input before processing.")
    output = (output_root or workspace() / "outputs") / uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    temporary_video = output / "annotated_raw.mp4"
    writer = cv2.VideoWriter(
        str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the output video.")
    analytics = Analytics()
    monitor = BagMonitor(monitor_config) if monitor_config else None
    started = time.perf_counter()
    try:
        with tracked_run("inference", source.stem) as run:
            metadata = {
                "inference": asdict(config),
                "monitor": asdict(monitor_config) if monitor_config else None,
                "source_name": source.name,
                "source_sha256": sha256(source),
                "source_fps": fps,
                "source_frames": total,
                "model_sha256": sha256(Path(config.model)),
                "mlflow_run_id": run.info.run_id,
            }
            mlflow.log_params(
                {
                    **asdict(config),
                    "source_sha256": metadata["source_sha256"],
                    "model_sha256": metadata["model_sha256"],
                }
            )
            if monitor_config:
                mlflow.log_params({f"monitor.{k}": v for k, v in asdict(monitor_config).items()})
            (output / "config.json").write_text(json.dumps(metadata, indent=2))
            model = model_factory(config.model)  # Fresh tracker state for every video/run.
            ilp_tracker = (
                ILPTracker() if (config.tracking and config.tracker_type == "ilp") else None
            )
            class_filter = config.classes
            if monitor and class_filter is None:
                class_filter = tuple(
                    k for k, v in model.names.items() if v in BAG_NAMES or v == "person"
                )
                if not any(v in BAG_NAMES for v in model.names.values()):
                    raise ValueError("Model has no supported bag classes.")
            with (
                (output / "detections.csv").open("w", newline="") as detection_file,
                (output / "timeline.csv").open("w", newline="") as timeline_file,
                (output / "bags.csv").open("w", newline="") as bag_file,
            ):
                detection_writer = csv.DictWriter(detection_file, fieldnames=DETECTION_FIELDS)
                timeline_writer = csv.DictWriter(
                    timeline_file,
                    fieldnames=[
                        "frame",
                        "timestamp_seconds",
                        "detections",
                        "mean_confidence",
                        "active_alerts",
                    ],
                )
                bag_writer = csv.DictWriter(bag_file, fieldnames=BAG_FIELDS)
                for csv_writer in (detection_writer, timeline_writer, bag_writer):
                    csv_writer.writeheader()
                index = 0
                while config.max_frames is None or index < config.max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    # Constant-frame-rate media time; independent of inference/playback speed.
                    timestamp = index / fps
                    kwargs = {
                        "conf": config.confidence,
                        "iou": config.iou,
                        "imgsz": config.image_size,
                        "device": config.device,
                        "classes": list(class_filter) if class_filter is not None else None,
                        "verbose": False,
                    }
                    if ilp_tracker is not None:
                        predict_fn = getattr(model, "predict", model.track)
                        result = predict_fn(frame, **kwargs)[0]
                        detections = result_detections(result)
                        for d in detections:
                            d["track_id"] = None
                        detections = ilp_tracker.update(detections, width, height, timestamp)
                    elif config.tracking:
                        result = model.track(
                            frame, persist=True, tracker="bytetrack.yaml", **kwargs
                        )[0]
                        detections = result_detections(result)
                    else:
                        predict_fn = getattr(model, "predict", model.track)
                        result = predict_fn(frame, **kwargs)[0]
                        detections = result_detections(result)
                    bag_rows = (
                        monitor.update(timestamp, detections, width, height) if monitor else []
                    )
                    for detection in detections:
                        row = {k: v for k, v in detection.items() if k != "bbox"}
                        row.update(
                            dict(zip(["x1", "y1", "x2", "y2"], detection["bbox"], strict=True))
                        )
                        detection_writer.writerow(
                            {"frame": index, "timestamp_seconds": timestamp, **row}
                        )
                    bag_writer.writerows(bag_rows)
                    timeline_writer.writerow(
                        {
                            "frame": index,
                            "timestamp_seconds": timestamp,
                            **analytics.update(detections),
                            "active_alerts": sum(r["status"] == "alert" for r in bag_rows),
                        }
                    )
                    writer.write(annotate(frame, detections, bag_rows, timestamp))
                    if progress and index % 10 == 0:
                        progress(index + 1, min(total, config.max_frames or total))
                    index += 1
            if analytics.frames == 0:
                raise ValueError("The video contains no decodable frames.")
            writer.release()
            capture.release()
            to_browser_video(temporary_video, output / "annotated.mp4")
            temporary_video.unlink()
            events = monitor.events if monitor else []
            (output / "events.json").write_text(json.dumps(events, indent=2))
            summary = analytics.summary(fps, time.perf_counter() - started, config.tracking)
            summary.update(
                {
                    "alert_count": sum(e["event"] == "alert" for e in events),
                    "mlflow_run_id": run.info.run_id,
                    "source_name": source.name,
                    "truncated": config.max_frames is not None and total > analytics.frames,
                    "alert_threshold_seconds": monitor_config.alert_seconds
                    if monitor_config
                    else None,
                }
            )
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in summary.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            )
            mlflow.log_artifacts(str(output), "analysis")
            (output / "COMPLETE").touch()
            if progress:
                progress(analytics.frames, analytics.frames)
            return output
    except Exception as error:
        (output / "FAILED.txt").write_text(str(error))
        raise
    finally:
        capture.release()
        writer.release()
        temporary_video.unlink(missing_ok=True)
