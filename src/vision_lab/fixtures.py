"""Explicitly synthetic video fixtures; these are not surveillance benchmarks."""

import json
from pathlib import Path

import cv2

from vision_lab.media import to_browser_video
from vision_lab.provenance import sha256


def make_timer_fixture(root: Path) -> Path:
    """A repeated COCO suitcase crop tests 300 seconds through real YOLO inference.

    No timestamps are accelerated or injected. The video has 310 frames at 1 fps.
    It validates detector → tracker → timer → alert plumbing, not owner association.
    """
    source = root / "data/datasets/coco128/images/train2017/000000000572.jpg"
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError("COCO128 image missing; run vision prepare first.")
    crop = image[365:610, 280:427]
    frame = cv2.copyMakeBorder(crop, 60, 55, 247, 246, cv2.BORDER_CONSTANT, value=(28, 23, 20))
    # Pad to even dimensions for browser encoding.
    frame = cv2.copyMakeBorder(
        frame, 0, frame.shape[0] % 2, 0, frame.shape[1] % 2, cv2.BORDER_CONSTANT, value=(28, 23, 20)
    )
    cv2.putText(
        frame,
        "SYNTHETIC TIMER TEST",
        (12, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    video_dir = root / "data/videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    temporary = video_dir / "timer-raw.mp4"
    target = video_dir / "Synthetic_Timer_310s.mp4"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 1, (frame.shape[1], frame.shape[0])
    )
    if not writer.isOpened():
        raise RuntimeError("Could not encode timer fixture")
    try:
        for _ in range(310):
            writer.write(frame)
        writer.release()
        to_browser_video(temporary, target)
    finally:
        writer.release()
        temporary.unlink(missing_ok=True)
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "synthetic still-image timer fixture",
                "source": "COCO train2017 image 000000000572 via COCO128",
                "source_sha256": sha256(source),
                "crop_xyxy": [280, 365, 427, 610],
                "fps": 1,
                "frames": 310,
                "expected": "alert after 3s stationary + 300s unattended",
                "limitations": "Repeated photograph; not an abandonment or ownership benchmark.",
            },
            indent=2,
        )
    )
    return target
