"""Utilities to generate and extend realistic surveillance test videos.

Enables creating long test videos from short CCTV / AI clips by seamlessly
extending the stationary unattended period with realistic sensor noise, or
generating return-and-pickup scenarios.
"""

from pathlib import Path

import cv2
import numpy as np


def extend_surveillance_video(
    source: Path,
    target: Path,
    duration_seconds: float = 60.0,
    with_pickup: bool = False,
    sensor_noise_sigma: float = 0.4,
) -> Path:
    """Extend a video clip where a person leaves an object to any desired duration."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("Video contains no readable frames.")

    target.parent.mkdir(parents=True, exist_ok=True)
    out_frames = list(frames)

    # Use the last second of footage as the stationary base
    tail_len = max(int(fps), 10)
    tail = frames[-tail_len:]

    if not with_pickup:
        needed = max(0, int(fps * duration_seconds) - len(out_frames))
        for i in range(needed):
            idx = i % (len(tail) * 2)
            if idx >= len(tail):
                idx = 2 * len(tail) - 1 - idx
            f = tail[idx].copy()
            if sensor_noise_sigma > 0:
                noise = np.random.normal(0, sensor_noise_sigma, f.shape).astype(np.float32)
                f = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            out_frames.append(f)
    else:
        # Stationary unattended period
        unattended_frames = int(fps * max(duration_seconds - len(frames) / fps, 10.0))
        for i in range(unattended_frames):
            idx = i % (len(tail) * 2)
            if idx >= len(tail):
                idx = 2 * len(tail) - 1 - idx
            f = tail[idx].copy()
            if sensor_noise_sigma > 0:
                noise = np.random.normal(0, sensor_noise_sigma, f.shape).astype(np.float32)
                f = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            out_frames.append(f)

        # Reverse departure to simulate person returning
        split_idx = int(len(frames) * 0.7)
        return_seq = frames[split_idx:][::-1]
        out_frames.extend(return_seq)

        # Reverse arrival to simulate picking up and walking away
        pickup_seq = frames[:split_idx][::-1]
        out_frames.extend(pickup_seq)

    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in out_frames:
        writer.write(f)
    writer.release()

    return target
