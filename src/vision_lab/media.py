"""Produce browser-compatible H.264 video with the bundled FFmpeg executable."""

import subprocess
from pathlib import Path

import imageio_ffmpeg


def to_browser_video(source: Path, destination: Path, seconds: float | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(source)]
    if seconds is not None:
        command += ["-t", str(seconds)]
    command += [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    if result.returncode:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Video conversion failed: {result.stderr[-2000:]}")
    return destination
