"""Command line entry points, sharing the dashboard pipeline."""

import argparse
import json
from pathlib import Path

from vision_lab.config import InferenceConfig, workspace
from vision_lab.unattended import MonitorConfig


def main():
    parser = argparse.ArgumentParser(description="Vision Lab: unattended luggage video analytics")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare", help="Download checksum-verified datasets, videos and YOLO weights"
    )
    commands.add_parser("fixture", help="Generate a clearly synthetic 310-second timer test video")
    infer = commands.add_parser("infer", help="Analyze a video and save MLflow artifacts")
    infer.add_argument("video", type=Path)
    infer.add_argument("--model", default="models/yolo11n.pt")
    infer.add_argument("--confidence", type=float, default=0.3)
    infer.add_argument("--image-size", type=int, default=640)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--max-frames", type=int)
    infer.add_argument("--alert-seconds", type=float, default=300)
    infer.add_argument("--stationary-seconds", type=float, default=3)
    infer.add_argument("--proximity-ratio", type=float, default=0.65)
    infer.add_argument(
        "--tracker",
        choices=["ilp", "bytetrack"],
        default="ilp",
        help="Tracker type: ilp (Integer Linear Programming) or bytetrack",
    )
    extend = commands.add_parser(
        "extend", help="Extend a surveillance video to a longer test duration"
    )
    extend.add_argument("source", type=Path, help="Input video path")
    extend.add_argument("--seconds", type=float, default=60.0, help="Target duration in seconds")
    extend.add_argument("--output", type=Path, default=None, help="Output video path")
    extend.add_argument(
        "--with-pickup", action="store_true", help="Include return-and-pickup sequence"
    )
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, default=Path("data/datasets/coco8.yaml"))
        command.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
        command.add_argument("--image-size", type=int, default=320)
        command.add_argument("--device", default="cpu")
        if name == "train":
            command.add_argument("--epochs", type=int, default=1)
            command.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        from vision_lab.data import prepare_assets

        manifest = workspace() / "configs/assets.lock.json"
        if not manifest.exists():
            manifest = Path("configs/assets.lock.json")
        print(prepare_assets(workspace(), manifest))
    elif args.command == "fixture":
        from vision_lab.fixtures import make_timer_fixture

        print(make_timer_fixture(workspace()))
    elif args.command == "infer":
        from vision_lab.pipeline import process_video

        output = process_video(
            args.video,
            InferenceConfig(
                model=args.model,
                confidence=args.confidence,
                image_size=args.image_size,
                device=args.device,
                tracker_type=args.tracker,
                max_frames=args.max_frames,
            ),
            MonitorConfig(
                alert_seconds=args.alert_seconds,
                stationary_seconds=args.stationary_seconds,
                proximity_ratio=args.proximity_ratio,
            ),
        )
        print(f"Analysis saved: {output}")
        print((output / "summary.json").read_text())
    elif args.command == "train":
        from vision_lab.training import train

        print(train(args.data, args.model, args.epochs, args.image_size, args.batch, args.device))
    elif args.command == "evaluate":
        from vision_lab.training import evaluate

        print(json.dumps(evaluate(args.data, args.model, args.image_size, args.device), indent=2))
    elif args.command == "extend":
        from vision_lab.generators import extend_surveillance_video

        target = args.output or args.source.parent / f"{args.source.stem}_{int(args.seconds)}s.mp4"
        out = extend_surveillance_video(args.source, target, args.seconds, args.with_pickup)
        print(f"Generated extended video: {out}")


if __name__ == "__main__":
    main()
