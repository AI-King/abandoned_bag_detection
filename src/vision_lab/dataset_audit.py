"""Validate directory-based YOLO datasets and fingerprint the exact training inputs."""

from pathlib import Path

import yaml

from vision_lab.provenance import sha256


def audit_dataset(yaml_path: Path) -> dict:
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = config["names"]
    count = len(names)
    base = Path(config.get("path", yaml_path.parent))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    audit = {"yaml_sha256": sha256(yaml_path), "names": names, "splits": {}}
    seen_images = set()
    for split in ("train", "val"):
        folder = base / config[split]
        if not folder.is_dir():
            raise ValueError(f"{split} must point to an image directory: {folder}")
        images = sorted(
            p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not images:
            raise ValueError(f"Empty {split} image directory")
        rows = []
        for image in images:
            digest = sha256(image)
            if digest in seen_images:
                raise ValueError(f"Duplicate image within or across dataset splits: {image.name}")
            seen_images.add(digest)
            relative = image.relative_to(base)
            parts = list(relative.parts)
            if "images" not in parts:
                raise ValueError("Use the standard images/ and labels/ YOLO directory structure")
            parts[parts.index("images")] = "labels"
            label = base.joinpath(*parts).with_suffix(".txt")
            lines = label.read_text().splitlines() if label.exists() else []
            for line in lines:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Expected class x y width height in {label.name}")
                class_id, *coordinates = fields
                x, y, width, height = map(float, coordinates)
                if not (
                    0 <= int(class_id) < count
                    and 0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                ):
                    raise ValueError(f"Invalid class or normalized box in {label.name}")
            rows.append(
                {
                    "image": str(relative),
                    "image_sha256": digest,
                    "label_sha256": sha256(label) if label.exists() else None,
                    "instances": len(lines),
                }
            )
        audit["splits"][split] = rows
    return audit
