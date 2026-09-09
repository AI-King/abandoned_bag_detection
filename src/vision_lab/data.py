"""Fetch public assets with a version-controlled checksum manifest."""

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

from vision_lab.media import to_browser_video
from vision_lab.provenance import sha256

SOURCES = {
    "coco128.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
    "coco8.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
    "LeftBag.mpg": "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/LeftBag/LeftBag.mpg",
    "lb1gt.xml": "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/LeftBag/lb1gt.xml",
    "LeftBag_PickedUp.mpg": (
        "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/LeftBag_PickedUp/LeftBag_PickedUp.mpg"
    ),
    "lbpugt.xml": "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/LeftBag_PickedUp/lbpugt.xml",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}


def download(url: str, destination: Path, expected_hash: str | None = None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "VisionLab/0.1"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            if expected_hash and sha256(temporary) != expected_hash:
                raise ValueError(f"Checksum mismatch for {destination.name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    digest = sha256(destination)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Checksum mismatch for {destination.name}; remove it and fetch again.")
    return digest


def prepare_assets(root: Path, manifest_path: Path, record_hashes: bool = False) -> Path:
    import ultralytics

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not record_hashes and not manifest:
        raise ValueError("Asset checksum manifest is missing.")
    for filename, url in SOURCES.items():
        destination = root / ("models" if filename.endswith(".pt") else "data/raw") / filename
        expected = manifest.get(filename, {}).get("sha256")
        if not expected and not record_hashes:
            raise ValueError(f"No pinned checksum for {filename}")
        print(f"Preparing {filename}", flush=True)
        digest = download(url, destination, expected)
        manifest[filename] = {"url": url, "sha256": digest}
    if record_hashes:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    dataset_parent = root / "data/datasets"
    dataset_parent.mkdir(parents=True, exist_ok=True)
    for dataset_name in ("coco8", "coco128"):
        with zipfile.ZipFile(root / f"data/raw/{dataset_name}.zip") as archive:
            for member in archive.infolist():
                target = (dataset_parent / member.filename).resolve()
                if not target.is_relative_to(dataset_parent.resolve()):
                    raise ValueError("Unsafe path in dataset archive")
            archive.extractall(dataset_parent)
    official = Path(ultralytics.__file__).parent / "cfg/datasets/coco8.yaml"
    config = yaml.safe_load(official.read_text(encoding="utf-8"))
    config["path"] = str((dataset_parent / "coco8").resolve())
    config.pop("download", None)
    config_path = dataset_parent / "coco8.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    prepare_luggage_subset(dataset_parent)
    for name in ("LeftBag", "LeftBag_PickedUp"):
        output = root / f"data/videos/{name}.mp4"
        if not output.exists():
            to_browser_video(root / f"data/raw/{name}.mpg", output)
    return config_path


def prepare_luggage_subset(dataset_parent: Path) -> Path:
    """Remap person + bag classes; use disjoint image IDs instead of COCO128's shared split.

    The upstream pretrained weights have already seen COCO train2017, so even this
    disjoint fine-tuning split is a development check, not an independent benchmark.
    """
    from collections import Counter

    mapping = {0: 0, 24: 1, 26: 2, 28: 3}
    source = dataset_parent / "coco128"
    target = dataset_parent / "luggage-mini"
    positive, negative = [], []
    labels = {}
    for image in sorted((source / "images/train2017").glob("*.jpg")):
        label_file = source / "labels/train2017" / f"{image.stem}.txt"
        lines = label_file.read_text().splitlines() if label_file.exists() else []
        selected = []
        has_bag = False
        for line in lines:
            parts = line.split()
            if parts and int(parts[0]) in mapping:
                original = int(parts[0])
                has_bag |= original != 0
                selected.append(f"{mapping[original]} " + " ".join(parts[1:]))
        labels[image.name] = selected
        (positive if has_bag else negative).append(image)
    split_images = {"train": [], "val": []}
    # Distribute positive and negative images separately, deterministically.
    for group in (positive, negative):
        for index, image in enumerate(group):
            split_images["val" if index % 4 == 0 else "train"].append(image)
    inventory = {}
    for split, images in split_images.items():
        counts = Counter()
        for image in images:
            image_dir = target / "images" / split
            label_dir = target / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_dir / image.name)
            rows = labels[image.name]
            (label_dir / f"{image.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            counts.update(int(row.split()[0]) for row in rows)
        inventory[split] = {
            "images": [image.name for image in images],
            "class_instances": dict(counts),
        }
    (target / "split_manifest.json").write_text(json.dumps(inventory, indent=2))
    yaml_path = dataset_parent / "luggage-mini.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(target.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person", 1: "backpack", 2: "handbag", 3: "suitcase"},
            }
        )
    )
    return yaml_path
