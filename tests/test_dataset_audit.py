import pytest
import yaml

from vision_lab.dataset_audit import audit_dataset


def build_dataset(tmp_path):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        (tmp_path / "images" / split / "one.jpg").write_bytes(split.encode())
        (tmp_path / "labels" / split / "one.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    path = tmp_path / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": ["bag"]}
        )
    )
    return path


def test_exact_content_is_fingerprinted(tmp_path):
    manifest = audit_dataset(build_dataset(tmp_path))
    assert manifest["splits"]["train"][0]["instances"] == 1
    assert len(manifest["splits"]["train"][0]["image_sha256"]) == 64


def test_train_val_leakage_rejected(tmp_path):
    path = build_dataset(tmp_path)
    (tmp_path / "images/val/one.jpg").write_bytes(b"train")
    with pytest.raises(ValueError, match="Duplicate"):
        audit_dataset(path)


def test_bad_label_rejected(tmp_path):
    path = build_dataset(tmp_path)
    (tmp_path / "labels/val/one.txt").write_text("0 0.5 0.5 -0.3 0.4\n")
    with pytest.raises(ValueError, match="Invalid"):
        audit_dataset(path)
