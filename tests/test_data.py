from pathlib import Path

import pytest

from vision_lab.data import download
from vision_lab.provenance import sha256


def test_download_uses_hash_to_detect_corruption(tmp_path):
    source = tmp_path / "original"
    source.write_bytes(b"known-data")
    destination = tmp_path / "copied"
    digest = sha256(source)
    assert download(source.as_uri(), destination, digest) == digest
    destination.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(source.as_uri(), destination, digest)


def test_bad_download_is_not_published(tmp_path):
    source = tmp_path / "original"
    source.write_bytes(b"data")
    destination = tmp_path / "copied"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(source.as_uri(), destination, "bad-hash")
    assert not destination.exists()
    assert not Path(str(destination) + ".part").exists()
