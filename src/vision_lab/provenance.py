"""Content hashes and source revision for reproducibility."""

import hashlib
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, str]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"git.commit": revision, "git.dirty": str(bool(dirty)).lower()}
    except (OSError, subprocess.CalledProcessError):
        return {"git.commit": "unknown", "git.dirty": "unknown"}
