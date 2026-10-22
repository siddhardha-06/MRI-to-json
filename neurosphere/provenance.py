"""Provenance block attached to every JSON and every results file."""
from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from . import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]

_PACKAGES = ("numpy", "scipy", "nibabel", "torch", "antspyx", "scikit-learn", "scikit-image", "pandas", "pingouin")


def git_commit() -> dict:
    """Current commit and whether the working tree has uncommitted changes to tracked files."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "sha_reason": "git not available", "dirty": None, "dirty_reason": "git not available"}


def software_versions(extra: tuple[str, ...] = ()) -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in _PACKAGES + tuple(extra):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance(**extra) -> dict:
    """Standard provenance dict. Pass experiment-specific fields (seed, data_subset, ...) as keywords."""
    block = {
        "pipeline_version": __version__,
        "git_commit": git_commit(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "software": software_versions(),
    }
    block.update(extra)
    return block
