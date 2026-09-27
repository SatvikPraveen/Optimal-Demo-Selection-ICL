"""
Environment manifest recorded alongside every result file, so a number in
a table can always be traced back to the code and libraries that produced it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

_PACKAGES = (
    "torch",
    "transformers",
    "sentence-transformers",
    "numpy",
    "scikit-learn",
    "datasets",
    "openai",
)


def git_revision(repo_root: str | Path | None = None) -> dict[str, str | bool | None]:
    """Current commit hash and dirty flag (``None`` when not in a git repo)."""
    try:
        cwd = str(repo_root) if repo_root else None
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": None, "dirty": None}


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_manifest(repo_root: str | Path | None = None) -> dict:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if cuda else None
    except Exception:  # pragma: no cover - torch missing or broken
        cuda, device_name = False, None
    try:
        from src import __version__
    except Exception:  # pragma: no cover
        __version__ = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "git": git_revision(repo_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": cuda,
        "cuda_device": device_name,
        "packages": package_versions(),
    }
