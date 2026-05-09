"""Minimal environment-file helpers for local development."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path, *, override: bool = False) -> bool:
    """Load simple KEY=VALUE pairs from a dotenv-style file."""
    if not path.exists():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value

    return True


def load_project_env(*, override: bool = False) -> bool:
    """Load repo-root .env if present."""
    project_root = Path(__file__).resolve().parent.parent
    return load_dotenv(project_root / ".env", override=override)
