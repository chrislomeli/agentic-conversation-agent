"""utils.py — Shared helpers for the storage layer."""

import os
from pathlib import Path


def resolve_project_root() -> Path:
    """Determine the repository root directory.

    Resolution order:
      1. JOURNAL_AGENT_ROOT env var (explicit override, used by tests)
      2. Walk up from this file until we find pyproject.toml
      3. Fall back to cwd
    """
    configured_root = os.getenv("JOURNAL_AGENT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate

    return Path.cwd().resolve()