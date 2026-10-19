"""Load repository ``.env`` so ``HF_TOKEN`` and friends are set before Hugging Face calls."""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


def load_repo_dotenv(*, override: bool = False) -> bool:
    """
    Load ``<repo>/.env`` into ``os.environ``.

    Uses ``python-dotenv`` when installed; otherwise applies a small KEY=VALUE parser.
    With ``override=False`` (default), variables already set in the environment win.
    """
    path = _REPO_ROOT / ".env"
    if not path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file_manual(path)
        return True
    load_dotenv(path, override=override)
    return True


def _load_env_file_manual(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val
