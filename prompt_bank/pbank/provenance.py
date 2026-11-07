"""Where a result came from: the code version and the time, recorded in every manifest and run file."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent   # prompt_bank/


def git_commit() -> str:
    """The commit prompt_bank/ is at, with "-dirty" if tracked files under it have uncommitted changes;
    "unknown" outside git. Untracked files are ignored: they are the outputs being written (a new bank, a
    run), not changes to the code that writes them."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE_DIR, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "."], cwd=PACKAGE_DIR,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
