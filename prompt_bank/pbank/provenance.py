"""Where a result came from: the code version and the time, recorded in every manifest and run file."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent   # prompt_bank/


def git_commit(*code_dirs: Path) -> str:
    """The commit the repository is at, with "-dirty" if tracked files under any of `code_dirs` (default:
    prompt_bank/) have uncommitted changes; "unknown" outside git. Untracked files are ignored: they are the
    outputs being written (a new bank, a run), not changes to the code that writes them. A caller whose results
    depend on code outside prompt_bank/ (steering/ imports pbank) passes every directory it depends on."""
    dirs = [Path(d).resolve() for d in (code_dirs or (PACKAGE_DIR,))]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dirs[0], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *map(str, dirs)],
                               cwd=dirs[0], capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
