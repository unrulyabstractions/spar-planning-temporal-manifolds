"""API keys from a `.env` file, so they never have to be typed into a command line or written to an output.

The file holds lines of the form `NAME=value` (an optional leading `export`, optional quotes, `#` comments).
It is looked for in prompt_bank/ and then in the repository root; both locations are git-ignored. A variable
that is already set in the environment is left alone, so `ANTHROPIC_API_KEY=... python generate.py` still wins.
Values are never printed or logged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .provenance import PACKAGE_DIR

ENV_FILE_LOCATIONS = (PACKAGE_DIR / ".env", PACKAGE_DIR.parent / ".env")


def load_env_file(locations: tuple[Path, ...] = ENV_FILE_LOCATIONS) -> Optional[Path]:
    """Set environment variables from the first `.env` file found; return its path (None if there is none)."""
    for path in locations:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.removeprefix("export ").partition("=")
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))
        return path
    return None
