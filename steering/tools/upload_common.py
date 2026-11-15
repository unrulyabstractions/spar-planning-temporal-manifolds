"""What upload_hf.py and upload_gcs.py share: which files go up, under which names, with which checksums.

Destination names mirror the repository: a file at <repo>/steering/methods/X/method.json is uploaded as
<prefix>/steering/methods/X/method.json, so a download unpacked at a checkout's root sits where the code expects it.
Every upload also carries <prefix>/SHA256SUMS (one "sha256  path" line per file, paths relative to <prefix>), so a
download is checked with `cd <download>/<prefix> && sha256sum -c SHA256SUMS`.

Refused, so that an upload can't publish what it shouldn't:
  * paths outside --root (default: the repository root);
  * files that look like credentials: names starting with ".env", ending in .pem / .key / .p12, or containing
    "credential", "secret" or "token"; anything inside a .git directory;
  * symbolic links (their target may lie anywhere).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET_NAME = re.compile(r"^\.env|\.(pem|key|p12)$|credential|secret|token", re.IGNORECASE)


@dataclass(frozen=True)
class UploadFile:
    dest: str          # path under the destination prefix
    local: Path
    size: int
    sha256: str


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(paths: list[Path], root: Path = REPO_ROOT) -> list[UploadFile]:
    """Every file under `paths` (files or directories), named by its path relative to `root`; refuses as above."""
    root = Path(root).resolve()
    files: dict[str, UploadFile] = {}
    for given in paths:
        given = Path(given)
        if given.is_symlink():
            raise ValueError(f"{given} is a symbolic link; upload its target instead")
        given = given.resolve()
        if not given.exists():
            raise FileNotFoundError(given)
        if root != given and root not in given.parents:
            raise ValueError(f"{given} is outside --root {root}")
        for f in sorted([given] if given.is_file() else given.rglob("*")):
            if f.is_symlink():
                raise ValueError(f"{f} is a symbolic link; refusing to follow it")
            if not f.is_file():
                continue
            rel = f.relative_to(root)
            if ".git" in rel.parts or any(SECRET_NAME.search(part) for part in rel.parts):
                raise ValueError(f"{rel} looks like a credential or repository internals; refusing to upload it")
            files[rel.as_posix()] = UploadFile(rel.as_posix(), f, f.stat().st_size, sha256_of(f))
    if not files:
        raise ValueError("nothing to upload")
    return list(files.values())


def sha256sums(files: list[UploadFile]) -> str:
    """The SHA256SUMS text for `files` (sha256sum -c format)."""
    return "".join(f"{f.sha256}  {f.dest}\n" for f in sorted(files, key=lambda f: f.dest))


def summary(files: list[UploadFile]) -> str:
    total = sum(f.size for f in files)
    tops: dict[str, list[int]] = {}
    for f in files:
        top = "/".join(f.dest.split("/")[:3])
        tops.setdefault(top, [0, 0])
        tops[top][0] += 1
        tops[top][1] += f.size
    lines = [f"  {n:5d} files {size / 1e6:10.1f} MB  {top}" for top, (n, size) in sorted(tops.items())]
    return "\n".join(lines + [f"  {len(files):5d} files {total / 1e6:10.1f} MB  total"])
