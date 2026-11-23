"""Where a result came from: the code version and the time, recorded in every manifest and run file.

The code version is the git commit. On a machine the code was copied to without its `.git` directory (a rented GPU
box, after an rsync that leaves dotfiles out), git cannot say. For that case, stamp the checkout before shipping it:

    python -m pbank.provenance stamp        # in the git checkout; writes CODE_STAMP.json at the repository root

The stamp holds the commit and the SHA-256 of every tracked file, except dotfiles and files under dot-directories
(`.gitignore` and the like): such a copy does not carry them, and they are not code. Without git, `git_commit` reads it and re-hashes
the files of the directories it is asked about: if they all match, the recorded version is the stamped commit; if any
differs or is missing, it is the commit with "-modified", so code edited on the box cannot pass as the commit. With
git available the stamp is ignored. CODE_STAMP.json is git-ignored and is not a dotfile, so the copy carries it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent   # prompt_bank/
STAMP_NAME = "CODE_STAMP.json"


def git_commit(*code_dirs: Path) -> str:
    """The commit the repository is at, with "-dirty" if tracked files under any of `code_dirs` (default:
    prompt_bank/) have uncommitted changes. Untracked files are ignored: they are the outputs being written (a new
    bank, a run), not changes to the code that writes them. A caller whose results depend on code outside
    prompt_bank/ (steering/ imports pbank) passes every directory it depends on.

    Outside git: the stamped commit if the repository carries a CODE_STAMP.json (module docstring), else "unknown"."""
    dirs = [Path(d).resolve() for d in (code_dirs or (PACKAGE_DIR,))]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dirs[0], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *map(str, dirs)],
                               cwd=dirs[0], capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return stamped_commit(dirs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_stamp(start: Path):
    """The CODE_STAMP.json of the checkout containing `start`, or None."""
    for d in (start, *start.parents):
        if (d / STAMP_NAME).is_file():
            return d / STAMP_NAME
    return None


def stamped_commit(dirs: list[Path]) -> str:
    """The version to record without git: the stamp's commit for `dirs`, "-modified" if their stamped files changed."""
    stamp_path = find_stamp(dirs[0])
    if stamp_path is None:
        return "unknown"
    stamp, root = json.loads(stamp_path.read_text()), stamp_path.parent
    relevant = {name: digest for name, digest in stamp["files"].items()
                if any((root / name).resolve().is_relative_to(d) for d in dirs)}
    if not relevant:
        return "unknown"            # a stamp that says nothing about these directories vouches for nothing
    same = all((root / name).is_file() and _sha256(root / name) == digest for name, digest in relevant.items())
    # The stamp's own "-dirty" refers to the whole checkout at stamping time; it is kept as written.
    return stamp["commit"] + ("" if same else "-modified")


def write_stamp(repo: Path) -> Path:
    """Write CODE_STAMP.json for the git checkout at `repo`: its commit ("-dirty" if any tracked file has
    uncommitted changes) and the SHA-256 of every tracked file as it is on disk now, dotfiles left out (module
    docstring)."""
    repo = Path(repo).resolve()
    run = lambda *args: subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout
    commit = run("rev-parse", "HEAD").strip()
    dirty = run("status", "--porcelain", "--untracked-files=no").strip()
    names = [n for n in run("ls-files", "-z").split("\0")
             if n and (repo / n).is_file() and not any(part.startswith(".") for part in n.split("/"))]
    stamp = {"commit": commit + ("-dirty" if dirty else ""), "stamped_utc": utc_now(),
             "files": {n: _sha256(repo / n) for n in sorted(names)}}
    (repo / STAMP_NAME).write_text(json.dumps(stamp, indent=1) + "\n")
    return repo / STAMP_NAME


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    if sys.argv[1:] == ["stamp"]:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=PACKAGE_DIR, capture_output=True, text=True)
        if top.returncode:
            sys.exit("not a git checkout: stamp the code where its .git is, before copying it")
        path = write_stamp(Path(top.stdout.strip()))
        stamp = json.loads(path.read_text())
        print(f"wrote {path}: commit {stamp['commit']}, {len(stamp['files'])} tracked files")
    elif sys.argv[1:] == ["show"]:
        print(git_commit())
    else:
        sys.exit("usage: python -m pbank.provenance stamp | show")
