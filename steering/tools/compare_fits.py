"""Compare two fit output trees exactly, e.g. before and after a change that should not change behaviour.

    python tools/compare_fits.py <tree A> <tree B> [--ignore-field FIELD ...]

A tree is any directory holding fit outputs (fits/<activations>/ or one L<layer>_<site>/ below it). Every file must
exist in both trees and be identical:
  * .npz: the same array names, and every array has the same dtype, shape and values (np.array_equal; no tolerance);
  * .json: equal after removing the top-level fields that record when and from which commit a file was written
    (`created_utc`, `code_commit`) and any --ignore-field;
  * anything else: byte for byte.
Prints one line per file and exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROVENANCE_FIELDS = ("created_utc", "code_commit")


def compare_file(a: Path, b: Path, ignore: tuple[str, ...]) -> str | None:
    """None if identical, else a description of the first difference."""
    if a.suffix == ".npz":
        za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
        if sorted(za.files) != sorted(zb.files):
            return f"array names differ: {sorted(za.files)} vs {sorted(zb.files)}"
        for name in za.files:
            x, y = za[name], zb[name]
            if x.dtype != y.dtype or x.shape != y.shape:
                return f"{name}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}"
            if not np.array_equal(x, y):
                diff = np.abs(x.astype(np.float64) - y.astype(np.float64)) if x.dtype.kind in "fc" else None
                return f"{name}: values differ" + (f" (max |diff| {diff.max():.3e})" if diff is not None else "")
        return None
    if a.suffix == ".json":
        ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
        for field in PROVENANCE_FIELDS + ignore:
            ja.pop(field, None), jb.pop(field, None)
        if ja != jb:
            keys = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
            return f"fields differ: {keys}"
        return None
    return None if a.read_bytes() == b.read_bytes() else "bytes differ"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--ignore-field", action="append", default=[], help="top-level JSON field to ignore (repeatable)")
    args = p.parse_args()
    files_a = {f.relative_to(args.a) for f in args.a.rglob("*") if f.is_file()}
    files_b = {f.relative_to(args.b) for f in args.b.rglob("*") if f.is_file()}
    failed = False
    for rel in sorted(files_a | files_b):
        if rel not in files_a or rel not in files_b:
            print(f"MISSING  {rel} (only in {'A' if rel in files_a else 'B'})")
            failed = True
            continue
        problem = compare_file(args.a / rel, args.b / rel, tuple(args.ignore_field))
        print(f"{'DIFFER ' if problem else 'same   '}  {rel}" + (f": {problem}" if problem else ""))
        failed |= problem is not None
    print(f"{len(files_a | files_b)} files, {'DIFFERENCES FOUND' if failed else 'all identical'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
