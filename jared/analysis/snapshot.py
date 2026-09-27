"""Freeze the derived data of a results run into a named snapshot for later
comparison. Copies what the explorer notebook reads (PCA coordinates,
dimensionality sweep, cross-mode transfer, every manifest and figure) into
`jared/snapshots/<name>/` with a provenance file. The 1 GB
activation files are NOT copied; they stay under results/ and can be
regenerated on the box.

    python analysis/snapshot.py results/horizon_v1_noprefill v1_noprefill --note "before the prefill fix"
    python analysis/snapshot.py results/horizon v2_prefill --note "prefill 'I choose:', 3072 think budget"
    python analysis/snapshot.py --list
"""

import argparse
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOTS = HERE.parent / "snapshots"
DERIVED = ["pca3.json", "feature_dimensionality.json", "cross_mode_transfer.json",
           "cross_mode_transfer.png", "feature_dimensionality.png"]


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def take(root, name, note=""):
    root = Path(root)
    dst = SNAPSHOTS / name
    if dst.exists():
        raise SystemExit(f"snapshot {name!r} already exists at {dst}; pick a new name")
    dst.mkdir(parents=True)
    copied = []
    for f in DERIVED:
        src = root / "analysis" / f
        if not src.exists():
            continue
        if f.endswith(".json") and src.stat().st_size > 1_000_000:
            with open(src, "rb") as fi, gzip.open(dst / (f + ".gz"), "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
            copied.append(f + ".gz")
        else:
            shutil.copy2(src, dst / f)
            copied.append(f)
    runs = {}
    for d in sorted(root.glob("exp*_*")):
        m = d / "manifest.json"
        if not m.exists():
            continue
        (dst / d.name).mkdir()
        shutil.copy2(m, dst / d.name / "manifest.json")
        for extra in ("answers.json", "perturbation.csv"):
            if (d / extra).exists():
                shutil.copy2(d / extra, dst / d.name / extra)
        for png in d.glob("*.png"):
            shutil.copy2(png, dst / d.name / png.name)
        for csv in d.glob("transfer_*.csv"):
            shutil.copy2(csv, dst / d.name / csv.name)
        mf = json.loads(m.read_text())
        runs[d.name] = {"settings": mf.get("settings"), "git": mf.get("git"), "started": mf.get("started"),
                        "wall_seconds": mf.get("wall_seconds")}
    prov = {"name": name, "note": note, "source": str(root.resolve()), "taken": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "analysis_git": git_hash(), "derived": copied, "runs": runs}
    (dst / "snapshot.json").write_text(json.dumps(prov, indent=1))
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e6
    print(f"snapshot {name!r}: {len(copied)} derived files, {len(runs)} runs, {size:.1f} MB -> {dst}")
    return dst


def load_json(name, fname):
    """Read a derived JSON from a snapshot, gz or plain."""
    d = SNAPSHOTS / name
    gz, plain = d / (fname + ".gz"), d / fname
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return json.load(f)
    if plain.exists():
        return json.loads(plain.read_text())
    raise FileNotFoundError(f"{fname} not in snapshot {name!r}")


def list_snapshots():
    rows = []
    for d in sorted(SNAPSHOTS.glob("*/snapshot.json")):
        p = json.loads(d.read_text())
        rows.append((p["name"], p["taken"], p.get("note", ""), ", ".join(sorted(p["runs"]))))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.root:
        for name, taken, note, runs in list_snapshots():
            print(f"{name:20} {taken}  {note}\n{'':20} runs: {runs}")
    else:
        take(a.root, a.name, a.note)
