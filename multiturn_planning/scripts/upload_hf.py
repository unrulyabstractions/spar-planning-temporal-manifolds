#!/usr/bin/env python
"""Upload a run's data to a Hugging Face dataset repo. DRY RUN by default: nothing leaves the machine.

Usage: upload_hf.py [--run qwen3-14b_mtp_s0]                        # dry run: verify checksums, list files + sizes
       upload_hf.py --upload --repo-id USER/NAME [--public]         # create the repo (PRIVATE unless --public) + upload

Upload set (everything the analysis needs; the capture is small enough to publish whole, no subset):
  - every file listed in runs/<run>/SHA256SUMS (the verified copy-back set: activation shards, index, meta, logs),
    re-hashed first; any mismatch aborts. Exception: results/<run>/ entries. The box only ran a quick sanity
    evaluate, and the full analysis was run locally afterwards and overwrote those files, so results/<run>/ is taken
    as it is now, and must be committed and unmodified in git.
  - results/<run>/ (the full local analysis) and runs/<run>.evaluate_local.log (its log)
  - runs/<run>/SHA256SUMS (the copy-back record, kept as is) and runs/<run>/SHA256SUMS.release, written here:
    checksums of every uploaded file, so a download is checked with `sha256sum -c runs/<run>/SHA256SUMS.release`
  - hf/README.md as the dataset card (README.md)
The repo layout mirrors multiturn_planning/, so `scripts/evaluate.py runs/<run>` runs on a download as is.
Idempotent: files whose remote copy already matches (LFS sha256, or git blob sha1 for small files) are skipped, so an
interrupted upload is resumed by running the same command again. An existing repo's visibility is never changed.
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # multiturn_planning/
BATCH_BYTES = 1 << 30                               # ~1 GiB of files per commit

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="qwen3-14b_mtp_s0")
ap.add_argument("--card", default="hf/README.md")
ap.add_argument("--upload", action="store_true", help="actually create the repo and upload (default: dry run)")
ap.add_argument("--repo-id", help="e.g. USER/ptm-multiturn-planning-qwen3-14b (required with --upload)")
ap.add_argument("--public", action="store_true", help="create the repo public (default private; ignored if it exists)")
a = ap.parse_args()
if a.upload and not a.repo_id:
    ap.error("--upload needs --repo-id")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ------------------------------------------------------------------ upload set, verified
run_dir, res_dir = HERE / "runs" / a.run, HERE / "results" / a.run
sums_file, release_file, card = run_dir / "SHA256SUMS", run_dir / "SHA256SUMS.release", HERE / a.card
for p in (sums_file, card, res_dir):
    if not p.exists():
        sys.exit(f"missing {p}")
dirty = subprocess.run(["git", "status", "--porcelain", "--", str(res_dir)], cwd=HERE, capture_output=True,
                       text=True, check=True).stdout.strip()
if dirty:
    sys.exit(f"results/{a.run}/ has uncommitted changes; commit them first:\n{dirty}")

listed = [line.split(None, 1) for line in sums_file.read_text().splitlines() if line.strip()]
files, bad = [], []
print(f"verifying {len(listed)} files against {sums_file.relative_to(HERE)} (results/ taken from git instead) ...")
for digest, rel in listed:
    rel = rel.strip().lstrip("*")
    if rel.startswith("results/"):
        continue
    p = HERE / rel
    if not p.exists():
        bad.append(f"MISSING {rel}"); continue
    local = sha256(p)
    if local != digest:
        bad.append(f"MISMATCH {rel}"); continue
    files.append((rel, p, local))
if bad:
    print("\n".join(bad)); sys.exit(f"checksum verification FAILED ({len(bad)} of {len(listed)}): nothing uploaded")
print(f"checksums: {len(files)} OK")
extra = sorted(res_dir.iterdir()) + [HERE / "runs" / f"{a.run}.evaluate_local.log"]
for p in extra:
    if p.is_file():
        files.append((str(p.relative_to(HERE)), p, sha256(p)))
release_file.write_text("".join(f"{d}  {rel}\n" for rel, _, d in files))
files.append((str(release_file.relative_to(HERE)), release_file, sha256(release_file)))
files.append((str(sums_file.relative_to(HERE)), sums_file, sha256(sums_file)))
files.append(("README.md", card, sha256(card)))
left_out = sorted(q.name for q in run_dir.iterdir() if q.is_file() and str(q.relative_to(HERE)) not in {f[0] for f in files})
if left_out:
    print(f"note: in runs/{a.run}/ but not uploaded: {left_out}")

total = sum(p.stat().st_size for _, p, _ in files)
print(f"\n{'repo path':60s} {'MB':>9s}")
for rel, p, _ in files:
    print(f"{rel:60s} {p.stat().st_size / 1e6:9.2f}")
print(f"{len(files)} files, {total / 1e9:.2f} GB ({total / 2**30:.2f} GiB); wrote {release_file.relative_to(HERE)}")
todos = sorted(set(re.findall(r"TODO\([^)]*\)", card.read_text())))
if todos:
    print(f"\ncard {a.card} still has placeholders: {todos}" + ("" if a.upload else " (fill them before --upload)"))
    if a.upload:
        sys.exit("refusing to upload a card with TODO(...) placeholders")
if not a.upload:
    print("\nDRY RUN: nothing uploaded. Add --upload --repo-id USER/NAME to create the repo (private) and upload.")
    sys.exit(0)

# ------------------------------------------------------------------ upload (explicit --upload only)
from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402

api = HfApi()
print(f"\nlogged in as {api.whoami()['name']}")
existed = api.repo_exists(a.repo_id, repo_type="dataset")
api.create_repo(a.repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
print(f"repo {a.repo_id}: {'exists (visibility unchanged)' if existed else ('created PUBLIC' if a.public else 'created PRIVATE')}")
remote = {}
if existed:
    for i in range(0, len(files), 100):
        for rf in api.get_paths_info(a.repo_id, [f[0] for f in files[i:i + 100]], repo_type="dataset"):
            if hasattr(rf, "blob_id"):
                remote[rf.path] = rf
todo = []
for rel, p, digest in files:
    rf = remote.get(rel)
    same = rf is not None and (rf.lfs.sha256 == digest if rf.lfs else rf.blob_id == git_blob_sha1(p))
    if not same:
        todo.append((rel, p))
print(f"{len(files) - len(todo)} files already up to date, {len(todo)} to upload")
batches, cur, size = [], [], 0
for rel, p in sorted(todo, key=lambda f: f[1].stat().st_size):          # small metadata files land first
    if cur and size + p.stat().st_size > BATCH_BYTES:
        batches.append(cur); cur, size = [], 0
    cur.append((rel, p)); size += p.stat().st_size
if cur:
    batches.append(cur)
for i, batch in enumerate(batches, 1):
    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p)) for rel, p in batch]
    info = api.create_commit(a.repo_id, ops, repo_type="dataset",
                             commit_message=f"data of {a.run} (batch {i}/{len(batches)}, {len(ops)} files)")
    print(f"batch {i}/{len(batches)}: {len(ops)} files -> {info.commit_url}")
print(f"done: https://huggingface.co/datasets/{a.repo_id}")
