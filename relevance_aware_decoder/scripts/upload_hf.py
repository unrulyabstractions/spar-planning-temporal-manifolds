#!/usr/bin/env python
"""Upload a run's tier-1 data to a Hugging Face dataset repo. DRY RUN by default: nothing leaves the machine.

Usage: upload_hf.py [--run qwen3-14b_relevance_s0]                  # dry run: verify checksums, list files + sizes
       upload_hf.py --upload --repo-id USER/NAME [--public]         # create the repo (PRIVATE unless --public) + upload

The upload set is exactly the files listed in runs/<run>/SHA256SUMS (the verified copy-back set: activation subset,
index, meta, manifest, logs, results/, data/prompts_s*), plus that SHA256SUMS file and hf/README.md as the card.
Every listed file is re-hashed first; any mismatch aborts. Full shards (acts_NNNN.safetensors) are never uploaded.
The repo layout mirrors relevance_aware_decoder/, so on a download `sha256sum -c runs/<run>/SHA256SUMS` works as is
and `scripts/evaluate.py ... --acts subset` can read it:
    README.md                          dataset card (hf/README.md)
    runs/<run>/                        index.parquet, meta.json, subset.json, acts_subset_L{LL}_c{k}.safetensors,
                                       manifest.json, SHA256SUMS, SHA256SUMS.shards, pipeline.log, gen_prompts.log
    runs/<run>.capture.log, runs/<run>.evaluate.log
    results/<run>/                     evaluation outputs (summary.md, CSVs, predictions.npz, ...)
    data/prompts_s0.parquet, data/prompts_s0.json, data/prompts_s0_smoke.parquet
Idempotent: files whose remote copy already matches (LFS sha256, or git blob sha1 for small files) are skipped, so an
interrupted upload is resumed by running the same command again. An existing repo's visibility is never changed.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # relevance_aware_decoder/
BATCH_BYTES = 1 << 30                               # ~1 GiB of files per commit
FULL_SHARD = re.compile(r"acts_\d+\.safetensors$")

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="qwen3-14b_relevance_s0")
ap.add_argument("--card", default="hf/README.md")
ap.add_argument("--upload", action="store_true", help="actually create the repo and upload (default: dry run)")
ap.add_argument("--repo-id", help="e.g. USER/ptm-relevance-aware-decoder-qwen3-14b (required with --upload)")
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


# ------------------------------------------------------------------ upload set = SHA256SUMS, verified
sums_file = HERE / "runs" / a.run / "SHA256SUMS"
card = HERE / a.card
if not sums_file.exists():
    sys.exit(f"missing {sums_file} (run the pipeline's checksum step first)")
if not card.exists():
    sys.exit(f"missing dataset card {card}")
listed = [line.split(None, 1) for line in sums_file.read_text().splitlines() if line.strip()]
files, bad = [], []
print(f"verifying {len(listed)} files against {sums_file.relative_to(HERE)} ...")
for digest, rel in listed:
    rel = rel.strip().lstrip("*")
    p = HERE / rel
    if FULL_SHARD.search(rel):
        sys.exit(f"refusing: full shard {rel} is listed in SHA256SUMS (tier 1 never includes full shards)")
    if not p.exists():
        bad.append(f"MISSING {rel}"); continue
    local = sha256(p)
    if local != digest:
        bad.append(f"MISMATCH {rel}"); continue
    files.append((rel, p, local))
if bad:
    print("\n".join(bad)); sys.exit(f"checksum verification FAILED ({len(bad)} of {len(listed)}): nothing uploaded")
print(f"checksums: {len(files)}/{len(listed)} OK")
files.append((f"runs/{a.run}/SHA256SUMS", sums_file, sha256(sums_file)))
files.append(("README.md", card, sha256(card)))
extra = sorted(q.name for q in (HERE / "runs" / a.run).iterdir()
               if q.is_file() and not FULL_SHARD.search(q.name) and f"runs/{a.run}/{q.name}" not in {f[0] for f in files})
if extra:
    print(f"note: in runs/{a.run}/ but not in SHA256SUMS, NOT uploaded: {extra}")

total = sum(p.stat().st_size for _, p, _ in files)
print(f"\n{'repo path':72s} {'MB':>9s}")
for rel, p, _ in files:
    print(f"{rel:72s} {p.stat().st_size / 1e6:9.2f}")
print(f"{len(files)} files, {total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)")
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
                             commit_message=f"tier-1 data of {a.run} (batch {i}/{len(batches)}, {len(ops)} files)")
    print(f"batch {i}/{len(batches)}: {len(ops)} files -> {info.commit_url}")
print(f"done: https://huggingface.co/datasets/{a.repo_id}")
