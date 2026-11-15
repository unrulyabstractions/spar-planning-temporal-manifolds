"""Upload steering outputs to a Hugging Face dataset repository. DRY RUN by default: nothing leaves the machine.

    python tools/upload_hf.py ../prompt_bank/runs/gemma-4-31B-it__t0_s0 activations/gemma-4-31B-it__t0_s0 \\
        fits/gemma-4-31B-it__t0_s0 methods/gemma-4-31B-it                     # dry run: files, sizes, checksums
    python tools/upload_hf.py ... --upload --repo-id ORG/NAME [--prefix PATH] [--public]

Destination names mirror the repository and every upload carries SHA256SUMS (tools/upload_common.py). As in the
team's earlier Hugging Face uploads (multiturn_planning/scripts/upload_hf.py): a new repo is created PRIVATE unless
--public is given; an existing repo's visibility is never changed; authentication is huggingface_hub's own (HF_TOKEN,
or a prior `hf auth login`), and no token is read or printed here. Unlike that script, nothing that already exists at
the destination is overwritten or skipped: if any destination path exists the upload is refused, so a published
method can't be silently replaced. After the commit, every file's remote LFS SHA-256 (or git blob SHA-1, for small
files) is compared with the local file.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_common import REPO_ROOT, collect, sha256sums, summary  # noqa: E402

BATCH_BYTES = 1 << 30


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def remote_matches(info, local: Path, sha256: str) -> bool:
    lfs = getattr(info, "lfs", None)
    return (lfs.sha256 == sha256) if lfs else getattr(info, "blob_id", None) == git_blob_sha1(local)


def main(argv: list[str] | None = None, api=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="files or directories to upload")
    p.add_argument("--root", type=Path, default=REPO_ROOT, help="destination names are paths relative to this")
    p.add_argument("--upload", action="store_true", help="actually upload (default: dry run)")
    p.add_argument("--repo-id", help="ORG/NAME (required with --upload)")
    p.add_argument("--prefix", default="", help="path inside the repo to upload under (default: its root)")
    p.add_argument("--public", action="store_true", help="create a new repo public (default private)")
    args = p.parse_args(argv)
    files = collect(args.paths, args.root)
    prefix = args.prefix.strip("/")
    under = (lambda rel: f"{prefix}/{rel}") if prefix else (lambda rel: rel)
    print(f"{len(files)} files to upload, named relative to {args.root.resolve()}:\n{summary(files)}")
    if not args.upload:
        print("\nDRY RUN: nothing uploaded. Add --upload --repo-id ORG/NAME to upload (a new repo is private).")
        return 0
    if not args.repo_id:
        p.error("--upload needs --repo-id")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    from huggingface_hub import CommitOperationAdd
    print(f"logged in as {api.whoami()['name']}")
    existed = api.repo_exists(args.repo_id, repo_type="dataset")
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    print(f"repo {args.repo_id}: " + ("exists (visibility unchanged)" if existed else
                                       f"created {'PUBLIC' if args.public else 'PRIVATE'}"))
    dests = [under(f.dest) for f in files] + [under("SHA256SUMS")]
    if existed:
        present = [i.path for k in range(0, len(dests), 100)
                   for i in api.get_paths_info(args.repo_id, dests[k:k + 100], repo_type="dataset")]
        if present:
            sys.exit(f"{len(present)} destination paths already exist in {args.repo_id}, e.g. {present[:3]}; uploads "
                     f"never overwrite. Choose another --prefix.")
    with tempfile.TemporaryDirectory() as tmp:
        sums = Path(tmp) / "SHA256SUMS"
        sums.write_text(sha256sums(files))
        todo = [(under(f.dest), f.local) for f in sorted(files, key=lambda f: f.size)] + [(under("SHA256SUMS"), sums)]
        batches, cur, size = [], [], 0
        for dest, local in todo:
            if cur and size + local.stat().st_size > BATCH_BYTES:
                batches.append(cur)
                cur, size = [], 0
            cur.append((dest, local))
            size += local.stat().st_size
        batches.append(cur)
        for i, batch in enumerate(batches, 1):
            ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(local)) for dest, local in batch]
            info = api.create_commit(args.repo_id, ops, repo_type="dataset",
                                     commit_message=f"steering outputs (batch {i}/{len(batches)}, {len(ops)} files)")
            print(f"batch {i}/{len(batches)}: {len(ops)} files -> {getattr(info, 'commit_url', info)}", flush=True)
        expected = {under(f.dest): (f.local, f.sha256) for f in files}
        expected[under("SHA256SUMS")] = (sums, hashlib.sha256(sums.read_bytes()).hexdigest())
        remote = {i.path: i for k in range(0, len(dests), 100)
                  for i in api.get_paths_info(args.repo_id, dests[k:k + 100], repo_type="dataset")}
        bad = [d for d, (local, digest) in expected.items() if d not in remote or not remote_matches(remote[d], local, digest)]
    print(f"verified {len(expected) - len(bad)}/{len(expected)} files against their remote hashes")
    if bad:
        sys.exit(f"upload verification FAILED for {len(bad)} files, e.g. {bad[:3]}")
    print(f"uploaded and verified -> https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    main()
