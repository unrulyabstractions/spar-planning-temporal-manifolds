"""Upload steering outputs to Google Cloud Storage with rclone. DRY RUN by default: nothing leaves the machine.

    python tools/upload_gcs.py ../prompt_bank/runs/gemma-4-31B-it__t0_s0 activations/gemma-4-31B-it__t0_s0 \\
        fits/gemma-4-31B-it__t0_s0 methods/gemma-4-31B-it                     # dry run: files, sizes, checksums
    python tools/upload_gcs.py ... --upload --dest BUCKET/PREFIX --key KEY.json

Destination names mirror the repository and every upload carries SHA256SUMS (tools/upload_common.py). The transfer
follows the team's existing GCS uploads (phrasing_study/scripts/run_pipeline.sh): rclone with a service-account key
file (--key, or $GCS_KEY; without either, rclone's application-default credentials), refusing a destination prefix that
already has files, then `rclone check --one-way` (MD5) and a file count. The key's path is used, never its contents.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_common import REPO_ROOT, collect, sha256sums, summary  # noqa: E402


def rclone_base(key: str | None) -> list[str]:
    auth = ["--gcs-service-account-file", key] if key else ["--gcs-env-auth"]
    return ["rclone", *auth, "--gcs-bucket-policy-only"]


def main(argv: list[str] | None = None, remote_prefix: str = ":gcs:") -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="files or directories to upload")
    p.add_argument("--root", type=Path, default=REPO_ROOT, help="destination names are paths relative to this")
    p.add_argument("--upload", action="store_true", help="actually upload (default: dry run)")
    p.add_argument("--dest", help="BUCKET/PREFIX (required with --upload)")
    p.add_argument("--key", default=os.environ.get("GCS_KEY"), help="service-account key file (default: $GCS_KEY)")
    args = p.parse_args(argv)
    files = collect(args.paths, args.root)
    print(f"{len(files)} files to upload, named relative to {args.root.resolve()}:\n{summary(files)}")
    if not args.upload:
        print("\nDRY RUN: nothing uploaded. Add --upload --dest BUCKET/PREFIX [--key KEY.json] to upload.")
        return 0
    if not args.dest:
        p.error("--upload needs --dest")
    base = rclone_base(args.key) if remote_prefix == ":gcs:" else ["rclone"]
    dest = f"{remote_prefix}{args.dest}"
    listing = subprocess.run([*base, "lsf", "-R", "--files-only", dest], capture_output=True, text=True)
    if listing.returncode != 0 and "directory not found" not in listing.stderr:
        sys.exit(f"cannot list {args.dest} (credentials? bucket name?):\n{listing.stderr[-1000:]}")
    if listing.returncode == 0 and listing.stdout.strip():
        sys.exit(f"{args.dest} already has files; uploads never overwrite. Choose another --dest prefix.")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "SHA256SUMS").write_text(sha256sums(files))
        (stage / "files.txt").write_text("".join(f.dest + "\n" for f in files))
        copy = [*base, "copy", "--files-from", str(stage / "files.txt"), "--transfers", "8", str(args.root.resolve()), dest]
        subprocess.run(copy, check=True)
        subprocess.run([*base, "copyto", str(stage / "SHA256SUMS"), f"{dest}/SHA256SUMS"], check=True)
        check = subprocess.run([*base, "check", "--one-way", "--files-from", str(stage / "files.txt"),
                                str(args.root.resolve()), dest], capture_output=True, text=True)
    n_remote = len(subprocess.run([*base, "lsf", "-R", "--files-only", dest], capture_output=True, text=True,
                                  check=True).stdout.split())
    ok = check.returncode == 0 and n_remote == len(files) + 1
    print(f"rclone check (MD5): {'equal' if check.returncode == 0 else 'DIFFERENCES'}; files at destination "
          f"{n_remote} (expected {len(files)} + SHA256SUMS)")
    if not ok:
        print(check.stderr[-2000:], file=sys.stderr)
        sys.exit("upload verification FAILED")
    print(f"uploaded and verified -> gs://{args.dest}/")
    return 0


if __name__ == "__main__":
    main()
