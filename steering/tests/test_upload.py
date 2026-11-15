"""The upload scripts: what they collect and refuse, GCS through real rclone into a local directory, and Hugging Face
through a fake API that records commits (no network in either)."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import upload_gcs  # noqa: E402
import upload_hf  # noqa: E402
from upload_common import collect, sha256_of  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "repo"
    for rel, text in {"steering/methods/m/caa/method.json": "{}", "steering/methods/m/caa/arrays.npz": "x" * 5000,
                      "steering/fits/r/search.jsonl": "{}\n", "prompt_bank/runs/r/responses.jsonl": "{}\n"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_collect_names_files_by_repo_path(tree):
    files = collect([tree / "steering/methods/m", tree / "prompt_bank/runs/r"], tree)
    print([f.dest for f in files])
    assert [f.dest for f in files] == ["steering/methods/m/caa/arrays.npz", "steering/methods/m/caa/method.json",
                                       "prompt_bank/runs/r/responses.jsonl"]
    assert files[0].sha256 == sha256_of(tree / "steering/methods/m/caa/arrays.npz")


@pytest.mark.parametrize("bad", [".env", ".env.local", "gcs-key.pem", "service_credentials.json", "hf_token.txt",
                                 "client_secret.json", ".git/config"])
def test_collect_refuses_credentials_and_git(tree, bad):
    (tree / "steering/fits/r" / bad).parent.mkdir(parents=True, exist_ok=True)
    (tree / "steering/fits/r" / bad).write_text("not for upload")
    with pytest.raises(ValueError, match="credential or repository internals"):
        collect([tree / "steering/fits/r"], tree)


def test_collect_refuses_outside_root_and_symlinks(tree, tmp_path):
    (tmp_path / "elsewhere.txt").write_text("x")
    with pytest.raises(ValueError, match="outside --root"):
        collect([tmp_path / "elsewhere.txt"], tree)
    (tree / "steering/fits/r/link").symlink_to(tmp_path / "elsewhere.txt")
    with pytest.raises(ValueError, match="symbolic link"):
        collect([tree / "steering/fits/r"], tree)


def test_gcs_upload_with_rclone_into_a_local_remote(tree, tmp_path, capsys):
    if subprocess.run(["which", "rclone"], capture_output=True).returncode != 0:
        pytest.skip("rclone not installed")
    paths = [str(tree / "steering/methods/m"), str(tree / "steering/fits/r"), "--root", str(tree)]
    assert upload_gcs.main(paths) == 0                                         # dry run
    assert not (tmp_path / "remote").exists()
    remote = tmp_path / "remote" / "bucket" / "prefix"
    assert upload_gcs.main(paths + ["--upload", "--dest", str(remote)], remote_prefix="") == 0
    out = capsys.readouterr().out
    print(out.strip().splitlines()[-2:])
    got = sorted(str(p.relative_to(remote)) for p in remote.rglob("*") if p.is_file())
    assert got == ["SHA256SUMS", "steering/fits/r/search.jsonl", "steering/methods/m/caa/arrays.npz",
                   "steering/methods/m/caa/method.json"]
    check = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=remote, capture_output=True, text=True)
    print(check.stdout.strip())
    assert check.returncode == 0 and "MD5): equal" in out
    with pytest.raises(SystemExit, match="already has files"):
        upload_gcs.main(paths + ["--upload", "--dest", str(remote)], remote_prefix="")


class FakeHub:
    """Records what upload_hf.py does; serves path infos with the LFS SHA-256 of what was committed."""

    def __init__(self, existing=()):
        self.files = {p: None for p in existing}
        self.created, self.commits = None, []

    def whoami(self):
        return {"name": "tester"}

    def repo_exists(self, repo_id, repo_type):
        return bool(self.files)

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.created = {"private": private, "existed": bool(self.files)}

    def get_paths_info(self, repo_id, paths, repo_type):
        return [SimpleNamespace(path=p, lfs=SimpleNamespace(sha256=self.files[p]) if self.files[p] else None,
                                blob_id=None) for p in paths if p in self.files]

    def create_commit(self, repo_id, ops, repo_type, commit_message):
        self.commits.append([op.path_in_repo for op in ops])
        for op in ops:
            self.files[op.path_in_repo] = sha256_of(Path(op.path_or_fileobj))
        return SimpleNamespace(commit_url=f"commit {len(self.commits)}")


def test_hf_upload_through_a_fake_hub(tree):
    paths = [str(tree / "steering/methods/m"), "--root", str(tree), "--repo-id", "org/name", "--prefix", "v1"]
    hub = FakeHub()
    assert upload_hf.main(paths, api=hub) == 0 and hub.commits == []         # dry run: no calls that change anything
    assert upload_hf.main(paths + ["--upload"], api=hub) == 0
    print(f"created {hub.created}; commits {hub.commits}")
    assert hub.created == {"private": True, "existed": False}
    assert sorted(hub.commits[0]) == ["v1/SHA256SUMS", "v1/steering/methods/m/caa/arrays.npz",
                                      "v1/steering/methods/m/caa/method.json"]
    with pytest.raises(SystemExit, match="never overwrite"):
        upload_hf.main(paths + ["--upload"], api=hub)


def test_hf_verification_catches_a_corrupted_remote(tree):
    class Corrupting(FakeHub):
        def create_commit(self, *a, **k):
            info = super().create_commit(*a, **k)
            self.files["v1/steering/methods/m/caa/arrays.npz"] = "0" * 64
            return info
    with pytest.raises(SystemExit, match="verification FAILED"):
        upload_hf.main([str(tree / "steering/methods/m"), "--root", str(tree), "--repo-id", "org/name",
                        "--prefix", "v1", "--upload"], api=Corrupting())
