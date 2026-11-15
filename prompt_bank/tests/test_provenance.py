"""git_commit's "-dirty" flag covers exactly the code directories it is given (default: prompt_bank/)."""

import subprocess

from pbank.provenance import git_commit


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_dirty_flag_follows_the_given_directories(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "code.py").write_text("x = 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    a, b = tmp_path / "a", tmp_path / "b"
    clean = git_commit(a, b)
    (b / "code.py").write_text("x = 2\n")
    (a / "output.json").write_text("{}")            # untracked output: never makes the code dirty
    print(f"clean {clean}; b edited: git_commit(a) {git_commit(a)}, git_commit(a, b) {git_commit(a, b)}")
    assert not clean.endswith("-dirty") and len(clean) == 40
    assert git_commit(a) == clean                    # a's code is unchanged
    assert git_commit(a, b) == clean + "-dirty"      # b's is not
    assert git_commit(b) == clean + "-dirty"
