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


def test_without_git_the_stamp_names_the_commit_and_catches_edits(tmp_path):
    # The case of a rented box: the checkout is stamped, then copied without .git.
    import shutil
    from pbank.provenance import STAMP_NAME, write_stamp
    src = tmp_path / "src"
    for d in ("a", "b"):
        (src / d).mkdir(parents=True)
        (src / d / "code.py").write_text("x = 1\n")
        (src / d / ".gitignore").write_text("out/\n")           # tracked, but a copy without dotfiles drops it
    git(src, "init", "-q")
    git(src, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    git(src, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    commit = git_commit(src / "a")
    (src / "a" / "run_output.json").write_text("{}")           # untracked: not in the stamp
    write_stamp(src)
    box = tmp_path / "box"
    shutil.copytree(src, box, ignore=shutil.ignore_patterns(".*"))
    a, b = box / "a", box / "b"
    shipped = git_commit(a, b)
    (b / "code.py").write_text("x = 2\n")
    edited = (git_commit(a), git_commit(b), git_commit(a, b))
    (b / "code.py").unlink()
    (box / STAMP_NAME).rename(box / "elsewhere.json")
    print(f"commit {commit[:10]}; copy without .git: {shipped[:10]}; after editing b: a {edited[0][-10:]}, "
          f"b {edited[1][-10:]}, a+b {edited[2][-10:]}; without a stamp: {git_commit(a)}")
    assert shipped == commit and len(commit) == 40
    assert edited == (commit, commit + "-modified", commit + "-modified")
    assert git_commit(a) == "unknown"
    # A stamp made with uncommitted changes says so, and a stamp silent about a directory vouches for nothing.
    (src / "a" / "code.py").write_text("x = 3\n")
    write_stamp(src)
    box2 = tmp_path / "box2"
    shutil.copytree(src, box2, ignore=shutil.ignore_patterns(".git"))
    (box2 / "c").mkdir()
    assert git_commit(box2 / "a") == commit + "-dirty" and git_commit(box2 / "c") == "unknown"
