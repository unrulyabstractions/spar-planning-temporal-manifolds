"""scripts/vast_status_remote.sh (the instance-side half of the 10-minute Vast budget check) must reach every verdict.

Runs the script locally against fake run folders. "Our work" processes are simulated with dummy `sleep` processes whose
argv0 carries a marker (PROC_RE override), and the GPU reading is injected (GPU_UTIL), so the verdicts never depend on what
else runs on the machine.
"""
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vast_status_remote.sh"


def verdict(repo: Path, proc_re: str, disk_min_pct: str = "0", gpu_util: str = "0") -> str:
    # disk threshold and GPU utilisation explicit: the local machine's real disk fill and GPU load must not decide
    env = dict(os.environ, REMOTE=str(repo), PROC_RE=proc_re, STALL_MIN="15", DISK_MIN_PCT=disk_min_pct, GPU_UTIL=gpu_util)
    out = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env).stdout
    lines = [l for l in out.splitlines() if l.startswith("VERDICT")]
    assert len(lines) == 1, out
    print(f"\n  {lines[0]}")
    return lines[0]


def fake_repo(tmp_path: Path, chain: str, sequence: str = "", capture_age_min: float = 0) -> Path:
    runs = tmp_path / "runs"; runs.mkdir(parents=True, exist_ok=True)
    (runs / "chain_m.log").write_text(chain)
    if sequence:
        (runs / "sequence.log").write_text(sequence)
    cap = runs / "m_investment_n2000_s0.log"; cap.write_text("[64/2000] batch 7.6s total 30s format_ok 16/16\n")
    t = time.time() - capture_age_min * 60; os.utime(cap, (t, t))
    return tmp_path


def dummy(marker: str):
    return subprocess.Popen(["bash", "-c", f"exec -a {marker} sleep 60"])


def test_failure_since_last_start_warns(tmp_path):
    repo = fake_repo(tmp_path, "2026-10-03 10:00:00 CHAIN START x\n2026-10-03 10:05:00 FAILED: capture\n")
    assert verdict(repo, "nomatch_" + uuid.uuid4().hex).startswith("VERDICT WARN failure")


def test_failure_before_a_restart_is_ignored(tmp_path):
    marker = "vs_work_" + uuid.uuid4().hex
    p = dummy(marker)
    try:
        repo = fake_repo(tmp_path, "", "2026-10-03 10:00:00 SEQUENCE START a\n2026-10-03 10:10:00 FAILED: count\n"
                                       "2026-10-03 10:20:00 SEQUENCE START b\n2026-10-03 10:21:00 chain start\n")
        assert verdict(repo, marker) == "VERDICT OK"
    finally:
        p.kill()


def test_sequence_done(tmp_path):
    repo = fake_repo(tmp_path, "2026-10-03 10:00:00 CHAIN DONE\n", "2026-10-03 10:00:00 SEQUENCE START\n2026-10-03 11:00:00 SEQUENCE DONE\n")
    assert verdict(repo, "nomatch_" + uuid.uuid4().hex) == "VERDICT DONE sequence finished"


def test_idle_gpu_and_no_work_warns(tmp_path):
    repo = fake_repo(tmp_path, "2026-10-03 10:00:00 CHAIN START x\n", "2026-10-03 10:00:00 SEQUENCE START\n")
    assert verdict(repo, "nomatch_" + uuid.uuid4().hex).startswith("VERDICT WARN idle")


def test_stalled_capture_warns(tmp_path):
    marker = "vs_capture.py_" + uuid.uuid4().hex       # argv0 contains capture.py, like a real capture process
    p = dummy(marker)
    try:
        repo = fake_repo(tmp_path, "2026-10-03 10:00:00 CHAIN START x\n", capture_age_min=30)
        assert verdict(repo, marker).startswith("VERDICT WARN stalled")
    finally:
        p.kill()


def test_running_capture_ok(tmp_path):
    marker = "vs_capture.py_" + uuid.uuid4().hex
    p = dummy(marker)
    try:
        repo = fake_repo(tmp_path, "2026-10-03 10:00:00 CHAIN START x\n", capture_age_min=1)
        assert verdict(repo, marker) == "VERDICT OK"
    finally:
        p.kill()


def test_low_disk_warns(tmp_path):
    marker = "vs_work_" + uuid.uuid4().hex
    p = dummy(marker)
    try:
        repo = fake_repo(tmp_path, "2026-10-03 10:00:00 CHAIN START x\n")
        assert verdict(repo, marker, disk_min_pct="101").startswith("VERDICT WARN disk")
    finally:
        p.kill()


def test_busy_gpu_without_listed_process_is_ok(tmp_path):
    repo = fake_repo(tmp_path, "2026-10-03 10:00:00 CHAIN START x\n", "2026-10-03 10:00:00 SEQUENCE START\n")
    assert verdict(repo, "nomatch_" + uuid.uuid4().hex, gpu_util="90") == "VERDICT OK"
