"""The usable-core count under a container CPU quota (pbank/cpu.py)."""

import os

from pbank.cpu import cgroup_quota, usable_cores


def test_quota_is_read_from_either_cgroup_layout(tmp_path):
    v2, q, p, none = tmp_path / "cpu.max", tmp_path / "quota", tmp_path / "period", tmp_path / "absent"
    v2.write_text("1536000 100000\n")                    # as on the vast.ai A100 host of 2026-11-22
    assert cgroup_quota(v2, none, none) == 15.36
    v2.write_text("max 100000\n")
    assert cgroup_quota(v2, none, none) is None           # no limit
    q.write_text("400000\n"), p.write_text("100000\n")
    assert cgroup_quota(none, q, p) == 4.0                # cgroup v1
    q.write_text("-1\n")
    assert cgroup_quota(none, q, p) is None
    v2.write_text("garbage")
    assert cgroup_quota(v2, none, none) is None and cgroup_quota(none, none, none) is None


def test_usable_cores_is_the_smaller_of_affinity_and_quota():
    allowed = len(os.sched_getaffinity(0))
    print(f"this machine: {allowed} schedulable cores, quota {cgroup_quota()}, usable {usable_cores()}")
    assert usable_cores(quota=None) == allowed
    assert usable_cores(quota=15.36) == min(allowed, 15)  # rounded down
    assert usable_cores(quota=0.5) == 1                   # never 0
    assert usable_cores(quota=10_000) == allowed


def test_limit_torch_threads_only_lowers(monkeypatch):
    import pytest
    torch = pytest.importorskip("torch")
    from pbank import cpu
    start = torch.get_num_threads()
    try:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.setattr(cpu, "usable_cores", lambda: 2)
        assert cpu.limit_torch_threads() == (start, min(start, 2))
        monkeypatch.setattr(cpu, "usable_cores", lambda: 10_000)
        assert torch.get_num_threads() == cpu.limit_torch_threads()[1] == min(start, 2)      # never raised
        monkeypatch.setenv("OMP_NUM_THREADS", "7")
        monkeypatch.setattr(cpu, "usable_cores", lambda: 1)
        assert cpu.limit_torch_threads()[1] == min(start, 2)                                  # explicit setting wins
    finally:
        torch.set_num_threads(start)
