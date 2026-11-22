"""How many CPU cores this process may really use, and keeping torch's thread count within that.

On a rented container the machine can have far more cores than the container is allowed: on a vast.ai A100 host,
`os.cpu_count()` was 128 while the container's CPU quota was 15.36 cores (cgroup `cpu.max` = "1536000 100000"). torch
sizes its thread pool from the core count, so it started 128 threads on 15 cores' worth of time. Small CPU tensor
work then crawled: the steering test suite, which runs a tiny model on the CPU, passed in 106 s with 15 threads and
had finished 43 of 99 tests after 600 s with 128 (2026-11-22). Nothing was wrong with the results, only the speed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def cgroup_quota(v2: Path = CGROUP_V2, v1_quota: Path = CGROUP_V1_QUOTA, v1_period: Path = CGROUP_V1_PERIOD):
    """The container's CPU quota in cores (e.g. 15.36), or None if there is none or it cannot be read."""
    try:
        if v2.is_file():
            quota, period = v2.read_text().split()[:2]
            return None if quota == "max" else int(quota) / int(period)
        if v1_quota.is_file() and v1_period.is_file():
            quota, period = int(v1_quota.read_text()), int(v1_period.read_text())
            return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        pass
    return None


def usable_cores(quota=...) -> int:
    """Cores this process can use: those it may be scheduled on, capped by the container's quota (rounded down, at
    least 1). `quota` is for tests; by default it is read from the cgroup."""
    allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = cgroup_quota() if quota is ... else quota
    return max(1, min(allowed, math.floor(quota))) if quota else allowed


def limit_torch_threads() -> tuple[int, int]:
    """Lower torch's intra-op thread count to `usable_cores()` if it is above it. Returns (before, after). An
    explicit OMP_NUM_THREADS is respected, since torch has then already been told what to use."""
    import torch

    before = torch.get_num_threads()
    if "OMP_NUM_THREADS" not in os.environ and before > usable_cores():
        torch.set_num_threads(usable_cores())
    return before, torch.get_num_threads()
