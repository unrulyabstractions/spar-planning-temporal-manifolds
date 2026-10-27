"""Shards of experiments/warm_cache.py cover every uncached prompt exactly once."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from warm_cache import shard_slice  # noqa: E402


@pytest.mark.parametrize("n_items, n_shards", [(0, 2), (1, 2), (7, 2), (500, 3), (5, 8)])
def test_shards_partition_the_items(n_items, n_shards):
    items = list(range(n_items))
    parts = [shard_slice(items, k, n_shards) for k in range(n_shards)]
    assert sum(parts, []) == items
    assert max(map(len, parts)) - min(map(len, parts)) <= 1
