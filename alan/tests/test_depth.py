import pytest
from ptm.depth import parse_cells, parse_layers, resolve_layer

STANDARD = {"0.35L": 14, "0.45L": 18, "0.55L": 22, "0.65L": 26, "0.72L": 29, "0.825L": 33, "0.92L": 37}


def test_standard_fractions_reproduce_qwen3_14b_indices():
    got = {k: resolve_layer(k, 40) for k in STANDARD}
    print("\n40 layers:", got, "\n64 layers:", {k: resolve_layer(k, 64) for k in STANDARD})
    assert got == STANDARD


def test_absolute_and_mixed_specs():
    assert parse_cells("22:T3,0.92L:R0", 40) == [(22, "T3"), (37, "R0")]
    assert parse_layers("14,0.55L,37", 40) == [14, 22, 37]
    assert resolve_layer("1.0L", 64) == 64 and resolve_layer("0L", 64) == 0
    with pytest.raises(ValueError):
        resolve_layer("1.5L", 40)
