import pytest

from spar_horizon.config import Settings


def test_default_fractions_are_increasing_in_unit_interval():
    f = Settings().think_fractions
    assert f == (0.25, 0.5, 0.75, 1.0)


def test_fractions_are_coerced_to_float_tuple():
    assert Settings(think_fractions=[1]).think_fractions == (1.0,)


@pytest.mark.parametrize("bad", [(0.0, 1.0), (0.5, 0.5), (0.75, 0.25), (0.5, 1.5)])
def test_bad_fractions_raise(bad):
    with pytest.raises(ValueError):
        Settings(think_fractions=bad)
