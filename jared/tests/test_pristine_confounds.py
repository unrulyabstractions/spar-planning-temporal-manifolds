import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from pristine_confounds import unit_names  # noqa: E402
from spar_horizon.horizons import UNIT_YEARS  # noqa: E402


def test_unit_names_invert_the_log_unit_size():
    units = ["seconds", "hours", "weeks", "months", "years"]
    assert list(unit_names(np.log10([UNIT_YEARS[u] for u in units]))) == units
