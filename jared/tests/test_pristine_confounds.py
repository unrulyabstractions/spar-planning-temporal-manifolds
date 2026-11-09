
import numpy as np

from pipeline.s1_data_vectors.analysis.pristine_confounds import unit_names
from spar_horizon.horizons import UNIT_YEARS


def test_unit_names_invert_the_log_unit_size():
    units = ["seconds", "hours", "weeks", "months", "years"]
    assert list(unit_names(np.log10([UNIT_YEARS[u] for u in units]))) == units
