import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from confound_robustness import none_only  # noqa: E402
from delay_margin import MIN_NONE  # noqa: E402


def test_none_only_keeps_none_rows_at_durations_with_enough_of_them():
    # duration a: MIN_NONE none rows and one near; duration b: one none row too few
    regime = np.array(["none"] * MIN_NONE + ["near"] + ["none"] * (MIN_NONE - 1))
    duration = np.array(["a"] * (MIN_NONE + 1) + ["b"] * (MIN_NONE - 1))
    X = np.arange(len(regime))[:, None].astype(float)
    data = {"regime": regime, "duration_id": duration, "y": np.zeros(len(regime))}
    X_kept, kept = none_only(X, data)
    assert list(X_kept[:, 0]) == list(range(MIN_NONE))
    assert set(kept["regime"]) == {"none"} and set(kept["duration_id"]) == {"a"}
