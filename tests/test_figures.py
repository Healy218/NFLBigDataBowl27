import numpy as np
import polars as pl
import pytest

from eeg.viz.figures import _partial


def test_partial_residual_slope_equals_the_multiple_regression_coefficient():
    """Frisch–Waugh: the slope of the partial plot is the coefficient of x with the controls, within groups."""
    rng = np.random.default_rng(0)
    n = 300
    g = rng.integers(0, 3, n)
    c = rng.normal(size=n) + g
    x = 0.5 * c + rng.normal(size=n)
    y = 2.0 * x - 1.5 * c + 3.0 * g + rng.normal(0, 0.1, n)
    d = pl.DataFrame({"g": g, "x": x, "c": c, "y": y})
    xv, yv = _partial(d, "y", "x", ["c"], "g")
    assert np.polyfit(xv, yv, 1)[0] == pytest.approx(2.0, abs=0.02)
    assert abs(xv.mean()) < 1e-9 and abs(yv.mean()) < 1e-9
