import numpy as np
import polars as pl
import pytest

from eeg.ngs import transfer as T


def _data(n=600, slope=0.5, seed=0):
    """Two groups with very different means; y depends on a, not on b."""
    rng = np.random.default_rng(seed)
    g = np.where(np.arange(n) % 2 == 0, "A", "B")
    shift = (g == "B") * 5.0
    a, b = rng.normal(0, 1, n) + shift, rng.normal(0, 1, n) - shift
    y = slope * (a - shift) + rng.normal(0, 1, n) + 3 * shift
    return pl.DataFrame({"group": g, "a": a, "b": b, "y": y})


def test_fit_within_recovers_slope_and_ignores_group_shift():
    out = T.fit_within(_data(), "y", ["a", "b"])
    sd_y = np.sqrt(0.25 + 1)
    assert out["b_a"] == pytest.approx(0.5 / sd_y, abs=0.06)
    assert out["lo_b"] < 0 < out["hi_b"]
    assert out["n"] == 600


def test_cv_r2_is_near_zero_for_noise_and_positive_for_signal():
    assert T.cv_r2(_data(slope=0.0), "y", ["a", "b"]) < 0.01
    assert T.cv_r2(_data(slope=1.0), "y", ["a", "b"]) == pytest.approx(0.5, abs=0.07)


def test_univariate_is_within_group():
    u = T.univariate(_data(slope=0.0), targets=["y"], predictors=["a", "b"])
    # pooled, b and y would be strongly negatively related through the group shift
    assert u.filter(pl.col("variable") == "b")["r"][0] == pytest.approx(0.0, abs=0.1)
