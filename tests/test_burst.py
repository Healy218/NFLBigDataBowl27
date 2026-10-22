import numpy as np
import polars as pl
import pytest

from eeg.physics import burst as B


def test_official_start_is_constant_acceleration_from_rest():
    p = pl.DataFrame({"player_id": [1, 2], "ten_split_s": [1.6, None], "forty_s": [4.6, 4.9], "mass_kg": [100.0, 90.0]})
    out = B.official_start(p)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["a10"] == pytest.approx(2 * B.D10_M / 1.6**2)
    assert row["fly30_off"] == pytest.approx(3.0)
    # work m a d over the split time
    assert row["p10_w"] == pytest.approx(100.0 * row["a10"] * B.D10_M / 1.6)


def test_per_prospect_is_geometric_mean_over_reps():
    reps = pl.DataFrame({"player_id": [1, 1, 2], "tau": [1.0, 4.0, 1.5]})
    out = B.per_prospect(reps, ["tau"]).sort("player_id")
    assert out["n_reps"].to_list() == [2, 1]
    assert out["tau_mean"].to_list() == pytest.approx([2.0, 1.5])


def test_rep_reliability_recovers_simulated_noise():
    """True log value sd 0.1, rep noise sd 0.1: one rep 0.5, mean of two 0.67; a group shift is ignored."""
    rng = np.random.default_rng(0)
    n = 4000
    true = rng.normal(0, 0.1, n)
    shift = np.where(np.arange(n) % 2 == 0, 0.0, 1.0)
    rows = []
    for i in range(n):
        for k in range(2):
            rows.append((i, f"{i}_{k}", float(np.exp(1 + shift[i] + true[i] + rng.normal(0, 0.1)))))
    reps = pl.DataFrame(rows, schema=["player_id", "rep_id", "tau"], orient="row")
    groups = pl.DataFrame({"player_id": list(range(n)), "group": ["A" if s == 0 else "B" for s in shift]})
    r = B.rep_reliability(reps, groups, ["tau"]).row(0, named=True)
    assert r["rel_one_rep"] == pytest.approx(0.5, abs=0.04)
    assert r["rel_mean_of_2"] == pytest.approx(2 / 3, abs=0.04)


def test_betas_recover_standardized_effects_within_group():
    rng = np.random.default_rng(1)
    n = 3000
    x1, x2 = rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)
    grp = np.where(np.arange(n) % 2 == 0, "A", "B")
    y = 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.05, n) + np.where(grp == "A", 0.0, 2.0)
    df = pl.DataFrame({"group": grp, "y": np.exp(y), "a": np.exp(x1), "b": np.exp(x2 + (grp == "B") * 3.0)})
    out = B.betas(df, "y", ["a", "b"])
    sd_y = np.sqrt(0.25 * 0.01 + 0.09 * 0.01 + 0.0025)
    assert out["beta_a"] == pytest.approx(0.5 * 0.1 / sd_y, abs=0.03)
    assert out["beta_b"] == pytest.approx(-0.3 * 0.1 / sd_y, abs=0.03)
    assert out["n"] == n
