import numpy as np
import polars as pl
import pytest

from eeg.clustering import audit


def test_test_retest_recovers_reliability():
    rng = np.random.default_rng(0)
    n = 4000
    true = rng.normal(1.3, 0.1, n)
    # rep noise SD = true SD -> reliability r = 0.5
    reps = pl.DataFrame(
        {
            "player_id": np.repeat(np.arange(n), 2),
            "rep_id": [f"{i}_{k}" for i in range(n) for k in "ab"],
            "draft_year": np.repeat(rng.choice([2023, 2024], n), 2),
            "tau": np.repeat(true, 2) + rng.normal(0, 0.1, 2 * n),
            "v_max": np.repeat(true, 2),
        }
    )
    # a single-rep player is ignored
    reps = pl.concat(
        [reps, reps.head(1).with_columns(pl.lit(-1, pl.Int64).alias("player_id"), pl.lit("x").alias("rep_id"))]
    )
    tr = audit.test_retest(reps)
    tau_all = tr.filter((pl.col("column") == "tau") & (pl.col("cohort") == "all")).row(0, named=True)
    assert tau_all["n"] == n and tau_all["r"] == pytest.approx(0.5, abs=0.04)
    assert tr.filter(pl.col("column") == "v_max")["r"].min() == pytest.approx(1.0)
    assert set(tr["cohort"]) == {"all", "2023", "2024"}


def test_partial_corr_removes_shared_driver():
    rng = np.random.default_rng(1)
    z = rng.normal(size=5000)
    x, y = z + rng.normal(size=5000), z + rng.normal(size=5000)
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(0.5, abs=0.05)
    assert audit.partial_corr(x, y, z) == pytest.approx(0.0, abs=0.05)


def test_tau_validity_detects_acceleration_signal():
    rng = np.random.default_rng(2)
    n = 3000
    vm, tau = rng.normal(9.5, 0.6, n), rng.normal(1.3, 0.1, n)
    split = 3.0 - 0.1 * vm + 1.0 * tau + rng.normal(0, 0.02, n)
    ctx = pl.DataFrame({"player_id": np.arange(n), "combine_year": 2024, "v_max": vm, "tau": tau})
    tv = audit.tau_validity(ctx, pl.DataFrame({"player_id": np.arange(n), "ten_split_s": split}))
    row = tv.filter(pl.col("cohort") == "all").row(0, named=True)
    assert row["partial_r"] > 0.9 and row["r2_vmax_tau"] > row["r2_vmax"]


def test_vif_and_pca():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 5000))
    X = np.column_stack([a, b, a + 0.1 * rng.normal(size=5000)])
    v = audit.vif(X)
    assert v[1] == pytest.approx(1.0, abs=0.05) and v[0] > 50 and v[2] > 50
    ev, vec = audit.pca(X)
    assert ev.sum() == pytest.approx(3.0) and (np.diff(ev) <= 0).all()
    assert ev[0] == pytest.approx(2.0, abs=0.05)  # a and its copy load together
    assert (vec[np.abs(vec).argmax(axis=0), np.arange(3)] > 0).all()
