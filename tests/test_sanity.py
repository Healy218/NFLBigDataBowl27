import numpy as np
import polars as pl
import pytest
from test_tee import COD_REPS, FORTY_REPS, PROSPECTS

from eeg import config
from eeg.physics import sanity, tee

RAW = config.load_raw()


def _tee_frame(n=30, seed=0):
    """Synthetic tee table: n prospects in one group and year, player 0 has a huge E_vert."""
    rng = np.random.default_rng(seed)
    cols = {c: rng.normal(1000.0, 50.0, n) for c in tee.CHANNELS}
    cols["E_vert"][0] = 5000.0
    df = pl.DataFrame({"player_id": np.arange(n), "combine_year": 2024, "pos_group": "WR", "mass_kg": 90.0, **cols})
    df = df.with_columns(*[(pl.col(c) / pl.col("mass_kg")).alias(f"{c}_per_kg") for c in tee.CHANNELS])
    return df.with_columns(*[tee.cohort_z(c) for c in (*tee.CHANNELS, *tee.REL_CHANNELS)])


def test_outlier_flags_both_scopes():
    flags = sanity.outlier_flags(_tee_frame(), 4.0)
    assert set(flags["player_id"]) == {0}
    assert set(flags["channel"]) == {"E_vert", "E_vert_per_kg"}
    assert set(flags["scope"]) == {"combine_year", "pos_group"}
    assert sanity.outlier_flags(_tee_frame(), 6.0).height == 0  # max |z| with n = 30 is 29/√30 ≈ 5.3


def test_spearman_ignores_float_noise_ties():
    # equal values that differ only by float noise must not reorder under a uniform rescale
    a = pl.Series([1.0, 1.0 + 1e-14, 1.0 - 1e-14, 2.0, 3.0, None])
    assert sanity.spearman(a, a * 1.1547005383792515) == pytest.approx(1.0)
    assert sanity.spearman(a, -a) == pytest.approx(-1.0)


def test_sweep_cases():
    cases = sanity.sweep_cases(RAW)
    keys = [(k, b) for k, b, _ in cases]
    assert ("broad.theta_deg", "low") in keys and ("broad.theta_deg", "high") not in keys  # high == default
    assert keys[-2:] == [("all", "low"), ("all", "high")]
    n_params = len(list(config.sweep_params(RAW)))
    assert len(cases) == 2 * n_params - 1 + 2
    corner = config.values(cases[-2][2])
    assert all(corner[p.section][p.name] == p.low for p in config.sweep_params(RAW))


def test_scale_only_params_keep_ranks_exactly():
    sens = sanity.sensitivity(PROSPECTS, FORTY_REPS, COD_REPS, RAW)
    assert set(sens["column"]) == set(sanity.CHECKED)
    for key, cols in (("broad.theta_deg", ("E_broad", "E_broad_per_kg")), ("bench.rom_arm_fraction", ("W_bench",))):
        rows = sens.filter((pl.col("param") == key) & pl.col("column").is_in(cols))
        assert rows.height > 0 and (rows["rho_pooled"] == 1.0).all()
    # identical builds are perfectly stable
    base = tee.build_tee(PROSPECTS, FORTY_REPS, COD_REPS, config.values(RAW))
    st = sanity.rank_stability(base, base, cols=("E_vert", "TEE_z_abs"))
    assert (st["rho_pooled"] == 1.0).all()
