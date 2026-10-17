import numpy as np
import polars as pl
import pytest

from eeg import config
from eeg.clustering import features as F

CFG = config.load()


def _tee(n=60, seed=1):
    """Synthetic tee table: 3 years, y = c_year · m^slope · noise, with a few gaps."""
    rng = np.random.default_rng(seed)
    year = np.repeat([2023, 2024, 2025], n // 3)
    m = rng.uniform(80, 150, n)
    noise = lambda: np.exp(rng.normal(0, 0.05, n))  # noqa: E731
    year_shift = np.select([year == 2024, year == 2025], [1.10, 0.95], 1.0)
    df = pl.DataFrame(
        {
            "player_id": np.arange(n),
            "combine_year": year,
            "pos_group": rng.choice(["WR", "TE", "DB", "OL", "DL"], n),
            "mass_kg": m,
            "E_vert": 20 * m**0.6 * noise() * year_shift,
            "E_broad": 10 * m**0.7 * noise(),
            "E_40": 900 * m**0.25 * noise(),
            "W_bench": 50 * m**1.2 * noise(),
            "W_cod_shuttle": 30 * m**0.65 * noise(),
            "W_cod_cone": 25 * m**0.75 * noise(),
        }
    )
    return df.with_columns(
        pl.when(pl.col("player_id") % 4 == 0).then(None).otherwise(pl.col("W_bench")).alias("W_bench"),
        pl.when(pl.col("player_id") == 7).then(None).otherwise(pl.col("E_broad")).alias("E_broad"),
    )


def _forty(tee, seed=2):
    rng = np.random.default_rng(seed)
    m = tee["mass_kg"].to_numpy()
    n = tee.height
    return pl.DataFrame(
        {
            "rep_id": [f"{i}a" for i in range(n)],
            "player_id": tee["player_id"],
            "v_max": 30 * m**-0.34 * np.exp(rng.normal(0, 0.03, n)),
            "tau": rng.normal(1.27, 0.1, n),
            "rmse": 0.2,
            "t40_model": 4.8,
        }
    ).filter(pl.col("player_id") != 11)  # no 40 trace


@pytest.fixture(scope="module")
def built():
    tee = _tee()
    return tee, F.build_features(tee, _forty(tee), CFG)


def test_mass_residual_recovers_slope_and_removes_year_level():
    tee = _tee()
    resid, fit = F.mass_residual(tee, "E_vert")
    assert fit["slope"] == pytest.approx(0.6, abs=0.05)
    assert fit["n"] == tee.height
    d = tee.with_columns(resid.alias("r"))
    # the 2024 / 2025 level shifts don't leak into the residual
    assert d.group_by("combine_year").agg(pl.col("r").mean())["r"].abs().max() < 1e-12
    assert abs(np.corrcoef(d["r"], np.log(d["mass_kg"]))[0, 1]) < 1e-9


def test_mass_residual_keeps_row_alignment_with_gaps():
    tee = _tee()
    resid, fit = F.mass_residual(tee, "W_bench")
    assert fit["n"] == tee["W_bench"].is_not_null().sum()
    assert (resid.is_null() == tee["W_bench"].is_null()).all()


def test_features_drop_incomplete_and_exclude_positions(built):
    tee, (features, context, fits) = built
    ids = set(features["player_id"])
    assert 7 not in ids and 11 not in ids  # missing broad / missing 40 trace
    assert len(ids) == tee.height - 2
    assert features.columns == ["player_id", "f_log_mass", *[f"f_{c}" for c in CFG["clustering"]["features"]]]
    assert "pos_group" in context.columns and context["player_id"].to_list() == features["player_id"].to_list()
    assert set(fits["column"]) == {*CFG["clustering"]["features"], *CFG["clustering"]["descriptors"]}


def test_features_are_cohort_z_scores_independent_of_mass(built):
    _, (features, context, _) = built
    df = features.join(context.select("player_id", "combine_year", "mass_kg"), on="player_id")
    for c in features.columns[1:]:
        g = df.group_by("combine_year").agg(pl.col(c).mean().alias("mu"), pl.col(c).std().alias("sd"))
        assert g["mu"].abs().max() < 1e-12 and (g["sd"] - 1).abs().max() < 1e-12
    for c in ("f_E_vert", "f_E_broad", "f_v_max"):
        assert abs(np.corrcoef(df[c], np.log(df["mass_kg"]))[0, 1]) < 0.05


def test_descriptors_null_where_channel_missing(built):
    _, (_, context, _) = built
    assert (context["d_W_bench"].is_null() == context["W_bench"].is_null()).all()
    assert context["d_W_bench"].is_null().any() and context["d_E_40"].is_not_null().all()
