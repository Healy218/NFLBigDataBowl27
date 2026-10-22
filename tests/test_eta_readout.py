import numpy as np
import polars as pl
import pytest

from eeg.ngs import eta_readout as R


def _seasons(n_players=3000, sd_true=0.07, sd_season=0.08, seed=0):
    """1–3 seasons per prospect: true value (group mean + spread) plus season noise."""
    rng = np.random.default_rng(seed)
    group = np.where(np.arange(n_players) % 2, "A", "B")
    true = np.where(group == "A", -0.3, -0.6) + rng.normal(0, sd_true, n_players)
    n = rng.integers(1, 4, n_players)
    pid = np.repeat(np.arange(n_players), n)
    x = true[pid] + rng.normal(0, sd_season, len(pid))
    return pl.DataFrame({"player_id": pid, "group": group[pid], "x": x}), true


def test_variance_components_recover_spreads():
    seasons, _ = _seasons()
    vc = R.variance_components(seasons, "x")
    assert np.sqrt(vc["sigma2"]) == pytest.approx(0.08, rel=0.05)
    assert np.sqrt(vc["tau2"]) == pytest.approx(0.07, rel=0.1)


def test_shrinkage_pulls_toward_group_mean_and_beats_raw():
    seasons, true = _seasons()
    vc = R.variance_components(seasons, "x")
    p = R.shrink(seasons, "x", vc).sort("player_id")
    one, three = p.filter(pl.col("n") == 1)["B"][0], p.filter(pl.col("n") == 3)["B"][0]
    assert 0 < one < three < 1
    t = true[p["player_id"].to_numpy()]
    mse = lambda c: np.mean((p[c].to_numpy() - t) ** 2)  # noqa: E731
    assert mse("shrunk") < mse("raw")
    covered = (p["lo90"].to_numpy() <= t) & (t <= p["hi90"].to_numpy())
    assert covered.mean() == pytest.approx(0.9, abs=0.03)
    assert p.filter(pl.col("group") == "A")["rank"].sort().to_list() == list(range(1, 1 + (p["group"] == "A").sum()))


def test_within_r_ignores_group_shift():
    rng = np.random.default_rng(1)
    n = 2000
    group = np.where(np.arange(n) % 2, "A", "B")
    x = rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n) + np.where(group == "A", 10.0, 0.0)
    x = x + np.where(group == "A", 10.0, 0.0)  # pooled r would be ~1
    r = R.within_r(pl.DataFrame({"group": group, "x": x, "y": y}), "x", "y")
    assert r["r"] == pytest.approx(0.5 / np.sqrt(1.25), abs=0.05) and r["lo95"] < r["r"] < r["hi95"]


def test_terciles_within_group():
    df = pl.DataFrame({"group": ["A"] * 6 + ["B"] * 3, "v": [1, 2, 3, 4, 5, 6, 100, 200, 300]})
    t = df.with_columns(R.terciles(df, "v").alias("t"))["t"].to_list()
    assert t == ["low", "low", "mid", "mid", "high", "high", "low", "mid", "high"]
