import numpy as np
import polars as pl
import pytest

from eeg.ngs import reliability as R


def _games(n_players=400, n_games=12, sd_true=0.1, sd_game=0.1, group_shift=0.0, seed=0):
    """Log-normal channel values: a player effect plus game noise; optional big shift between two groups."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0, sd_true, n_players) + group_shift * (np.arange(n_players) % 2)
    rows = {
        "player_id": np.repeat(np.arange(n_players), n_games),
        "season": 2024,
        "game_id": np.tile(np.arange(n_games), n_players),
    }
    val = np.exp(np.repeat(true, n_games) + rng.normal(0, sd_game, n_players * n_games) + 2.0)
    games = pl.DataFrame(rows).with_columns(**{c: pl.Series(val) for c in ("s_peak", "burst_a")})
    groups = pl.DataFrame({"player_id": np.arange(n_players), "group": np.where(np.arange(n_players) % 2, "B", "A")})
    return games, groups


def test_spearman_brown():
    assert R.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert R.spearman_brown(0.5, 3) == pytest.approx(0.75)


def test_split_half_recovers_reliability_and_within_removes_group_shift():
    games, groups = _games()
    rel = R.split_half(games, groups, channels=("s_peak",))
    med = rel.filter(pl.col("stat") == "game_med").row(0, named=True)
    # mean of 12 games would give 0.01 / (0.01 + 0.01 / 12) = 0.92; the median is a little less efficient
    assert 0.8 < med["rel_pooled"] < 0.95 and med["rel_within"] == pytest.approx(med["rel_pooled"], abs=0.03)
    shifted = R.split_half(*_games(group_shift=0.5), channels=("s_peak",)).filter(pl.col("stat") == "game_med")
    assert shifted["rel_pooled"][0] > 0.97  # the group shift counts as signal when pooled
    assert shifted["rel_within"][0] == pytest.approx(med["rel_within"], abs=0.03)


def test_reliability_grows_with_games():
    games, groups = _games(n_games=16)
    curve = R.reliability_by_games(games, {"s_peak": "game_med"}, ks=(1, 2, 8), groups=groups)
    r = curve["reliability"].to_list()
    assert r[0] < r[1] < r[2] and r[0] == pytest.approx(0.5, abs=0.1)  # one game: 0.01 / (0.01 + 0.01)
    assert curve["n"].to_list() == [400, 400, 400]


def test_lab_rep_reliability():
    rng = np.random.default_rng(1)
    true = rng.normal(0, 0.1, 500)
    reps = pl.DataFrame(
        {
            "player_id": np.repeat(np.arange(500), 3),
            "rep_id": [f"{i}_{k}" for i in range(500) for k in "abc"],
            "lab_x": np.exp(np.repeat(true, 3) + rng.normal(0, 0.1, 1500)),
        }
    )
    out = R.lab_rep_reliability(reps, ["lab_x"]).row(0, named=True)
    assert out["n"] == 500 and out["rel_pooled"] == pytest.approx(0.5, abs=0.08)


def test_eta_reliability():
    rng = np.random.default_rng(2)
    f, l = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)  # independent, unit variance
    e = R.eta_reliability(0.8, 0.6, f, l)
    assert e["rel_eta"] == pytest.approx(1 - (0.2 + 0.4) / 2, abs=0.03)
    assert abs(e["r_field_lab"]) < 0.05 and e["sd_log_eta"] == pytest.approx(np.sqrt(2), abs=0.05)
