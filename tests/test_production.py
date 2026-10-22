import numpy as np
import polars as pl
import pytest

from eeg.outcomes import production as P


def _play(pid, season, game, play, **kw):
    base = {c: None for c in P.COLUMNS if c not in ("game_id", "play_id", "nfl_id")}
    base |= dict(player_id=pid, season=season, game_id=game, play_id=play, pass_result="C", blitzing=None)
    return base | kw


def _plays():
    rows = [
        # WR 1: 4 routes in 2 games, 1 target for 12 yards, separation on 3
        _play(1, 2024, 10, 1, target=True, rec_yards=12, separation_at_pass_forward=3.0),
        _play(1, 2024, 10, 2, target=False, separation_at_pass_forward=2.0),
        _play(1, 2024, 11, 1, target=False, separation_at_pass_forward=1.0),
        _play(1, 2024, 11, 2, target=False),
        # OL 2: 3 pass blocks, 1 pressure; 1 run play with nothing
        _play(2, 2024, 10, 1, pressure_allowed=True, peak_pressure_probability_allowed=0.6, sack_allowed=0.0),
        _play(2, 2024, 10, 2, pressure_allowed=False, peak_pressure_probability_allowed=0.2, sack_allowed=0.0),
        _play(2, 2024, 10, 3, pressure_allowed=False, peak_pressure_probability_allowed=0.1, sack_allowed=0.0),
        _play(2, 2024, 10, 4, pass_result=""),
        # DL 3: 2 rushes (1 pressure, half a sack), 1 run snap with a solo tackle; a WR-only column is ignored
        _play(3, 2024, 10, 1, blitzing=True, time_to_pressure=2.5, sack=0.5, player_get_off=0.9),
        _play(3, 2024, 10, 2, blitzing=True, sack=0.0, player_get_off=1.1),
        _play(3, 2024, 10, 3, pass_result="", tackle=1, assist=0, tackle_for_loss=True, target=True),
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def _groups():
    return pl.DataFrame({"player_id": [1, 2, 3], "group": ["WR", "OL", "DL"]})


def _season():
    s = P.season_values(P.game_sums(_plays(), _groups()))
    return {r["player_id"]: r for r in s.iter_rows(named=True)}


def test_receiver_measures_use_routes_and_measured_plays():
    wr = _season()[1]
    assert (wr["plays"], wr["games"], wr["plays_per_game"]) == (4, 2, 2.0)
    assert wr["target_rate"] == pytest.approx(0.25)
    assert wr["yprr"] == pytest.approx(3.0)
    assert wr["separation"] == pytest.approx(2.0) and wr["separation_n"] == 3
    assert wr["yac_oe"] is None and wr["yac_oe_n"] is None
    assert wr["tackle_rate"] is None


def test_line_measures_and_groups_that_do_not_apply_stay_null():
    s = _season()
    ol, dl = s[2], s[3]
    assert ol["pressure_allowed_rate"] == pytest.approx(1 / 3) and ol["pressure_allowed_rate_n"] == 3
    assert ol["peak_pressure_prob"] == pytest.approx(0.3)
    assert ol["pressure_rate"] is None
    assert dl["pressure_rate"] == pytest.approx(0.5) and dl["pressure_rate_n"] == 2
    assert dl["sack_rate"] == pytest.approx(0.25)
    assert dl["get_off"] == pytest.approx(1.0)
    assert dl["tackle_rate"] == pytest.approx(1 / 3) and dl["tfl_rate"] == pytest.approx(1 / 3)
    assert dl["target_rate"] is None


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")  # plays per game is constant here
def test_split_half_recovers_a_stable_rate():
    """Players with true target rates 0.05..0.35 over 16 games of 30 routes: halves agree closely."""
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(40):
        p = 0.05 + 0.3 * pid / 39
        for game in range(16):
            for play in range(30):
                rows.append((pid, 2024, game, play, bool(rng.random() < p)))
    plays = pl.DataFrame(rows, schema=["player_id", "season", "game_id", "play_id", "target"], orient="row")
    plays = plays.with_columns(*(pl.lit(None).alias(c) for c in P.COLUMNS if c not in plays.columns and c != "nfl_id"))
    groups = pl.DataFrame({"player_id": list(range(40)), "group": ["WR"] * 40})
    games = P.game_sums(plays, groups)
    qual = P.season_values(games)
    rel = P.split_half(games, qual).filter(pl.col("measure") == "target_rate").row(0, named=True)
    assert rel["n_half"] == 40 and rel["rel"] > 0.9


def test_retest_pairs_consecutive_seasons_only():
    q = pl.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3],
            "season": [2023, 2024, 2023, 2025, 2023, 2024],
            "group": ["WR"] * 6,
            "yprr": [1.0, 1.2, 2.0, 2.0, 3.0, 2.9],
        }
    )
    r = P.retest(q, [m for m in P.MEASURES if m.name == "yprr"]).filter(pl.col("group") == "WR").row(0, named=True)
    assert r["pairs"] == 2
