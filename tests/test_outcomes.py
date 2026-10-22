import polars as pl

from eeg.outcomes import table as T

LAST = 2025


def _players():
    """1: 2023 round 1, plays every season. 2: 2023 round 7, plays 2023 only (exit). 3: 2023 undrafted, plays
    2023 and 2025 (gap). 4: 2025 round 4, never plays."""
    return pl.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "draft_year": [2023, 2023, 2023, 2025],
            "draft_round": [1, 7, None, 4],
            "draft_pick_within_round": [5, 10, None, 3],
            "draft_overall_pick": [5, 230, None, 103],
        }
    )


def _pp():
    rows = []
    for pid, season, n in [(1, 2023, 4), (1, 2024, 3), (1, 2025, 2), (2, 2023, 2), (3, 2023, 1), (3, 2025, 1)]:
        for i in range(n):
            # two plays per game; prospect 1 is on offense on even plays
            rows.append((season * 100 + i // 2, i, pid, "AAA", "AAA" if i % 2 == 0 else "BBB", season))
    return pl.DataFrame(
        rows, schema=["game_id", "play_id", "player_id", "team_abbr", "possession_team", "season"], orient="row"
    ).with_columns((pl.col("team_abbr") == pl.col("possession_team")).alias("offense"))


def _successes():
    return pl.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "career_offensive_snaps": [9, 4, 2, 0],
            "career_defensive_snaps": [0, 0, 0, 0],
            "career_special_teams_snaps": [1, 1, 1, 0],
            "career_games_active": [3, 1, 2, 0],
            "career_games_started": [3, 0, 0, 0],
            "ap_all_pro_1st_team": [0, 0, 0, 0],
            "ap_all_pro_2nd_team": [1, 0, 0, 0],
            "pro_bowl_original_ballot": [1, 0, 0, 0],
        }
    )


def _seasons():
    return T.player_seasons(T.season_grid(_players(), LAST), T.season_playing_time(_pp()))


def test_season_grid_covers_every_league_season_with_zeros():
    s = _seasons()
    assert s.height == 3 + 3 + 3 + 1
    assert s.filter(pl.col("player_id") == 4)["year"].to_list() == [1]
    p2 = s.filter(pl.col("player_id") == 2).sort("season")
    assert p2["plays"].to_list() == [2, 0, 0]
    assert p2["active"].to_list() == [True, False, False]
    p1 = s.filter((pl.col("player_id") == 1) & (pl.col("season") == 2023)).row(0, named=True)
    assert (p1["plays"], p1["games"], p1["off_plays"], p1["def_plays"], p1["team"]) == (4, 2, 2, 2, "AAA")


def test_careers_exit_gap_draft_and_coverage():
    c = T.careers(_seasons(), _players(), _successes(), LAST).sort("player_id")
    assert c["exited"].to_list() == [False, True, False, False]
    assert c["gap_seasons"].to_list() == [0, 0, 1, 0]
    assert c["never_played"].to_list() == [False, False, False, True]
    assert c["undrafted"].to_list() == [False, False, True, False]
    assert c["round_band"].cast(pl.String).to_list() == ["1", "6-7", "UDFA", "4-5"]
    assert c["seasons_available"].to_list() == [3, 3, 3, 1]
    assert c["pp_coverage"].to_list()[:3] == [1.0, 0.5, 1.0]
    assert c["pp_coverage"][3] is None
    assert c["all_pro"].to_list() == [1, 0, 0, 0]
    assert c["scrim_per_season"][0] == 3.0


def test_by_band_counts_exits_only_where_possible():
    c = T.careers(_seasons(), _players(), _successes(), LAST)
    b = T.by_band(c, LAST).with_columns(pl.col("round_band").cast(pl.String))
    row = {r["round_band"]: r for r in b.iter_rows(named=True)}
    assert row["6-7"]["could_exit"] == 1 and row["6-7"]["exited"] == 1
    assert row["4-5"]["could_exit"] == 0
    assert row["UDFA"]["exited"] == 0
