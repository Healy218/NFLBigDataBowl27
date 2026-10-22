"""Week 1 explosion rankings, just for fun: which team put the most explosion on the field in week 1?

    uv run python fun/week1_explosion/week1_explosion.py

For each regular-season week 1 (2023, 2024, 2025), from the competition's game tracking and player_play:
- **Starters:** players with at least half of their unit's snaps in the game: offense or defense plays of their
  team; WR and TE against the team's pass plays (player_play holds only their route plays).
- **Game joules:** each starter's top-speed kinetic energy in the game, ½ m v² (J), from the highest sustained
  peak speed over their plays (field_plays `ke_peak`, J/kg, times Combine mass).
- **Explosive axis:** the starter's Combine explosive profile axis (size-adjusted explosion, SD); profiled
  prospects only.
- Team totals: the sum over starters of each, ranked by game joules.

Big caveat: the data tracks only the 2023–2025 draft classes (WR, TE, DB, OL, DL), so a team's total is the
explosion of its young starters from those classes, not of its whole lineup. More such starters, bigger sum.

Writes fun/week1_explosion/ranking.md (team totals) and data/processed/fun_week1_players.parquet (per player,
named; not committed).
"""

from __future__ import annotations

import polars as pl

from eeg.clustering.audit import _md
from eeg.config import PROJECT_ROOT

RAW = PROJECT_ROOT / "data/raw/kaggle/nfl-big-data-bowl-2027"
PROCESSED, INTERIM = PROJECT_ROOT / "data/processed", PROJECT_ROOT / "data/interim"
HERE = PROJECT_ROOT / "fun/week1_explosion"
STARTER_SHARE = 0.5
SEASONS = (2023, 2024, 2025)


def week1_games() -> pl.DataFrame:
    g = pl.read_csv(RAW / "games.csv")
    return g.filter((pl.col("week") == 1) & (pl.col("season_type") == "REG")).select("game_id", "season")


def starters(games: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """Per player-game in week 1: team, snaps, the unit's plays, share and a starter flag."""
    pp = pl.read_csv(
        RAW / "player_play.csv",
        columns=["game_id", "play_id", "team_abbr", "nfl_id", "possession_team", "pass_result"],
        infer_schema_length=100_000,
    ).join(games, on="game_id")
    plays = pp.select("game_id", "play_id", "possession_team", "pass_result").unique(["game_id", "play_id"])
    teams = pp.select("game_id", "team_abbr").unique()
    unit = (
        teams.join(plays, on="game_id")
        .group_by("game_id", "team_abbr")
        .agg(
            (pl.col("possession_team") == pl.col("team_abbr")).sum().alias("off_plays"),
            (pl.col("possession_team") != pl.col("team_abbr")).sum().alias("def_plays"),
            ((pl.col("possession_team") == pl.col("team_abbr")) & (pl.col("pass_result").fill_null("") != ""))
            .sum()
            .alias("pass_plays"),
        )
    )
    snaps = (
        pp.rename({"nfl_id": "player_id"})
        .group_by("season", "game_id", "team_abbr", "player_id")
        .agg(pl.len().alias("snaps"))
        .join(groups, on="player_id")
        .join(unit, on=["game_id", "team_abbr"])
    )
    denom = (
        pl.when(pl.col("group").is_in(["WR", "TE"]))
        .then(pl.col("pass_plays"))
        .when(pl.col("group").is_in(["OL"]))
        .then(pl.col("off_plays"))
        .otherwise(pl.col("def_plays"))
    )
    return snaps.with_columns((pl.col("snaps") / denom).alias("share")).with_columns(
        (pl.col("share") >= STARTER_SHARE).alias("starter")
    )


def game_joules(games: pl.DataFrame, mass: pl.DataFrame) -> pl.DataFrame:
    """Per player-game: top sustained speed and its kinetic energy in joules (½ m v²)."""
    fp = pl.read_parquet(PROCESSED / "field_plays.parquet").join(games, on=["game_id", "season"])
    return (
        fp.group_by("game_id", "player_id")
        .agg(pl.col("s_peak").max().alias("top_speed_mps"), pl.col("ke_peak").max().alias("ke_per_kg"))
        .join(mass, on="player_id")
        .with_columns((pl.col("ke_per_kg") * pl.col("mass_kg")).alias("game_joules"))
    )


def team_table(players: pl.DataFrame) -> pl.DataFrame:
    """Per team: starters, game joules sum (kJ), explosive axis sum, ranks."""
    s = players.filter("starter")
    t = (
        s.group_by("team_abbr")
        .agg(
            pl.len().alias("starters"),
            (pl.col("game_joules").sum() / 1000).alias("game_kJ"),
            pl.col("ax_explosive").is_not_null().sum().alias("profiled"),
            pl.col("ax_explosive").sum().alias("axis_sum"),
            (pl.col("game_joules").mean() / 1000).alias("kJ_per_starter"),
        )
        .sort("game_kJ", descending=True)
        .with_row_index("rank", offset=1)
        .with_columns(pl.col("axis_sum").rank("ordinal", descending=True).cast(pl.Int64).alias("axis_rank"))
    )
    return t.select("rank", "team_abbr", "starters", "game_kJ", "kJ_per_starter", "axis_sum", "axis_rank", "profiled")


def main() -> None:
    cs = pl.read_parquet(INTERIM / "combine_si.parquet")
    groups = cs.select("player_id", pl.col("pos_group").alias("group"))
    mass = cs.select("player_id", "mass_kg")
    names = cs.select("player_id", "name", "pos")
    axis = pl.read_parquet(PROCESSED / "profiles.parquet").select("player_id", "ax_explosive")
    games = week1_games()

    players = (
        starters(games, groups)
        .join(game_joules(games, mass), on=["game_id", "player_id"], how="left")
        .join(axis, on="player_id", how="left")
        .join(names, on="player_id")
    )
    players.write_parquet(PROCESSED / "fun_week1_players.parquet")

    sections = []
    for season in SEASONS:
        p = players.filter(pl.col("season") == season)
        t = team_table(p)
        s = p.filter("starter")
        sections.append(
            f"""## Week 1, {season}

{p["team_abbr"].n_unique()} teams with a tracked prospect; {s.height} starters (≥ {STARTER_SHARE:.0%} of their unit's snaps),
{int(s["game_joules"].is_not_null().sum())} with game joules, {int(s["ax_explosive"].is_not_null().sum())} with a Combine profile.
`game_kJ`: sum of starters' top-speed kinetic energy in the game (kJ). `axis_sum`: sum of their Combine explosive
axis (SD; profiled starters only). Ranked by `game_kJ`.

{_md(t, 1)}
"""
        )
    report = f"""# Week 1 explosion rankings (just for fun)

Generated by `uv run python fun/week1_explosion/week1_explosion.py`. Which team put the most explosion on the field
in regular-season week 1?

**Read with care:** the competition data tracks only the 2023–2025 draft classes in five groups (WR, TE, DB, OL,
DL). A team's total is the explosion of its *young starters from those classes*, not its whole lineup, so teams
that start more of them score higher. `kJ_per_starter` is the fairer per-player view.

{chr(10).join(sections)}"""
    (HERE / "ranking.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
