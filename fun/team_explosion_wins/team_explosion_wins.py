"""Does a team's explosion predict its season? And how many sticks of dynamite is that? (just for fun)

    uv run python fun/team_explosion_wins/team_explosion_wins.py

Team explosion per game, from the competition's game tracking (as in fun/week1_explosion), every regular-season
game of 2023–2025:
- `peak_kJ`: the sum over the team's starters (≥ half of their unit's snaps) of each one's top-speed kinetic energy
  in the game, ½ m v² (the week 1 / eight-week ranking measure);
- `all_play_kJ`: the sum over every play of every tracked player of the team of that play's peak ½ m v²: the
  game's whole explosion budget;
- `starters` and `kJ_per_starter`: how many young starters, and how explosive each is.

Results from the public nflverse schedule (data/external/nflverse/games.csv, matched on the competition game id;
public data is allowed under fun/ only): wins, ties, points for and against.

Does it predict? Per team-season (96), centred on the season mean (2023 tracks one draft class, 2025 three, so raw
totals aren't comparable across seasons): r of weeks 1–8 explosion with weeks 9–18 win share and point
differential per game (prediction: later games only), the same controlling for the weeks 1–8 record (does it add
to how the team was already playing?), and r over the full season (description). 95% Fisher-z intervals, df
adjusted for the season centring.

Big caveat: only the 2023–2025 draft classes in WR, TE, DB, OL, DL are tracked, so "team explosion" is the
explosion of each team's young starters, not its whole roster. Teams that start many rookies are often
rebuilding, which can push the relation negative.

Dynamite: one standard stick (about 8 x 1.25 in) releases roughly 2.1 MJ; published figures run 1–2.5 MJ with stick
size and nitroglycerin content. (Dynamite is nitroglycerin-based, not TNT; 1 g of TNT is 4,184 J.)

Writes fun/team_explosion_wins/report.md (team aggregates).
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from eeg.clustering.audit import _md
from eeg.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "fun/week1_explosion"))
from week1_explosion import INTERIM, PROCESSED, RAW, SEASONS, TEAM_ALIASES, starters  # noqa: E402

HERE = PROJECT_ROOT / "fun/team_explosion_wins"
SCHEDULE = PROJECT_ROOT / "data/external/nflverse/games.csv"
STICK_J = 2.1e6  # one stick of dynamite, J (published range ~1-2.5 MJ)
SPLIT_WEEK = 8


def team_games() -> pl.DataFrame:
    """Every regular-season team-game with the result, from the competition games and the nflverse schedule."""
    comp = pl.read_csv(RAW / "games.csv").filter(pl.col("season_type") == "REG")
    s = pl.read_csv(SCHEDULE, infer_schema_length=0).filter(pl.col("game_type") == "REG")
    s = s.select(
        pl.col("old_game_id").cast(pl.Int64).alias("game_id"),
        pl.col("home_score").cast(pl.Int64),
        pl.col("away_score").cast(pl.Int64),
    )
    g = comp.join(s, on="game_id", how="inner")
    assert g.height == comp.height, "every competition game should have a result"
    side = []
    for team, opp, pf, pa in [
        ("home_team_abbr", "visitor_team_abbr", "home_score", "away_score"),
        ("visitor_team_abbr", "home_team_abbr", "away_score", "home_score"),
    ]:
        side.append(
            g.select(
                "game_id", "season", "week",
                pl.col(team).replace(TEAM_ALIASES).alias("team_abbr"),
                pl.col(pf).alias("points_for"), pl.col(pa).alias("points_against"),
            )
        )  # fmt: skip
    return pl.concat(side).with_columns(
        pl.when(pl.col("points_for") > pl.col("points_against"))
        .then(1.0)
        .when(pl.col("points_for") == pl.col("points_against"))
        .then(0.5)
        .otherwise(0.0)
        .alias("win")
    )


def game_explosion(games: pl.DataFrame) -> pl.DataFrame:
    """Per team-game: starters, peak_kJ (starters' top-speed ½ m v²), all_play_kJ (every tracked play's peak)."""
    cs = pl.read_parquet(INTERIM / "combine_si.parquet")
    groups = cs.select("player_id", pl.col("pos_group").alias("group"))
    mass = cs.select("player_id", "mass_kg")
    roles = starters(games.select("game_id", "season").unique(), groups)
    fp = (
        pl.read_parquet(PROCESSED / "field_plays.parquet")
        .join(games.select("game_id").unique(), on="game_id")
        .join(mass, on="player_id")
        .with_columns((pl.col("ke_peak") * pl.col("mass_kg")).alias("J"))
    )
    team_of = roles.select("game_id", "player_id", "team_abbr")
    peak = (
        fp.group_by("game_id", "player_id")
        .agg(pl.col("J").max())
        .join(roles.filter("starter").select("game_id", "player_id", "team_abbr"), on=["game_id", "player_id"])
        .group_by("game_id", "team_abbr")
        .agg(pl.len().alias("starters"), (pl.col("J").sum() / 1000).alias("peak_kJ"))
    )
    every = (
        fp.join(team_of, on=["game_id", "player_id"])
        .group_by("game_id", "team_abbr")
        .agg((pl.col("J").sum() / 1000).alias("all_play_kJ"))
    )
    return (
        games.join(peak, on=["game_id", "team_abbr"], how="left")
        .join(every, on=["game_id", "team_abbr"], how="left")
        .with_columns(pl.col("starters", "peak_kJ", "all_play_kJ").fill_null(0))
    )


def season_table(tg: pl.DataFrame) -> pl.DataFrame:
    """Per team-season: explosion and results, weeks 1–SPLIT_WEEK and after, and the full season."""

    def part(df: pl.DataFrame, tag: str) -> pl.DataFrame:
        return df.group_by("season", "team_abbr").agg(
            pl.len().alias(f"games{tag}"),
            pl.col("peak_kJ").mean().alias(f"peak_kJ{tag}"),
            pl.col("all_play_kJ").mean().alias(f"all_play_kJ{tag}"),
            pl.col("starters").mean().alias(f"starters{tag}"),
            (pl.col("peak_kJ").sum() / pl.col("starters").sum()).fill_nan(None).alias(f"kJ_per_starter{tag}"),
            pl.col("win").mean().alias(f"win_share{tag}"),
            ((pl.col("points_for") - pl.col("points_against")).mean()).alias(f"point_diff{tag}"),
        )

    first = part(tg.filter(pl.col("week") <= SPLIT_WEEK), "_early")
    late = part(tg.filter(pl.col("week") > SPLIT_WEEK), "_late")
    full = part(tg, "")
    wins = tg.group_by("season", "team_abbr").agg(
        pl.col("win").sum().alias("wins"),
        pl.col("peak_kJ").sum().alias("season_peak_kJ"),
        pl.col("all_play_kJ").sum().alias("season_all_play_kJ"),
    )
    return (
        full.join(first, on=["season", "team_abbr"])
        .join(late, on=["season", "team_abbr"])
        .join(wins, on=["season", "team_abbr"])
    )


def _centred(d: pl.DataFrame, cols) -> pl.DataFrame:
    return d.drop_nulls(cols).with_columns((pl.col(c) - pl.col(c).mean().over("season")) for c in cols)


def _ci(r: float, n: int, k: int) -> tuple[float, float]:
    half = 1.96 / np.sqrt(n - 3 - k)
    return float(np.tanh(np.arctanh(r) - half)), float(np.tanh(np.arctanh(r) + half))


def assoc(d: pl.DataFrame, x: str, y: str, control: str | None = None) -> dict:
    """r of x with y within season (optionally partial on control), with a 95% interval."""
    cols = [x, y] + ([control] if control else [])
    c = _centred(d, cols)
    xv, yv = c[x].to_numpy(), c[y].to_numpy()
    if control:
        z = c[control].to_numpy()
        xv = xv - np.polyval(np.polyfit(z, xv, 1), z)
        yv = yv - np.polyval(np.polyfit(z, yv, 1), z)
    r = float(np.corrcoef(xv, yv)[0, 1])
    lo, hi = _ci(r, c.height, (c["season"].n_unique() - 1) + (1 if control else 0))
    return dict(explosion=x, outcome=y, control=control or "—", n=c.height, r=r, lo95=lo, hi95=hi)


def main() -> None:
    tg = game_explosion(team_games())
    st = season_table(tg).with_columns(
        (pl.col("season_all_play_kJ") * 1000 / STICK_J).alias("season_sticks"),
        (pl.col("peak_kJ") * 1000 / STICK_J * 100).alias("pct_stick_per_game"),
    )
    tests = []
    for x in ("peak_kJ_early", "kJ_per_starter_early", "starters_early", "all_play_kJ_early"):
        tests.append(assoc(st, x, "win_share_late"))
        tests.append(assoc(st, x, "win_share_late", control="win_share_early"))
        tests.append(assoc(st, x, "point_diff_late"))
    for x in ("peak_kJ", "kJ_per_starter", "starters"):
        tests.append(assoc(st, x, "win_share"))
    tests = pl.DataFrame(tests)

    show = [
        "team_abbr", "wins", pl.col("point_diff").round(1),
        pl.col("peak_kJ").round(1).alias("peak_kJ_per_game"), pl.col("pct_stick_per_game").round(2),
        pl.col("starters").round(1), pl.col("season_sticks").round(1),
    ]  # fmt: skip
    sections = []
    for season in SEASONS:
        s = st.filter(pl.col("season") == season).sort("peak_kJ", descending=True)
        sections.append(f"### {season}\n\n{_md(s.select(show), 1)}\n")
    boom = st.sort("season_sticks", descending=True).head(5).select("season", *show)

    report = f"""# Does team explosion predict the season? (in sticks of dynamite, just for fun)

Generated by `uv run python fun/team_explosion_wins/team_explosion_wins.py`. Team explosion from the competition's
game tracking (young starters of the 2023–2025 classes only); results from the public nflverse schedule.
One stick of dynamite ≈ 2.1 MJ (published figures 1–2.5 MJ; dynamite is
nitroglycerin-based, not TNT).

- `peak_kJ_per_game`: the team's starters' top-speed kinetic energy, summed per game, averaged over the season
  (`pct_stick_per_game`: the same as a percentage of one stick).
- `season_sticks`: every tracked play's peak ½ m v² for the team's tracked players, summed over the season, in
  sticks of dynamite: the season's whole explosion budget.

## Does it predict?

Per team-season, centred on the season mean. `_early` = weeks 1–{SPLIT_WEEK}, `_late` = weeks {SPLIT_WEEK + 1}–18 (the
prediction uses only later games). `control` = partial r holding the weeks 1–{SPLIT_WEEK} win share fixed. Rows with
`win_share` and no suffix are the full season (description, not prediction).

{_md(tests, 2)}

## The biggest explosion budgets

{_md(boom, 1)}

## Every team-season, ranked by peak kJ per game

{chr(10).join(sections)}"""
    (HERE / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
