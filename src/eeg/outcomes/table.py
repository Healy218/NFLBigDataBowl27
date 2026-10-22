"""Regular-season outcome table: playing time per player-season, draft slot and career totals per prospect.

    uv run python -m eeg.outcomes.table

Sources (competition data only): player_play.csv (one row per prospect per scrimmage play, joined to games.csv
for the season type), players.csv (draft) and player_career_successes.csv (career totals, regular season,
right-censored at `outcomes.last_season`).

What a "play" is depends on the group (checked against career snaps, `reports/outcomes.md`):
- OL, DL, DB: every regular-season scrimmage snap (player_play matches career snaps exactly).
- WR, TE: only plays on which they ran a route (58% and 40% of their snaps); run-blocking snaps are absent from
  player_play and game tracking alike. Playing time is compared within group only.

Per player-season, for every season a prospect was in the league (draft year to `outcomes.last_season`, active
or not): plays, offensive and defensive plays, games with a play, team (most plays), year in the league.

Per prospect: draft round, pick and overall pick (null = undrafted) with a round band; career scrimmage and
special-teams snaps, games active and started, honours; seasons available and active; first and last active
season; `exited` (played, then no plays through the last season: the earliest tenure end, right-censored);
`gap_seasons` (inactive seasons between the first and last active one, e.g. injury); player_play coverage of
career scrimmage snaps.

Writes data/processed/outcomes_seasons.parquet and data/processed/outcomes.parquet (not committed),
reports/outcomes.md (aggregates only) and reports/figures/outcomes.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

KEY = ["player_id", "season"]
BANDS = ("1", "2-3", "4-5", "6-7", "UDFA")


def read_player_play(raw_dir: Path, season_types) -> pl.DataFrame:
    """Prospect-plays of the kept season types: keys, season, week, team and whether the team had the ball."""
    games = pl.read_csv(raw_dir / "games.csv").select("game_id", "season", "season_type", "week")
    return (
        pl.scan_csv(raw_dir / "player_play.csv", null_values="NA", infer_schema_length=100_000)
        .select("game_id", "play_id", pl.col("nfl_id").alias("player_id"), "team_abbr", "possession_team")
        .join(games.lazy(), on="game_id")
        .filter(pl.col("season_type").is_in(season_types))
        .with_columns((pl.col("team_abbr") == pl.col("possession_team")).alias("offense"))
        .collect()
    )


def round_band(round_col: str = "draft_round") -> pl.Expr:
    r = pl.col(round_col)
    return (
        pl.when(r.is_null())
        .then(pl.lit("UDFA"))
        .when(r == 1)
        .then(pl.lit("1"))
        .when(r <= 3)
        .then(pl.lit("2-3"))
        .when(r <= 5)
        .then(pl.lit("4-5"))
        .otherwise(pl.lit("6-7"))
        .cast(pl.Enum(BANDS))
    )


def season_grid(players: pl.DataFrame, last_season: int) -> pl.DataFrame:
    """One row per prospect per league season, draft year to last_season, with the year in the league (1 = rookie)."""
    return (
        players.select("player_id", "draft_year")
        .with_columns(pl.int_ranges("draft_year", last_season + 1).alias("season"))
        .explode("season")
        .with_columns((pl.col("season") - pl.col("draft_year") + 1).alias("year"))
        .select(*KEY, "year")
    )


def season_playing_time(pp: pl.DataFrame) -> pl.DataFrame:
    """Per player-season with a play: plays, offensive and defensive plays, games with a play, team (most plays)."""
    return pp.group_by(KEY).agg(
        pl.len().alias("plays"),
        pl.col("offense").sum().alias("off_plays"),
        (~pl.col("offense")).sum().alias("def_plays"),
        pl.col("game_id").n_unique().alias("games"),
        pl.col("team_abbr").mode().sort().first().alias("team"),
    )


def player_seasons(grid: pl.DataFrame, playing_time: pl.DataFrame) -> pl.DataFrame:
    """Every league season of every prospect, zero where they had no play; `active` = at least one play."""
    counts = ["plays", "off_plays", "def_plays", "games"]
    return (
        grid.join(playing_time, on=KEY, how="left")
        .with_columns(pl.col(c).fill_null(0).cast(pl.Int64) for c in counts)
        .with_columns((pl.col("plays") > 0).alias("active"))
        .sort(KEY)
    )


def careers(seasons: pl.DataFrame, players: pl.DataFrame, successes: pl.DataFrame, last_season: int) -> pl.DataFrame:
    """Per prospect: draft slot, career totals, seasons active, exit and gap seasons, player_play coverage."""
    active = pl.col("season").filter("active")
    per = seasons.group_by("player_id").agg(
        pl.len().alias("seasons_available"),
        pl.col("active").sum().alias("seasons_active"),
        active.min().alias("first_active"),
        active.max().alias("last_active"),
        pl.col("plays").sum().alias("pp_plays"),
        pl.col("games").sum().alias("pp_games"),
    )
    car = successes.select(
        "player_id",
        (pl.col("career_offensive_snaps") + pl.col("career_defensive_snaps")).alias("scrim_snaps"),
        pl.col("career_special_teams_snaps").alias("st_snaps"),
        pl.col("career_games_active").alias("games_active"),
        pl.col("career_games_started").alias("games_started"),
        (pl.col("ap_all_pro_1st_team") + pl.col("ap_all_pro_2nd_team")).alias("all_pro"),
        pl.col("pro_bowl_original_ballot").alias("pro_bowl"),
    )
    return (
        players.select(
            "player_id",
            "draft_year",
            "draft_round",
            "draft_pick_within_round",
            "draft_overall_pick",
            pl.col("draft_round").is_null().alias("undrafted"),
            round_band().alias("round_band"),
        )
        .join(per, on="player_id", how="left")
        .join(car, on="player_id", how="left")
        .with_columns(
            (pl.col("scrim_snaps") == 0).alias("never_played"),
            ((pl.col("seasons_active") > 0) & (pl.col("last_active") < last_season)).alias("exited"),
            (pl.col("last_active") - pl.col("first_active") + 1 - pl.col("seasons_active"))
            .fill_null(0)
            .alias("gap_seasons"),
            (pl.col("scrim_snaps") / pl.col("seasons_available")).alias("scrim_per_season"),
            (pl.col("games_started") / pl.col("seasons_available")).alias("starts_per_season"),
            pl.when(pl.col("scrim_snaps") > 0).then(pl.col("pp_plays") / pl.col("scrim_snaps")).alias("pp_coverage"),
        )
        .sort("player_id")
    )


def by_class(car: pl.DataFrame, seasons: pl.DataFrame) -> pl.DataFrame:
    """Per draft class: size, undrafted, never played, exits, gap seasons, active share by year in the league."""
    act = (
        seasons.join(car.select("player_id", "draft_year"), on="player_id")
        .group_by("draft_year", "year")
        .agg(pl.col("active").mean())
        .with_columns(pl.format("active_y{}", "year").alias("k"))
        .pivot(on="k", index="draft_year", values="active")
    )
    agg = car.group_by("draft_year").agg(
        pl.len().alias("n"),
        pl.col("undrafted").sum(),
        pl.col("never_played").sum(),
        (pl.col("seasons_active") == 0).sum().alias("no_pp_play"),
        pl.col("exited").sum(),
        (pl.col("gap_seasons") > 0).sum().alias("with_gap"),
        pl.col("scrim_snaps").median().alias("scrim_median"),
    )
    return agg.join(act, on="draft_year").sort("draft_year")


def by_band(car: pl.DataFrame, last_season: int) -> pl.DataFrame:
    """Per round band: never played, and exits among prospects who could exit (first play before the last season)."""
    can_exit = pl.col("first_active") < last_season
    return (
        car.group_by("round_band")
        .agg(
            pl.len().alias("n"),
            pl.col("never_played").mean().alias("never_played_share"),
            can_exit.sum().alias("could_exit"),
            pl.col("exited").filter(can_exit).sum().alias("exited"),
            pl.col("exited").filter(can_exit).mean().alias("exit_share"),
        )
        .sort("round_band")
    )


def coverage(car: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """player_play plays / career scrimmage snaps by group (pooled and median per prospect)."""
    return (
        car.join(groups, on="player_id")
        .filter(pl.col("scrim_snaps") > 0)
        .group_by("group")
        .agg(
            pl.len().alias("n"),
            (pl.col("pp_plays").sum() / pl.col("scrim_snaps").sum()).alias("pooled"),
            pl.col("pp_coverage").median().alias("median"),
            (pl.col("pp_coverage") >= 0.99).mean().alias("complete_share"),
        )
    )


def draft_dependence(car: pl.DataFrame, groups: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Median career scrimmage snaps per available season by group × round band; per group, the Spearman ρ of
    snaps per season with overall pick (drafted only; pooled and within draft class); prospects per band."""
    from scipy.stats import spearmanr

    d = car.join(groups, on="player_id")
    bands = (
        d.group_by("group", "round_band")
        .agg(pl.col("scrim_per_season").median())
        .pivot(on="round_band", index="group", values="scrim_per_season", sort_columns=True)
    )
    n = d.group_by("round_band").len().sort("round_band")
    bands = bands.select("group", *(b for b in BANDS if b in bands.columns))
    rows = []
    for (g,), x in d.filter(~pl.col("undrafted")).group_by("group"):
        # rank within draft class so class length (seasons available) doesn't enter
        x = x.with_columns(
            pl.col("draft_overall_pick").rank().over("draft_year").alias("pick_r"),
            pl.col("scrim_snaps").rank().over("draft_year").alias("snap_r"),
        )
        rho = spearmanr(x["draft_overall_pick"], x["scrim_per_season"]).statistic
        within = np.corrcoef(
            x.with_columns((pl.col(c) - pl.col(c).mean().over("draft_year")) for c in ("pick_r", "snap_r"))
            .select("pick_r", "snap_r")
            .to_numpy()
            .T
        )[0, 1]
        und = d.filter((pl.col("group") == g) & pl.col("undrafted"))
        rows.append(
            dict(
                group=g,
                drafted=x.height,
                rho_pooled=rho,
                rho_within_class=within,
                undrafted=und.height,
                undrafted_never_played=und["never_played"].mean(),
                drafted_never_played=x["never_played"].mean(),
            )
        )
    return bands, pl.DataFrame(rows), n


def plot_outcomes(car: pl.DataFrame, seasons: pl.DataFrame, groups: pl.DataFrame, order, path) -> None:
    """Left: share of prospects with a play by year in the league, per class. Right: median career snaps per
    available season by round band, per group."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    s = seasons.join(car.select("player_id", "draft_year"), on="player_id")
    for (cls,), x in s.group_by("draft_year", maintain_order=True):
        a = x.group_by("year").agg(pl.col("active").mean()).sort("year")
        ax0.plot(a["year"], a["active"], marker="o", label=f"{cls} class")
    ax0.set_xticks([1, 2, 3])
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("year in the league")
    ax0.set_ylabel("share with a regular-season play")
    ax0.set_title("Who is still on the field")
    ax0.legend(fontsize=8)
    d = car.join(groups, on="player_id")
    for g in order:
        m = d.filter(pl.col("group") == g).group_by("round_band").agg(pl.col("scrim_per_season").median())
        m = m.sort("round_band")
        ax1.plot([BANDS.index(b) for b in m["round_band"]], m["scrim_per_season"], marker="o", label=g)
    ax1.set_xticks(range(len(BANDS)), BANDS)
    ax1.set_xlabel("draft round")
    ax1.set_ylabel("career scrimmage snaps per season (median)")
    ax1.set_title("Playing time falls with draft slot")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.clustering.readout import POS_ORDER
    from eeg.config import PROJECT_ROOT
    from eeg.ingest.load_combine import RAW_DIR
    from eeg.ngs.channels import describe

    cfg = config.load()
    last, types = cfg["outcomes"]["last_season"], cfg["games"]["season_types"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    groups = pl.read_parquet(interim / "combine_si.parquet").select("player_id", pl.col("pos_group").alias("group"))
    players = pl.read_csv(RAW_DIR / "players.csv", null_values="NA").rename({"nfl_id": "player_id"})
    successes = pl.read_csv(RAW_DIR / "player_career_successes.csv", null_values="NA").rename({"nfl_id": "player_id"})

    pp = read_player_play(RAW_DIR, types)
    seasons = player_seasons(season_grid(players, last), season_playing_time(pp))
    car = careers(seasons, players, successes, last)
    seasons.write_parquet(processed / "outcomes_seasons.parquet")
    car.write_parquet(processed / "outcomes.parquet")

    enum = pl.Enum(POS_ORDER)
    cov = coverage(car, groups).with_columns(pl.col("group").cast(enum)).sort("group")
    bands, rho, band_n = draft_dependence(car, groups)
    bands = bands.with_columns(pl.col("group").cast(enum)).sort("group")
    rho = rho.with_columns(pl.col("group").cast(enum)).sort("group")
    act = seasons.filter("active").join(groups, on="player_id").with_columns(pl.col("group").cast(enum))
    act = act.sort("group", "year").rename({"group": "pos_group"})
    plot_outcomes(car, seasons, groups, POS_ORDER, reports / "figures" / "outcomes.png")

    report = f"""# Regular-season outcomes

Generated by `uv run python -m eeg.outcomes.table`. Competition data only: `player_play.csv` ({" + ".join(types)}),
`players.csv`, `player_career_successes.csv`. Careers are right-censored after the {last} season. Per-season and
per-prospect tables are in `data/processed/outcomes_seasons.parquet` and `outcomes.parquet` (not committed).
Aggregates only; position groups are used to read results.

- Prospects: {car.height}; player-seasons in the league: {seasons.height} ({seasons["active"].sum()} with a play).
- Undrafted: {car["undrafted"].sum()} ({car["undrafted"].mean():.0%}). Never played a scrimmage snap:
  {car["never_played"].sum()}.

## What a play is: player_play against career snaps

Career snaps are regular-season only (player_play never exceeds them). For OL, DL and DB, player_play holds every
scrimmage snap. For WR and TE it holds only the plays on which they ran a route, so their "plays" are routes, and
run-blocking snaps are missing (game tracking has the same plays). Playing time is compared within group only.

{_md(cov, 2)}

## By draft class

`never_played`: no career scrimmage snap. `no_pp_play`: no play in player_play (adds WR/TE with snaps but no
route). `exited`: played, then no play through {last} (the earliest tenure end; the {last} class can't exit yet).
`with_gap`: an inactive season between two active ones. `active_yN`: share of the class with a play in their Nth
season.

{_md(by_class(car, seasons), 2)}

## Plays per active season (player_play)

Routes for WR and TE, all scrimmage snaps otherwise.

{_md(describe(act, ["plays", "games"], by="pos_group"), 0)}

By year in the league (all groups; median plays):

{_md(act.group_by("year").agg(pl.len().alias("n"), pl.col("plays").median(), pl.col("games").median()).sort("year"), 0)}

## How playing time depends on draft slot (open decision 2)

Median career scrimmage snaps per available season, by group and round band (prospects per band:
{", ".join(f"{r['round_band']} {r['len']}" for r in band_n.iter_rows(named=True))}):

{_md(bands, 0)}

Spearman ρ of snaps per season with overall pick among drafted prospects (pooled, and within draft class on
ranks), and the share that never played:

{_md(rho, 2)}

## Exits by draft slot (open decision 3)

`could_exit`: first play before {last}, so there was a later season to miss.

{_md(by_band(car, last), 2)}

Figure: `reports/figures/outcomes.png`.
"""
    (reports / "outcomes.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
