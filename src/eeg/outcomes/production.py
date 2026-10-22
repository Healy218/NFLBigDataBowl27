"""Position production per player-season from player_play, with coverage, reliability and validity checks.

    uv run python -m eeg.outcomes.production

Every measure is a ratio of season sums, num / den, so a season value weights plays equally, and the same sums
give half-season values for split-half reliability. Opportunities (the denominator) depend on the group:
- WR, TE (rows are route plays): target rate, receiving yards per route run (YPRR), mean separation when the
  ball is thrown, yards after catch over expected per reception.
- OL: pass blocks (`pressure_allowed` set): pressure allowed rate, mean peak pressure probability allowed, sack
  allowed rate. Lower is better. Run blocks carry no player columns.
- DL: pass rushes (`blitzing` on a pass play): pressure rate (`time_to_pressure` set) and sack rate; get-off
  time (s, mean over plays where it is measured; lower is faster); tackle involvements (solo or assist) and
  tackles for loss per snap.
- DB: tackle involvements and tackles for loss per snap. player_play has no coverage outcome for the defender
  (no targeted-defender column), so DB production is thin.
- Every group: plays per game played (snap volume when active; routes per game for WR/TE).

Groups decide which measures apply (scope) and are used to read results; they are never model features.

Checks per measure, on player-seasons with ≥ `games.min_season_plays` plays and ≥ `games.min_season_games`
games with a play:
- split-half reliability: games in order alternate between halves, r between half values within group, stepped
  up with Spearman–Brown;
- retest: r between consecutive qualified seasons of a prospect, within group;
- validity: within-group r with plays per game (what the team reveals by playing him), and Spearman ρ with
  overall pick (drafted only, within group).

Writes data/processed/production_seasons.parquet (not committed) and reports/production.md (aggregates only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from eeg.ngs.reliability import spearman_brown

KEY = ["player_id", "season"]


@dataclass(frozen=True)
class Measure:
    name: str
    groups: tuple[str, ...]
    num: pl.Expr  # per play; summed
    den: pl.Expr  # per play; summed (opportunities)
    better: str  # "higher" or "lower"
    label: str


def _b(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64).fill_null(0.0)


PASS = pl.col("pass_result").fill_null("") != ""
RUSH = PASS & pl.col("blitzing").fill_null(False)
BLOCK = pl.col("pressure_allowed").is_not_null()
YACOE = pl.col("yards_after_catch") - pl.col("expected_yards_after_catch")
INVOLVED = (pl.col("tackle").fill_null(0) + pl.col("assist").fill_null(0)).clip(0, 1).cast(pl.Float64)
REC = ("WR", "TE")

MEASURES = (
    Measure("plays_per_game", ("WR", "TE", "OL", "DL", "DB"), pl.lit(1.0), pl.lit(1.0), "higher", "plays per game"),
    Measure("target_rate", REC, _b("target"), pl.lit(1.0), "higher", "targets per route"),
    Measure("yprr", REC, _b("rec_yards"), pl.lit(1.0), "higher", "receiving yards per route run"),
    Measure(
        "separation",
        REC,
        pl.col("separation_at_pass_forward").fill_null(0.0),
        pl.col("separation_at_pass_forward").is_not_null().cast(pl.Float64),
        "higher",
        "separation at the throw (yd)",
    ),
    Measure("yac_oe", REC, YACOE.fill_null(0.0), YACOE.is_not_null().cast(pl.Float64), "higher", "YAC over expected"),
    Measure(
        "pressure_allowed_rate",
        ("OL",),
        _b("pressure_allowed"),
        BLOCK.cast(pl.Float64),
        "lower",
        "pressures allowed per pass block",
    ),
    Measure(
        "peak_pressure_prob",
        ("OL",),
        pl.col("peak_pressure_probability_allowed").fill_null(0.0),
        pl.col("peak_pressure_probability_allowed").is_not_null().cast(pl.Float64),
        "lower",
        "peak pressure probability allowed",
    ),
    Measure(
        "sack_allowed_rate",
        ("OL",),
        _b("sack_allowed"),
        BLOCK.cast(pl.Float64),
        "lower",
        "sacks allowed per pass block",
    ),
    Measure(
        "pressure_rate",
        ("DL",),
        (RUSH & pl.col("time_to_pressure").is_not_null()).cast(pl.Float64),
        RUSH.cast(pl.Float64),
        "higher",
        "pressures per pass rush",
    ),
    Measure(
        "sack_rate",
        ("DL",),
        pl.when(RUSH).then(_b("sack")).otherwise(0.0),
        RUSH.cast(pl.Float64),
        "higher",
        "sacks per pass rush",
    ),
    Measure(
        "get_off",
        ("DL",),
        pl.col("player_get_off").fill_null(0.0),
        pl.col("player_get_off").is_not_null().cast(pl.Float64),
        "lower",
        "get-off time (s)",
    ),
    Measure("tackle_rate", ("DL", "DB"), INVOLVED, pl.lit(1.0), "higher", "tackle involvements per snap"),
    Measure("tfl_rate", ("DL", "DB"), _b("tackle_for_loss"), pl.lit(1.0), "higher", "tackles for loss per snap"),
)

COLUMNS = [
    "game_id", "play_id", "nfl_id", "pass_result", "blitzing", "target", "rec_yards", "separation_at_pass_forward",
    "yards_after_catch", "expected_yards_after_catch", "pressure_allowed", "peak_pressure_probability_allowed",
    "sack_allowed", "time_to_pressure", "sack", "player_get_off", "tackle", "assist", "tackle_for_loss",
]  # fmt: skip


def read_plays(raw_dir: Path, season_types) -> pl.DataFrame:
    games = pl.read_csv(raw_dir / "games.csv").select("game_id", "season", "season_type")
    return (
        pl.scan_csv(raw_dir / "player_play.csv", null_values="NA", infer_schema_length=100_000)
        .select(COLUMNS)
        .rename({"nfl_id": "player_id"})
        .join(games.lazy(), on="game_id")
        .filter(pl.col("season_type").is_in(season_types))
        .collect()
    )


def game_sums(plays: pl.DataFrame, groups: pl.DataFrame, measures=MEASURES) -> pl.DataFrame:
    """Per player-game: plays and, per measure, num and den sums (den 0 where the measure doesn't apply to the
    player's group). plays_per_game is handled at the season level (plays / games)."""
    d = plays.join(groups, on="player_id")
    exprs = []
    for m in measures:
        if m.name == "plays_per_game":
            continue
        applies = pl.col("group").is_in(m.groups)
        exprs += [
            pl.when(applies).then(m.num * m.den).otherwise(None).alias(f"{m.name}__num"),
            pl.when(applies).then(m.den).otherwise(None).alias(f"{m.name}__den"),
        ]
    sums = [pl.col(c).sum() for e in exprs for c in [e.meta.output_name()]]
    return (
        d.with_columns(exprs)
        .group_by(*KEY, "game_id", "group")
        .agg(pl.len().alias("plays"), *sums)
        .sort(*KEY, "game_id")
    )


def season_values(games: pl.DataFrame, by=KEY, measures=MEASURES) -> pl.DataFrame:
    """Sum player-games over `by` and turn each measure into num / den, plus the opportunity count `{name}_n`
    (both null with no opportunity, including measures that don't apply to the group)."""
    names = [m.name for m in measures if m.name != "plays_per_game"]
    agg = games.group_by(*by, "group").agg(
        pl.col("plays").sum(),
        pl.col("game_id").n_unique().alias("games"),
        *(pl.col(f"{n}__{p}").sum() for n in names for p in ("num", "den")),
    )
    vals = [pl.when(pl.col(f"{n}__den") > 0).then(pl.col(f"{n}__num") / pl.col(f"{n}__den")).alias(n) for n in names]
    ns = [pl.when(pl.col(f"{n}__den") > 0).then(pl.col(f"{n}__den")).alias(f"{n}_n") for n in names]
    return agg.with_columns((pl.col("plays") / pl.col("games")).alias("plays_per_game"), *vals, *ns).select(
        *by, "group", "plays", "games", "plays_per_game", *names, *(f"{n}_n" for n in names)
    )


def qualify(min_plays: int, min_games: int) -> pl.Expr:
    return (pl.col("plays") >= min_plays) & (pl.col("games") >= min_games)


def _within(df: pl.DataFrame, cols) -> pl.DataFrame:
    df = df.drop_nulls(cols)
    return df.with_columns((pl.col(c) - pl.col(c).mean().over("group")).alias(c) for c in cols)


def _r(df: pl.DataFrame, a: str, b: str) -> float:
    return float(np.corrcoef(df[a].to_numpy(), df[b].to_numpy())[0, 1]) if df.height > 2 else float("nan")


def split_half(games: pl.DataFrame, qualified: pl.DataFrame, measures=MEASURES) -> pl.DataFrame:
    """Per measure × group: n, half-season r and Spearman–Brown reliability (qualified player-seasons only)."""
    g = games.join(qualified.select(KEY), on=KEY)
    g = g.with_columns(((pl.col("game_id").rank("ordinal").over(KEY) - 1) % 2).alias("half"))
    h = season_values(g, [*KEY, "half"], measures)
    pairs = (
        h.filter(pl.col("half") == 0)
        .drop("half")
        .join(h.filter(pl.col("half") == 1).drop("half"), on=[*KEY, "group"], suffix="_b")
    )
    rows = []
    for m in measures:
        for grp in m.groups:
            d = pairs.filter(pl.col("group") == grp).drop_nulls([m.name, f"{m.name}_b"])
            r = _r(d, m.name, f"{m.name}_b")
            rows.append(dict(measure=m.name, group=grp, n_half=d.height, r_half=r, rel=spearman_brown(r)))
    return pl.DataFrame(rows)


def retest(qualified: pl.DataFrame, measures=MEASURES) -> pl.DataFrame:
    """Per measure × group: r between consecutive qualified seasons."""
    nxt = qualified.with_columns(pl.col("season") - 1)
    pairs = qualified.join(nxt, on=[*KEY, "group"], suffix="_next")
    rows = []
    for m in measures:
        for grp in m.groups:
            d = pairs.filter(pl.col("group") == grp).drop_nulls([m.name, f"{m.name}_next"])
            rows.append(dict(measure=m.name, group=grp, pairs=d.height, retest=_r(d, m.name, f"{m.name}_next")))
    return pl.DataFrame(rows)


def validity(qualified: pl.DataFrame, draft: pl.DataFrame, measures=MEASURES) -> pl.DataFrame:
    """Per measure × group: r with plays per game, and Spearman ρ with overall pick (drafted, first qualified
    season, so each prospect counts once)."""
    from scipy.stats import spearmanr

    first = qualified.sort("season").group_by("player_id").first().join(draft, on="player_id")
    rows = []
    for m in measures:
        for grp in m.groups:
            q = qualified.filter(pl.col("group") == grp).drop_nulls([m.name, "plays_per_game"])
            f = first.filter(pl.col("group") == grp).drop_nulls([m.name, "draft_overall_pick"])
            rho = spearmanr(f[m.name], f["draft_overall_pick"]).statistic if f.height > 2 else float("nan")
            rows.append(
                dict(
                    measure=m.name,
                    group=grp,
                    r_plays_per_game=_r(q, m.name, "plays_per_game") if m.name != "plays_per_game" else float("nan"),
                    rho_pick=rho,
                    n_drafted=f.height,
                )
            )
    return pl.DataFrame(rows)


def coverage(qualified: pl.DataFrame, measures=MEASURES) -> pl.DataFrame:
    """Per measure × group: qualified seasons and prospects with a value, median opportunities and value."""
    rows = []
    for m in measures:
        for grp in m.groups:
            q = qualified.filter(pl.col("group") == grp).drop_nulls(m.name)
            n_col = "plays" if m.name == "plays_per_game" else f"{m.name}_n"
            rows.append(
                dict(
                    measure=m.name,
                    group=grp,
                    better=m.better,
                    seasons=q.height,
                    prospects=q["player_id"].n_unique(),
                    opp_median=q[n_col].median(),
                    median=q[m.name].median(),
                )
            )
    return pl.DataFrame(rows)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.clustering.readout import POS_ORDER
    from eeg.config import PROJECT_ROOT
    from eeg.ingest.load_combine import RAW_DIR

    cfg = config.load()
    g, o = cfg["games"], cfg["outcomes"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    groups = pl.read_parquet(interim / "combine_si.parquet").select("player_id", pl.col("pos_group").alias("group"))
    draft = pl.read_parquet(processed / "outcomes.parquet").select("player_id", "draft_overall_pick")

    games = game_sums(read_plays(RAW_DIR, g["season_types"]), groups)
    seasons = season_values(games).with_columns(
        qualify(g["min_season_plays"], g["min_season_games"]).alias("qualified")
    )
    seasons = seasons.sort(KEY)
    seasons.write_parquet(processed / "production_seasons.parquet")
    qual = seasons.filter("qualified")

    keys = ["measure", "group"]
    table = (
        coverage(qual)
        .join(split_half(games, qual), on=keys)
        .join(retest(qual), on=keys)
        .join(validity(qual, draft), on=keys)
        .with_columns(pl.col("group").cast(pl.Enum(POS_ORDER)))
        .sort("group", maintain_order=True)
        .select("group", "measure", "better", "seasons", "prospects", "opp_median", "median", "rel", "retest",
                "pairs", "r_plays_per_game", "rho_pick")
    )  # fmt: skip
    labels = "\n".join(f"- `{m.name}`: {m.label} ({', '.join(m.groups)})" for m in MEASURES)

    report = f"""# Position production (player_play, {" + ".join(g["season_types"])})

Generated by `uv run python -m eeg.outcomes.production`. Per player-season measures from `player_play.csv`
(competition data only); the table is `data/processed/production_seasons.parquet` (not committed). Aggregates only.
Each measure is a ratio of season sums (num / den). Checks use qualified player-seasons (≥ {g["min_season_plays"]}
plays and ≥ {g["min_season_games"]} games with a play): {qual.height} of {seasons.height}, {qual["player_id"].n_unique()} prospects.

Measures (`better`: direction of good):
{labels}

Columns: `seasons` / `prospects` with a value; `opp_median`: median opportunities per season (routes, pass blocks,
rushes, snaps, or plays); `rel`: split-half reliability within the group (alternate games, Spearman–Brown);
`retest`: r between consecutive seasons (`pairs`); `r_plays_per_game`: within-group r with plays per game (does
the team play him more?); `rho_pick`: Spearman ρ with overall pick in the first qualified season (drafted;
negative = earlier picks score higher).

{_md(table, 2)}

## Decision (open decision 1, `outcomes` in config/physics.yaml)

- Primary outcome, every group: `{o["primary"]}` (all prospects, zeros kept; within group only).
- Secondary production outcome: {", ".join(f"{k} `{v}`" for k, v in o["production"].items())}. DB: none credible.
- In-game explosion used with η, not as an outcome: {", ".join(f"`{v}`" for v in o["realization"])}.
- Binary rates of rare events (sacks, sacks allowed, tackles for loss) are too noisy per season (rel 0.2–0.4);
  continuous or more frequent versions win (peak pressure probability over pressure allowed).
"""
    (reports / "production.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
