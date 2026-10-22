"""Field and lab channels: speed, kinetic energy, burst acceleration, propulsive power (per kg).

    uv run python -m eeg.ngs.channels

Per frame, from reported speed v (`s_mps`) and a_s = smoothed dv/dt (`a_s_mps2`):
- E_k / m = ½ v²                                  [J/kg]
- P / m   = (a_s + k v²) v,  k = ½ ρ C_d A / m     [W/kg]  (propulsive: inertia + drag)
  with A from Combine height and mass (`sprint.frontal_area_bsa_factor`, as for E_40).
  Mass is Combine mass on both sides, so it cancels except in the small drag term.

Peaks are "sustained": the frame value must hold for `games.sustain_frames`
(rolling minimum). Per play (clean tracks only, glitch tracks dropped):
- s_peak        sustained peak speed; ke_peak = ½ s_peak²
- a_peak        sustained peak acceleration anywhere in the play
- burst_a       sustained peak acceleration in [0, `games.burst_window_s`] after the
                snap, only on plays starting below `games.burst_start_speed_mps`
- p_peak        sustained peak propulsive power per kg
Per game: the maximum of each over the game's plays. Per player-season, over games with at least
`games.min_game_plays` plays, candidate statistics for each channel: `{ch}_max` (season peak),
`{ch}_game_med` (median of game maxima), `{ch}_top3` (mean of the 3 highest game maxima) and
`{ch}_top3_of{K}` (expected mean of the 3 highest plays in K plays drawn at random from the season,
K = `games.season_sample_plays`; computed exactly over all K-subsets). Maxima over games grow with
exposure (more games and snaps, more chances at a fast one), so they rise with playing time; the
fixed-K statistic is the same functional for every player whatever their exposure (a U-statistic).
`qualified` marks seasons with enough exposure (`games.min_season_plays`, `games.min_season_games`).
The season value used downstream is `games.season_statistic` (top3_of50; see eeg.ngs.reliability).

Lab: the same estimators on each prospect's best 40 rep up to the finish line
(the 40 starts from rest, so its acceleration peak is the burst analogue), plus
the 40 model's v_max, a0 = v_max/τ and peak power v_max²/(4τ).

Writes data/processed/field_{plays,games,seasons}.parquet, data/processed/lab_channels.parquet
(not committed) and reports/field_channels.md (aggregates only).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import gammaln

from eeg.ngs.kinematics import savgol_track
from eeg.physics.sprint import frontal_area

TRACK = ["game_id", "play_id", "player_id"]
CHANNELS = ("s_peak", "ke_peak", "a_peak", "burst_a", "p_peak")
LAB = {  # field channel -> lab channel with the same estimator
    "s_peak": "lab_s_peak",
    "ke_peak": "lab_ke_peak",
    "a_peak": "lab_a_peak",
    "burst_a": "lab_a_peak",
    "p_peak": "lab_p_peak",
}
TOP_K = 3


def drag_coef(prospects: pl.DataFrame, sprint: dict) -> pl.DataFrame:
    """Per player k = ½ ρ C_d A / m [1/m], so drag deceleration is k v²."""
    area = frontal_area(
        prospects["ht_m"].to_numpy(), prospects["mass_kg"].to_numpy(), sprint["frontal_area_bsa_factor"]
    )
    k = 0.5 * sprint["air_density"] * sprint["drag_coefficient"] * area / prospects["mass_kg"].to_numpy()
    return prospects.select("player_id").with_columns(pl.Series("drag_k", k))


def _frame_power(speed: str, accel: str) -> pl.Expr:
    return ((pl.col(accel) + pl.col("drag_k") * pl.col(speed) ** 2) * pl.col(speed)).alias("p_per_kg")


def play_channels(frames: pl.DataFrame, drag: pl.DataFrame, games_cfg: dict) -> pl.DataFrame:
    """One row per clean track: sustained peaks of speed, acceleration, burst acceleration and power."""
    n = games_cfg["sustain_frames"]
    at_snap = pl.col("t_snap_s").abs() < 1e-6
    f = (
        frames.filter(~pl.col("glitch_track"))
        .join(drag, on="player_id", how="left")
        .with_columns(_frame_power("s_mps", "a_s_mps2"))
        .with_columns(
            pl.col("s_mps").rolling_min(n).over(TRACK).alias("s_sust"),
            pl.col("a_s_mps2").rolling_min(n).over(TRACK).alias("a_sust"),
            pl.col("p_per_kg").rolling_min(n).over(TRACK).alias("p_sust"),
            pl.col("s_mps").filter(at_snap).first().over(TRACK).alias("s_at_snap"),
        )
    )
    eligible = pl.col("s_at_snap") < games_cfg["burst_start_speed_mps"]
    in_window = pl.col("t_snap_s").is_between(0, games_cfg["burst_window_s"])
    return (
        f.group_by(*TRACK, "season", "season_type", "week", maintain_order=True)
        .agg(
            pl.col("s_sust").max().alias("s_peak"),
            pl.col("a_sust").max().alias("a_peak"),
            pl.col("a_sust").filter(eligible & in_window).max().alias("burst_a"),
            pl.col("p_sust").max().alias("p_peak"),
            eligible.first().fill_null(False).alias("burst_eligible"),
        )
        .with_columns((0.5 * pl.col("s_peak") ** 2).alias("ke_peak"))
        .select(*TRACK, "season", "season_type", "week", "burst_eligible", *CHANNELS)
    )


def game_channels(plays: pl.DataFrame) -> pl.DataFrame:
    """Per player-game: plays, burst-eligible plays, and the maximum of each channel."""
    return (
        plays.group_by("player_id", "season", "season_type", "week", "game_id")
        .agg(
            pl.len().alias("plays"),
            pl.col("burst_eligible").sum().alias("burst_plays"),
            *(pl.col(c).max() for c in CHANNELS),
        )
        .sort("player_id", "season", "game_id")
    )


def season_stats(channel: str, top_k: int = TOP_K) -> dict[str, pl.Expr]:
    """The candidate season statistics of one channel over a set of game rows, keyed by column suffix."""
    c = pl.col(channel)
    return {
        "max": c.max(),
        "game_med": c.median(),
        f"top{top_k}": c.drop_nulls().sort(descending=True).head(top_k).mean(),
    }


def expected_top(values: np.ndarray, sample: int, top: int = TOP_K) -> float:
    """Expected mean of the `top` largest of `sample` values drawn without replacement from values (exact, over
    all subsets); NaN with fewer than `sample` values. The i-th largest value is the j-th largest of a random
    subset with probability C(i−1, j−1) C(N−i, K−j) / C(N, K)."""
    v, n = np.sort(np.asarray(values, float))[::-1], len(values)
    if n < sample:
        return float("nan")

    def lcomb(a, b):
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)

    i = np.arange(1, n + 1)
    w = np.zeros(n)
    for j in range(1, top + 1):
        ok = (i - 1 >= j - 1) & (n - i >= sample - j)
        w[ok] += np.exp(lcomb(i[ok] - 1, j - 1) + lcomb(n - i[ok], sample - j) - lcomb(n, sample))
    return float(w @ v / top)


def play_season_stats(
    plays: pl.DataFrame, games: pl.DataFrame, sample: int, channels=CHANNELS, top: int = TOP_K, by=("season",)
) -> pl.DataFrame:
    """Per player-season (or player × `by`): `{ch}_top{top}_of{sample}` over plays in the given (counting) games."""
    key = ["player_id", *by]
    rows = []
    p = plays.join(games.select("player_id", "game_id").unique(), on=["player_id", "game_id"])
    for k, r in p.group_by(key, maintain_order=True):
        rows.append(dict(zip(key, k)) | {c: expected_top(r[c].drop_nulls().to_numpy(), sample, top) for c in channels})
    out = pl.DataFrame(rows, schema={**{c: p.schema[c] for c in key}, **{c: pl.Float64 for c in channels}})
    return out.with_columns(pl.col(c).fill_nan(None).alias(f"{c}_top{top}_of{sample}") for c in channels).drop(channels)


def counting_games(games: pl.DataFrame, min_game_plays: int) -> pl.DataFrame:
    """Games that enter season statistics: at least min_game_plays plays."""
    return games.filter(pl.col("plays") >= min_game_plays)


def season_channels(
    games: pl.DataFrame, min_game_plays: int = 1, min_plays: int = 0, min_games: int = 0, top_k: int = TOP_K
) -> pl.DataFrame:
    """Per player-season, over counting games: exposure, a `qualified` flag, and per channel the season peak,
    median game max and mean of the top-k game maxima."""
    stats = [e.alias(f"{c}_{k}") for c in CHANNELS for k, e in season_stats(c, top_k).items()]
    return (
        counting_games(games, min_game_plays)
        .group_by("player_id", "season")
        .agg(
            pl.len().alias("games"),
            pl.col("plays").sum(),
            pl.col("burst_plays").sum(),
            pl.col("burst_a").is_not_null().sum().alias("burst_games"),
            *stats,
        )
        .with_columns(((pl.col("plays") >= min_plays) & (pl.col("games") >= min_games)).alias("qualified"))
        .sort("player_id", "season")
    )


def lab_channels(
    drills: pl.DataFrame, best: pl.DataFrame, drag: pl.DataFrame, games_cfg: dict, per_rep: bool = False
) -> pl.DataFrame:
    """Per prospect, on the best 40 rep up to the finish: the field estimators and the 40 model values.

    ``best`` lists the reps to use (rep_id, v_max, tau, t_finish). With per_rep, one row per rep
    (rep_id kept) instead of per prospect, e.g. to measure rep-to-rep reliability.
    """
    key = ["player_id", "rep_id"] if per_rep else ["player_id"]
    n = games_cfg["sustain_frames"]
    reps = (
        drills.filter(pl.col("drill") == "forty")
        .join(best.select("rep_id", "v_max", "tau", "t_finish"), on="rep_id")
        .sort("rep_id", "t_s")
        .with_columns(
            pl.int_range(pl.len()).over("rep_id").alias("frame_id"), pl.len().over("rep_id").alias("n_frames")
        )
    )
    a_s = savgol_track(
        reps["s_mps"].to_numpy(), reps["frame_id"].to_numpy(), reps["n_frames"].to_numpy(),
        games_cfg["accel_window_frames"], dt=games_cfg["frame_dt_s"],
    )  # fmt: skip
    return (
        reps.with_columns(pl.Series("a_s", a_s).fill_nan(None))
        .join(drag, on="player_id", how="left")
        .with_columns(_frame_power("s_mps", "a_s"))
        .with_columns(
            pl.col("s_mps").rolling_min(n).over("rep_id").alias("s_sust"),
            pl.col("a_s").rolling_min(n).over("rep_id").alias("a_sust"),
            pl.col("p_per_kg").rolling_min(n).over("rep_id").alias("p_sust"),
        )
        .filter(pl.col("t_s") <= pl.col("t_finish"))
        .group_by(key)
        .agg(
            pl.col("s_sust").max().alias("lab_s_peak"),
            pl.col("a_sust").max().alias("lab_a_peak"),
            pl.col("p_sust").max().alias("lab_p_peak"),
            pl.col("v_max").first().alias("lab_v_max"),
            (pl.col("v_max") / pl.col("tau")).first().alias("lab_a0"),
            (pl.col("v_max") ** 2 / (4 * pl.col("tau"))).first().alias("lab_p_model"),
        )
        .with_columns((0.5 * pl.col("lab_s_peak") ** 2).alias("lab_ke_peak"))
        .sort(key)
    )


def describe(seasons: pl.DataFrame, cols, by: str | None = None) -> pl.DataFrame:
    """Median and p10/p90 of each column (optionally by group)."""
    rows = []
    groups = [(None, seasons)] if by is None else [(k[0], g) for k, g in seasons.group_by(by, maintain_order=True)]
    for key, g in groups:
        for c in cols:
            v = g[c].drop_nulls().to_numpy()
            if len(v) == 0:
                continue
            q = np.quantile(v, [0.1, 0.5, 0.9])
            rows.append(
                {**({by: key} if by else {}), "channel": c, "n": len(v), "p10": q[0], "median": q[1], "p90": q[2]}
            )
    return pl.DataFrame(rows)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.clustering.readout import POS_ORDER
    from eeg.config import PROJECT_ROOT
    from eeg.ingest.load_games import SEASONS
    from eeg.ngs.validate import compare
    from eeg.physics.sprint import best_reps

    cfg = config.load()
    g = cfg["games"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    prospects = pl.read_parquet(interim / "combine_si.parquet").select("player_id", "pos_group", "ht_m", "mass_kg")
    drag = drag_coef(prospects, cfg["sprint"])
    cols = ["player_id", "season", "season_type", "week", "game_id", "play_id", "t_snap_s", "s_mps", "a_s_mps2",
            "glitch_track"]  # fmt: skip
    plays = pl.concat(
        [play_channels(pl.read_parquet(interim / f"game_tracking_{s}.parquet", columns=cols), drag, g) for s in SEASONS]
    )
    games = game_channels(plays)
    seasons = season_channels(games, g["min_game_plays"], g["min_season_plays"], g["min_season_games"]).join(
        play_season_stats(plays, counting_games(games, g["min_game_plays"]), g["season_sample_plays"]),
        on=["player_id", "season"],
        how="left",
    )
    best = best_reps(pl.read_parquet(interim / "forty_fits.parquet"), cfg["sprint"]["max_fit_rmse_mps"])
    lab = lab_channels(pl.read_parquet(interim / "timed_drills.parquet"), best, drag, g)
    for name, df in [("field_plays", plays), ("field_games", games), ("field_seasons", seasons), ("lab_channels", lab)]:
        df.write_parquet(processed / f"{name}.parquet")

    stat = g["season_statistic"]
    q = seasons.filter("qualified").join(prospects.select("player_id", "pos_group"), on="player_id")
    q = q.with_columns(pl.col("pos_group").cast(pl.Enum(POS_ORDER))).sort("pos_group", "player_id", "season")
    peaks = [f"{c}_{stat}" for c in CHANNELS]
    pooled = describe(q, peaks)
    by_pos = describe(q, [f"{c}_{stat}" for c in ("s_peak", "burst_a", "p_peak")], by="pos_group")
    exposure = describe(q, ["games", "plays", "burst_plays", "burst_games"], by="pos_group").select(
        "pos_group", "channel", "median", "p10", "p90"
    )
    corr = q.select(peaks).drop_nulls()
    corr_tbl = pl.DataFrame(np.corrcoef(corr.to_numpy().T), schema=peaks).insert_column(0, pl.Series("", peaks))
    both = q.join(lab, on="player_id")
    lab_cmp = compare(both, {f"{c}_{stat}": LAB[c] for c in CHANNELS}).select(
        "field", "lab", "n", "field_above_lab", "ratio_median", "pearson", "spearman"
    )
    lab_model = compare(lab, {"lab_s_peak": "lab_v_max", "lab_a_peak": "lab_a0", "lab_p_peak": "lab_p_model"}).select(
        "field", "lab", "n", "ratio_median", "pearson", "spearman"
    )

    report = f"""# Field channels

Generated by `uv run python -m eeg.ngs.channels`. Clean tracks only (glitch tracks dropped), {" + ".join(g["season_types"])}.
Peaks hold for {g["sustain_frames"]} frames. Burst: first {g["burst_window_s"]} s after the snap, plays starting below
{g["burst_start_speed_mps"]} m/s. Per-play, per-game and per-season tables are in `data/processed/field_*.parquet`
and the lab side in `data/processed/lab_channels.parquet` (not committed). Aggregates only.

- Plays: {plays.height:,} ({plays["burst_eligible"].mean():.0%} burst-eligible); player-games: {games.height:,};
  player-seasons: {seasons.height} ({q.height} qualified, {q["player_id"].n_unique()} prospects).
- Season statistic `{stat}` (`games.season_statistic`): expected mean of the 3 best plays in {g["season_sample_plays"]} random
  plays of the season (exposure-free), over plays in counting games (≥ {g["min_game_plays"]} plays).
  Qualified: ≥ {g["min_season_plays"]} plays and ≥ {g["min_season_games"]} counting games.
- Units: speed m/s, ke J/kg, acceleration m/s², power W/kg.

## Season values (qualified player-seasons)

{_md(pooled, 2)}

By position group (post hoc reading only):

{_md(by_pos, 2)}

## Exposure by position group (per player-season)

`burst_plays`: plays starting from near rest; `burst_games`: games with at least one.

{_md(exposure, 0)}

## Correlation between season channels (Pearson, {corr.height} player-seasons)

{_md(corr_tbl, 2)}

## First look: season value against the lab 40 (same estimators)

Not η yet: role dependence is decided in step 5.

{_md(lab_cmp, 3)}

Inside the lab, measured 40 peaks against the 40 model ({lab.height} prospects):

{_md(lab_model, 3)}
"""
    (reports / "field_channels.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
