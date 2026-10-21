"""Game tracking -> one frame per row, SI units, glitch flags.

    uv run python -m eeg.ingest.load_games

Joins games.csv for season type and week and keeps `games.season_types`
(REG + POST). A track is one prospect on one play: (game_id, play_id, nfl_id),
~1 s before the snap to the end of the play at 10 Hz.

Per frame it adds:
- t_snap_s: time since the track's first `ball_snap` event (null if none)
- v_xy_mps: speed from central x/y differences (one-sided at track ends)
- a_t_mps2: tangential acceleration, the central difference of v_xy_mps. NGS
  `a` is kept as reported (a_mps2) for comparison.
- flags (`games.*` in config/physics.yaml), never dropped. Glitches are
  physically impossible or broken frames:
    flag_gap       step from the previous frame is not frame_dt_s
    flag_jump      the x/y step from the previous frame implies > max_speed_mps
    flag_speed     reported speed > max_speed_mps
    flag_accel     |reported a| > max_accel_mps2
  glitch is any of these; glitch_track marks every frame of a track with one.
  A quality flag, not a glitch:
    flag_mismatch  |reported speed − v_xy_mps| > max_speed_mismatch_mps. Mostly
                   low-speed contact, where positions stop abruptly and the
                   (smoothed) reported speed lags.

The source `dis` (unverified) and `o` (body orientation, not needed for
kinematics) are dropped. Writes data/interim/game_tracking_{season}.parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from eeg.config import PROJECT_ROOT
from eeg.ingest import units as u
from eeg.ingest.load_combine import RAW_DIR

OUT_DIR = PROJECT_ROOT / "data" / "interim"
SEASONS = (2023, 2024, 2025)
TRACK_KEY = ["game_id", "play_id", "nfl_id"]
GLITCH_FLAGS = ["flag_gap", "flag_jump", "flag_speed", "flag_accel"]
FLAGS = [*GLITCH_FLAGS, "flag_mismatch"]


def read_raw(season: int, raw_dir: Path = RAW_DIR) -> pl.LazyFrame:
    return pl.scan_csv(raw_dir / f"game_tracking_{season}.csv", null_values="NA", try_parse_dates=True)


def read_games(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / "games.csv").select("game_id", "season", "season_type", "week")


def central_diff(col: str, t: str = "t_s") -> pl.Expr:
    """d col / d t within a track: central difference, one-sided at the ends, null for one-frame tracks."""
    nxt, prv = pl.col(col).shift(-1).fill_null(pl.col(col)), pl.col(col).shift(1).fill_null(pl.col(col))
    t_nxt, t_prv = pl.col(t).shift(-1).fill_null(pl.col(t)), pl.col(t).shift(1).fill_null(pl.col(t))
    return ((nxt - prv) / (t_nxt - t_prv)).fill_nan(None).over(TRACK_KEY)


def build_game_frames(raw: pl.LazyFrame | pl.DataFrame, games: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Frames of the kept season types, sorted by track and time, with derived kinematics and flags."""
    g = cfg["games"]
    lf = (
        raw.lazy()
        .join(games.lazy(), on="game_id")
        .filter(pl.col("season_type").is_in(g["season_types"]))
        .sort([*TRACK_KEY, "time"])
    )
    t_s = (pl.col("time") - pl.col("time").first()).dt.total_milliseconds() / 1000.0
    lf = lf.with_columns(
        t_s.over(TRACK_KEY).alias("t_s"),
        pl.int_range(pl.len()).over(TRACK_KEY).alias("frame_id"),
        (pl.col("x") * u.yd_to_m).alias("x_m"),
        (pl.col("y") * u.yd_to_m).alias("y_m"),
        (pl.col("s") * u.yd_to_m).alias("s_mps"),
        (pl.col("a") * u.yd_to_m).alias("a_mps2"),
    )
    snap_t = pl.col("t_s").filter(pl.col("event") == "ball_snap").first()
    dt_prev = pl.col("t_s").diff().over(TRACK_KEY)
    step = ((pl.col("x_m").diff() ** 2 + pl.col("y_m").diff() ** 2).sqrt()).over(TRACK_KEY)
    lf = lf.with_columns(
        (pl.col("t_s") - snap_t.over(TRACK_KEY)).alias("t_snap_s"),
        central_diff("x_m").alias("vx"),
        central_diff("y_m").alias("vy"),
        ((dt_prev - g["frame_dt_s"]).abs() > 1e-6).fill_null(False).alias("flag_gap"),
        (step / dt_prev > g["max_speed_mps"]).fill_null(False).alias("flag_jump"),
    )
    lf = lf.with_columns((pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt().alias("v_xy_mps"))
    lf = lf.with_columns(
        central_diff("v_xy_mps").alias("a_t_mps2"),
        (pl.col("s_mps") > g["max_speed_mps"]).fill_null(False).alias("flag_speed"),
        (pl.col("a_mps2").abs() > g["max_accel_mps2"]).fill_null(False).alias("flag_accel"),
        ((pl.col("s_mps") - pl.col("v_xy_mps")).abs() > g["max_speed_mismatch_mps"])
        .fill_null(False)
        .alias("flag_mismatch"),
    )
    glitch = pl.any_horizontal(GLITCH_FLAGS)
    return (
        lf.with_columns(glitch.alias("glitch"), glitch.any().over(TRACK_KEY).alias("glitch_track"))
        .select(
            pl.col("nfl_id").alias("player_id"),
            "season",
            "season_type",
            "week",
            "game_id",
            "play_id",
            "frame_id",
            "t_s",
            "t_snap_s",
            "x_m",
            "y_m",
            "s_mps",
            "a_mps2",
            "v_xy_mps",
            "a_t_mps2",
            "dir",
            "event",
            *FLAGS,
            "glitch",
            "glitch_track",
        )  # fmt: skip
        .collect()
    )


def summarize(frames: pl.DataFrame) -> pl.DataFrame:
    """Per season type: prospects, tracks, frames, flag counts and glitch tracks."""
    tracks = pl.struct("game_id", "play_id", "player_id")
    return (
        frames.group_by("season", "season_type")
        .agg(
            pl.col("player_id").n_unique().alias("prospects"),
            tracks.n_unique().alias("tracks"),
            pl.len().alias("frames"),
            pl.col("t_snap_s").is_null().sum().alias("frames_no_snap"),
            *(pl.col(f).sum() for f in FLAGS),
            tracks.filter(pl.col("glitch_track")).n_unique().alias("glitch_tracks"),
        )
        .sort("season", "season_type")
    )


def main() -> None:
    from eeg import config

    cfg, games = config.load(), read_games()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for season in SEASONS:
        out = build_game_frames(read_raw(season), games, cfg)
        path = OUT_DIR / f"game_tracking_{season}.parquet"
        out.write_parquet(path)
        print(f"{out.height} frames -> {path.relative_to(PROJECT_ROOT)}")
        summaries.append(summarize(out))
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(pl.concat(summaries))


if __name__ == "__main__":
    main()
