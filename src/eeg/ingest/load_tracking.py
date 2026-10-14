"""Combine tracking for the timed drills -> one frame per row, SI units.

    uv run python -m eeg.ingest.load_tracking

Keeps PLAYER frames of the 40, shuttle and 3-cone, adds a rep key and time since
the first frame of the rep, and converts to SI. Field positions stay in yards too
(x_yd), because drill geometry is defined on yard lines (40: x = 10 -> 50).
The source `dis` column is dropped: it does not match the x/y displacement
(see config/data_dictionary.md). Writes data/interim/timed_drills.parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from eeg.config import PROJECT_ROOT
from eeg.ingest import units as u
from eeg.ingest.load_combine import RAW_DIR

OUT_PATH = PROJECT_ROOT / "data" / "interim" / "timed_drills.parquet"

TIMED_DRILLS = {"FORTY_YARD_DASH": "forty", "SHORT_SHUTTLE": "shuttle", "THREE_CONE_DRILL": "cone"}
REP_KEY = ["nfl_id", "event_id", "attempt"]


def read_raw(raw_dir: Path = RAW_DIR) -> pl.LazyFrame:
    return pl.scan_csv(raw_dir / "combine_tracking.csv", null_values="NA", try_parse_dates=True)


def build_timed_drills(raw: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """PLAYER frames of the timed drills, one row per frame, sorted by rep and time."""
    lf = raw.lazy().filter(
        (pl.col("entity_type") == "PLAYER") & pl.col("drill_type").is_in(list(TIMED_DRILLS))
    )
    return (
        lf.sort([*REP_KEY, "time"])
        .with_columns(
            pl.col("drill_type").replace_strict(TIMED_DRILLS).alias("drill"),
            pl.concat_str([pl.col(c).cast(pl.String) for c in REP_KEY], separator="_").alias("rep_id"),
            ((pl.col("time") - pl.col("time").first()).dt.total_milliseconds() / 1000.0)
            .over(REP_KEY)
            .alias("t_s"),
            pl.col("x").alias("x_yd"),
            pl.col("y").alias("y_yd"),
            (pl.col("x") * u.yd_to_m).alias("x_m"),
            (pl.col("y") * u.yd_to_m).alias("y_m"),
            (pl.col("s") * u.yd_to_m).alias("s_mps"),
            (pl.col("a") * u.yd_to_m).alias("a_mps2"),
        )
        .select(
            pl.col("nfl_id").alias("player_id"), "draft_year", "drill", "rep_id", "event_id", "attempt",
            "time", "t_s", "x_yd", "y_yd", "x_m", "y_m", "s_mps", "a_mps2", "dir",
        )  # fmt: skip
        .collect()
    )


def main() -> None:
    out = build_timed_drills(read_raw())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH)
    print(f"{out.height} frames -> {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(
        out.group_by("drill")
        .agg(pl.col("player_id").n_unique().alias("players"), pl.col("rep_id").n_unique().alias("reps"), pl.len().alias("frames"))
        .sort("drill")
    )


if __name__ == "__main__":
    main()
