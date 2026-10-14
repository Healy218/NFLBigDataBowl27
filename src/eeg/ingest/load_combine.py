"""Kaggle Combine results -> one SI-unit row per prospect.

    uv run python -m eeg.ingest.load_combine

Reads combine_results.csv (+ name and NFL position from players.csv) and writes
data/interim/combine_si.parquet. See config/data_dictionary.md for the source
schema; heights and arm/hand/wingspan are already numeric inches, and
combine_position is already one of the 5 study groups.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from eeg.config import PROJECT_ROOT
from eeg.ingest import units as u

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle" / "nfl-big-data-bowl-2027"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "combine_si.parquet"

POS_GROUPS = {"WR", "TE", "DB", "OL", "DL"}

# source column -> output name (source units kept alongside SI columns)
RENAME = {
    "nfl_id": "player_id",
    "draft_year": "combine_year",
    "combine_position": "pos_group",
    "combine_height": "ht_in",
    "combine_weight": "wt_lb",
    "arm_length": "arm_in",
    "hand_size": "hand_in",
    "wing_span": "wingspan_in",
    "ten_yd_split": "ten_split_s",
    "forty": "forty_s",
    "vertical": "vertical_in",
    "broad_jump": "broad_in",
    "bench_reps": "bench_reps",
    "three_cone": "cone_s",
    "short_shuttle": "shuttle_s",
    "ngs_athleticism_score": "ngs_athleticism_score",
    "ngs_college_production_score": "ngs_college_production_score",
    "ngs_final_score": "ngs_final_score",
}

# inch/lb column -> (SI column, factor)
TO_SI = {
    "ht_in": ("ht_m", u.in_to_m),
    "wt_lb": ("mass_kg", u.lb_to_kg),
    "arm_in": ("arm_m", u.in_to_m),
    "wingspan_in": ("wingspan_m", u.in_to_m),
    "vertical_in": ("vertical_m", u.in_to_m),
    "broad_in": ("broad_m", u.in_to_m),
}

# The 6 drill channels that feed TEE (ten-yd split is part of the 40, not a channel)
DRILLS = {
    "forty": "forty_s",
    "vertical": "vertical_in",
    "broad": "broad_in",
    "bench": "bench_reps",
    "cone": "cone_s",
    "shuttle": "shuttle_s",
}


def read_raw(raw_dir: Path = RAW_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(combine_results, players) as published."""
    read = lambda name: pl.read_csv(raw_dir / name, null_values="NA")  # noqa: E731
    return read("combine_results.csv"), read("players.csv")


def build_combine_si(results: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """One row per prospect: source units, SI units, missingness flags."""
    if not results["nfl_id"].is_unique().all():
        raise ValueError("combine_results.nfl_id is not unique")
    bad = set(results["combine_position"].unique()) - POS_GROUPS
    if bad:
        raise ValueError(f"unexpected combine_position values: {sorted(bad)}")

    df = (
        results.join(
            players.select("nfl_id", pl.col("display_name").alias("name"), pl.col("nfl_position").alias("pos")),
            on="nfl_id",
            how="left",
            validate="1:1",
        )
        .rename(RENAME)
        .with_columns(*[(pl.col(src) * f).alias(si) for src, (si, f) in TO_SI.items()])
        .with_columns(
            *[pl.col(c).is_null().alias(f"{k}_missing") for k, c in DRILLS.items()],
            pl.sum_horizontal([pl.col(c).is_not_null() for c in DRILLS.values()]).cast(pl.Int8).alias("n_drills_raw"),
        )
    )

    cols = [
        "player_id", "name", "combine_year", "pos_group", "pos",
        *[c for c in RENAME.values() if c not in {"player_id", "combine_year", "pos_group"}],
        *[si for si, _ in TO_SI.values()],
        *[f"{k}_missing" for k in DRILLS], "n_drills_raw",
    ]  # fmt: skip
    return df.select(cols).sort("combine_year", "pos_group", "player_id")


def main() -> None:
    out = build_combine_si(*read_raw())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH)
    print(f"{out.height} prospects -> {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(out.group_by("combine_year", "pos_group").len().pivot(on="pos_group", index="combine_year", values="len").sort("combine_year"))
    print("n_drills_raw:", dict(out["n_drills_raw"].value_counts().sort("n_drills_raw").iter_rows()))


if __name__ == "__main__":
    main()
