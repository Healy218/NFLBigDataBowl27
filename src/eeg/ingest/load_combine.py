"""Combine results -> one SI-unit row per in-scope prospect.

    uv run python -m eeg.ingest.load_combine [--refresh]

Reads nflverse Combine data (cached to data/raw/combine/), keeps the 5 target
position groups, converts to SI and writes data/interim/combine_si.parquet.

Notes on the source (nflverse ``load_combine``, 2000-present):
- No arm length column. ``arm_in`` is carried as null until another source
  (e.g. the Kaggle competition data) fills it; bench ROM falls back to the
  height imputation in that case.
- ``pfr_id`` / ``cfb_id`` are sometimes shared by different people (e.g. two
  2005 Texas "Derrick Johnson"s). Those rows get ``*_id_conflict = True`` and
  must not be used for ID joins without a manual check.
- 2021 had no in-person Combine workouts (COVID); its results are pro-day
  numbers, flagged by ``test_setting = "pro_day"``.
- Position labels drift by year (DE -> EDGE from ~2018; generic OL/DL/DB in
  2019-21; OG -> G, S -> SAF later). POS_GROUPS maps all of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from eeg.config import PROJECT_ROOT
from eeg.ingest import units as u

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "combine" / "combine_nflverse.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "combine_si.parquet"

# Raw nflverse position label -> study group. Labels not listed are out of
# scope (QB, RB, FB, LB/ILB/OLB, K, P, LS, and the one-off CB/WR).
# EDGE -> DL: DE disappears as a label once EDGE appears, so excluding EDGE
# would drop every defensive end after ~2019.
POS_GROUPS: dict[str, str] = {
    "WR": "WR",
    "TE": "TE",
    "CB": "DB",
    "S": "DB",
    "SAF": "DB",
    "DB": "DB",
    "OT": "OL",
    "OG": "OL",
    "G": "OL",
    "C": "OL",
    "OL": "OL",
    "DT": "DL",
    "DE": "DL",
    "DL": "DL",
    "EDGE": "DL",
}

PRO_DAY_SEASONS = {2021}

# raw column -> (output name in source units, SI name, SI factor)
MEASURES: dict[str, tuple[str, str, float]] = {
    "wt": ("wt_lb", "mass_kg", u.lb_to_kg),
    "vertical": ("vertical_in", "vertical_m", u.in_to_m),
    "broad_jump": ("broad_in", "broad_m", u.in_to_m),
}
TIMES = {"forty": "forty_s", "cone": "cone_s", "shuttle": "shuttle_s"}
DRILL_FLAGS = ["forty", "vertical", "broad", "bench", "cone", "shuttle"]


def fetch_raw(refresh: bool = False, path: Path = RAW_PATH) -> pl.DataFrame:
    """nflverse Combine table, cached locally as parquet."""
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    import nflreadpy as nfl

    df = nfl.load_combine()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def parse_height_in(col: str = "ht") -> pl.Expr:
    """``"6-2"`` -> 74.0 inches; anything else -> null."""
    parts = pl.col(col).str.strip_chars().str.extract_groups(r"^(\d+)-(\d{1,2})$")
    feet = parts.struct.field("1").cast(pl.Float64)
    inches = parts.struct.field("2").cast(pl.Float64)
    return (feet * 12 + inches).alias("ht_in")


def _slug(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", "-")
        .str.strip_chars("-")
    )


def build_combine_si(raw: pl.DataFrame) -> pl.DataFrame:
    """Scope to the 5 groups, convert to SI, add missingness and ID-quality flags."""
    # ID conflicts are judged on the full table: a WR can share a pfr_id with an ILB.
    conflicts = [
        (pl.col(c).is_not_null() & pl.col(c).is_duplicated()).alias(f"{c.split('_')[0]}_id_conflict")
        for c in ("pfr_id", "cfb_id")
    ]
    df = raw.with_columns(conflicts).filter(pl.col("pos").is_in(list(POS_GROUPS)))

    df = df.with_columns(
        pl.concat_str(
            [pl.col("season").cast(pl.String), _slug("player_name"), _slug("school"), pl.col("pos")],
            separator="_",
        ).alias("player_id"),
        pl.col("season").alias("combine_year"),
        pl.col("pos").replace_strict(POS_GROUPS).alias("pos_group"),
        pl.when(pl.col("season").is_in(list(PRO_DAY_SEASONS)))
        .then(pl.lit("pro_day"))
        .otherwise(pl.lit("combine"))
        .alias("test_setting"),
        parse_height_in("ht"),
        pl.lit(None, dtype=pl.Float64).alias("arm_in"),
        pl.col("bench").alias("bench_reps"),
        *[pl.col(src).alias(name) for src, (name, _, _) in MEASURES.items()],
        *[pl.col(src).alias(name) for src, name in TIMES.items()],
    ).with_columns(
        (pl.col("ht_in") * u.in_to_m).alias("ht_m"),
        (pl.col("arm_in") * u.in_to_m).alias("arm_m"),
        *[(pl.col(name) * f).alias(si) for name, si, f in MEASURES.values()],
    )

    present = {
        "ht": "ht_in",
        "wt": "wt_lb",
        "arm": "arm_in",
        "forty": "forty_s",
        "vertical": "vertical_in",
        "broad": "broad_in",
        "bench": "bench_reps",
        "cone": "cone_s",
        "shuttle": "shuttle_s",
    }
    df = df.with_columns(
        *[pl.col(c).is_null().alias(f"{k}_missing") for k, c in present.items()],
        pl.sum_horizontal([pl.col(present[d]).is_not_null() for d in DRILL_FLAGS])
        .cast(pl.Int8)
        .alias("n_drills_raw"),
    )

    cols = [
        "player_id", "player_name", "combine_year", "test_setting", "school",
        "pos", "pos_group", "pfr_id", "cfb_id", "pfr_id_conflict", "cfb_id_conflict",
        "draft_year", "draft_round", "draft_ovr",
        "ht_in", "wt_lb", "arm_in", "vertical_in", "broad_in", "bench_reps",
        "forty_s", "cone_s", "shuttle_s",
        "ht_m", "mass_kg", "arm_m", "vertical_m", "broad_m",
        *[f"{k}_missing" for k in present], "n_drills_raw",
    ]  # fmt: skip
    out = df.select(cols).sort("combine_year", "pos_group", "player_name")
    if out["player_id"].is_duplicated().any():
        dups = out.filter(pl.col("player_id").is_duplicated())["player_id"].to_list()
        raise ValueError(f"player_id not unique: {dups[:5]}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download from nflverse")
    args = ap.parse_args(argv)

    raw = fetch_raw(refresh=args.refresh)
    out = build_combine_si(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH)

    print(f"raw rows: {raw.height}  ->  in-scope prospects: {out.height}")
    print(out.group_by("pos_group").len().sort("pos_group"))
    print(f"wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
