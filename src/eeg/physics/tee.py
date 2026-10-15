"""Total Explosive Energy (TEE): per-drill Joules, J/kg, and a cohort z-score composite.

    uv run python -m eeg.physics.tee

Inputs are data/interim/combine_si.parquet plus the rep tables written by
``eeg.physics.sprint`` (forty_fits.parquet) and ``eeg.physics.cod`` (cod_reps.parquet).
None of the swept parameters touch the rep fits or segmentation, so the sensitivity
sweep can rebuild TEE from the same rep tables with an overridden config.

Channels (TEE_Absolute, J):  E_vert, E_broad, E_40, W_cod_shuttle, W_cod_cone, W_bench
TEE_Relative (J/kg):         <channel>_per_kg, using Combine weigh-in mass
Composites:                  mean of per-channel z-scores within combine_year, for
                             prospects with >= pipeline.composite_min_drills channels
Sayers peak power (W) is a separate channel and never enters the composites.
Writes data/processed/tee.parquet.
"""

from __future__ import annotations

import polars as pl

from eeg.physics import cod, sprint
from eeg.physics.bench import work_bench
from eeg.physics.broad import energy_broad
from eeg.physics.vertical import energy_vertical, sayers_peak_power

CHANNELS = ("E_vert", "E_broad", "E_40", "W_cod_shuttle", "W_cod_cone", "W_bench")
REL_CHANNELS = tuple(f"{c}_per_kg" for c in CHANNELS)


def lab_channels(prospects: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Vertical, broad, bench Joules and Sayers power from official results."""
    v = cfg["vertical"]
    m = pl.col("mass_kg")
    return prospects.select(
        "player_id",
        energy_vertical(m, pl.col("vertical_m")).alias("E_vert"),
        energy_broad(m, pl.col("broad_m"), cfg["broad"]["theta_deg"]).alias("E_broad"),
        work_bench(pl.col("bench_reps"), pl.col("arm_m"), cfg["bench"]["rom_arm_fraction"]).alias("W_bench"),
        sayers_peak_power(m, pl.col("vertical_m"), v["sayers_a"], v["sayers_b"], v["sayers_c"]).alias("P_sayers"),
        pl.col("pos_group").is_in(v["sayers_out_of_sample_groups"]).alias("sayers_extrapolated"),
    )


def tracked_channels(prospects: pl.DataFrame, forty_reps: pl.DataFrame, cod_reps: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """E_40 and W_cod per drill from the tracking rep tables (one row per prospect with any)."""
    s = cfg["sprint"]
    e40 = sprint.forty_energy(sprint.best_reps(forty_reps, s["max_fit_rmse_mps"]), prospects, s).select(
        "player_id", "E_40"
    )
    w = (
        cod.cod_energy(cod.best_reps(cod_reps), prospects)
        .select("player_id", pl.format("W_cod_{}", "drill").alias("channel"), "W_cod")
        .pivot(on="channel", index="player_id", values="W_cod")
    )
    for c in ("W_cod_shuttle", "W_cod_cone"):
        if c not in w.columns:
            w = w.with_columns(pl.lit(None, pl.Float64).alias(c))
    return e40.join(w.select("player_id", "W_cod_shuttle", "W_cod_cone"), on="player_id", how="full", coalesce=True)


def cohort_z(col: str, by: str = "combine_year") -> pl.Expr:
    """z-score within cohort over non-null values (null stays null)."""
    return ((pl.col(col) - pl.col(col).mean().over(by)) / pl.col(col).std().over(by)).alias(f"z_{col}")


def composite(cols: list[str], min_n: int, name: str) -> pl.Expr:
    """Mean of the given z columns where at least min_n are present, else null."""
    n = pl.sum_horizontal(pl.col(c).is_not_null() for c in cols)
    return pl.when(n >= min_n).then(pl.mean_horizontal(cols)).otherwise(None).alias(name)


def build_tee(prospects: pl.DataFrame, forty_reps: pl.DataFrame, cod_reps: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """One row per prospect: absolute and per-kg channels, Sayers, flags and composites.

    ``cfg`` is the full ``config.load()`` (or ``config.values(with_override(...))``).
    W_cod_<drill>_unofficial marks a tracked W_cod for a prospect with no official time.
    """
    base = prospects.select("player_id", "combine_year", "pos_group", "mass_kg", "shuttle_s", "cone_s")
    df = (
        base.join(lab_channels(prospects, cfg), on="player_id", how="left")
        .join(tracked_channels(prospects, forty_reps, cod_reps, cfg), on="player_id", how="left")
        .with_columns(
            (pl.col("W_cod_shuttle").is_not_null() & pl.col("shuttle_s").is_null()).alias("W_cod_shuttle_unofficial"),
            (pl.col("W_cod_cone").is_not_null() & pl.col("cone_s").is_null()).alias("W_cod_cone_unofficial"),
            *[(pl.col(c) / pl.col("mass_kg")).alias(f"{c}_per_kg") for c in CHANNELS],
            *[pl.col(c).is_null().alias(f"{c}_missing") for c in CHANNELS],
            pl.sum_horizontal(pl.col(c).is_not_null() for c in CHANNELS).alias("n_drills"),
        )
        .drop("shuttle_s", "cone_s")
    )
    by = cfg["pipeline"]["zscore_within"]
    min_n = cfg["pipeline"]["composite_min_drills"]
    df = df.with_columns(*[cohort_z(c, by) for c in (*CHANNELS, *REL_CHANNELS)])
    return df.with_columns(
        composite([f"z_{c}" for c in CHANNELS], min_n, "TEE_z_abs"),
        composite([f"z_{c}" for c in REL_CHANNELS], min_n, "TEE_z_rel"),
    ).sort("player_id")


def main() -> None:
    from eeg import config
    from eeg.config import PROJECT_ROOT

    interim, processed = PROJECT_ROOT / "data" / "interim", PROJECT_ROOT / "data" / "processed"
    tee = build_tee(
        pl.read_parquet(interim / "combine_si.parquet"),
        pl.read_parquet(interim / "forty_fits.parquet"),
        pl.read_parquet(interim / "cod_reps.parquet"),
        config.load(),
    )
    processed.mkdir(parents=True, exist_ok=True)
    tee.write_parquet(processed / "tee.parquet")
    print(f"{tee.height} prospects -> data/processed/tee.parquet; {tee['TEE_z_abs'].is_not_null().sum()} with a composite")
    print(tee.select(pl.col(c).is_not_null().sum() for c in (*CHANNELS, "P_sayers")))
    print(
        tee.group_by("pos_group")
        .agg(pl.len().alias("n"), *[pl.col(c).median() for c in (*CHANNELS, "TEE_z_abs", "TEE_z_rel")])
        .sort("pos_group")
    )


if __name__ == "__main__":
    main()
