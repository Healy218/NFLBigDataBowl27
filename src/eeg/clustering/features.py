"""Phase 3 feature matrix: drill output relative to body mass, no positions.

    uv run python -m eeg.clustering.features

Absolute Joules and J/kg both mostly measure body mass (and so position group).
Each feature is therefore "output for your size":

    resid = log y - (a_year + b1 x + b2 x² ...)    x = log m - mean, all prospects
    f_<y> = resid z-scored within combine_year

and log mass is one explicit feature, f_log_mass (also z-scored within year). The
trend's degree is clustering.mass_trend_degree (1 = power law). Its coefficients
are fit on year-demeaned values (a within-year fit), so cohort drift in the level
of a channel never leaks into them. Positions are never used in a fit.

Features (config clustering.features) define the clusters; prospects missing any
are dropped. Descriptors (clustering.descriptors) get the same treatment, fit on
the clustered prospects that have them, and only describe archetypes afterwards.

Writes data/processed/features.parquet (player_id + f_* only),
data/processed/features_context.parquet (year, pos_group, mass, raw values,
descriptor z-scores d_*) and reports/feature_fits.csv (slope and fit stats).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.physics import sprint

KEY_COLS = ("player_id", "combine_year", "pos_group", "mass_kg")


def mass_residual(df: pl.DataFrame, col: str, by: str = "combine_year", degree: int = 1) -> tuple[pl.Series, dict]:
    """log(col) minus a polynomial trend in log mass, fit on within-`by` demeaned values.

    x = log m centred on the fitted rows' mean, so ``slope`` is d log y / d log m at
    the mean mass and ``curvature`` the x² coefficient (null for degree 1). Rows with
    a null or non-positive value get a null residual. Also returns n, the r of log
    col with log mass, and the residual SD.
    """
    ok = pl.col(col).is_not_null() & (pl.col(col) > 0)
    d = (
        df.with_row_index("_i")
        .filter(ok)
        .select("_i", by, pl.col(col).log().alias("ly"), pl.col("mass_kg").log().alias("lm"))
    )
    r = d.select(pl.corr("ly", "lm")).item()
    d = d.with_columns((pl.col("lm") - pl.col("lm").mean()).alias("x"))
    powers = [f"x{p}" for p in range(1, degree + 1)]
    d = d.with_columns(pl.col("x").pow(p).alias(f"x{p}") for p in range(1, degree + 1))
    d = d.with_columns((pl.col(c) - pl.col(c).mean().over(by)).alias(c) for c in ("ly", *powers))
    X, y = d.select(powers).to_numpy(), d["ly"].to_numpy()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    out = pl.Series(col, [None] * df.height, dtype=pl.Float64).scatter(d["_i"].to_numpy(), resid)
    fit = dict(
        column=col,
        slope=float(coef[0]),
        curvature=float(coef[1]) if degree >= 2 else None,
        n=d.height,
        r_log_mass=r,
        resid_sd=float(resid.std(ddof=1)),
    )
    return out, fit


def zscore_within(col: str, by: str = "combine_year") -> pl.Expr:
    return (pl.col(col) - pl.col(col).mean().over(by)) / pl.col(col).std().over(by)


def sprint_params(forty_reps: pl.DataFrame, max_rmse: float) -> pl.DataFrame:
    """v_max and tau from each player's best 40 rep (the rep behind E_40)."""
    return sprint.best_reps(forty_reps, max_rmse).select("player_id", "v_max", "tau")


def build_features(
    tee: pl.DataFrame, forty_reps: pl.DataFrame, cfg: dict
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(features, context, fits). ``cfg`` is the full ``config.load()``.

    features: player_id and f_* for prospects with every feature present.
    context:  the same prospects with key columns, raw values and descriptor z-scores d_*.
    fits:     one row per feature / descriptor with slope, n, r_log_mass, resid_sd.
    """
    by = cfg["pipeline"]["zscore_within"]
    degree = cfg["clustering"]["mass_trend_degree"]
    feats, descs = cfg["clustering"]["features"], cfg["clustering"]["descriptors"]
    df = tee.join(sprint_params(forty_reps, cfg["sprint"]["max_fit_rmse_mps"]), on="player_id", how="left")
    df = df.filter(pl.all_horizontal(pl.col(c).is_not_null() & (pl.col(c) > 0) for c in feats))

    fits = []
    for c in (*feats, *descs):
        resid, fit = mass_residual(df, c, by, degree)
        df = df.with_columns(resid.alias(f"resid_{c}"))
        fits.append(fit | dict(role="feature" if c in feats else "descriptor"))
    df = (
        df.with_columns(pl.col("mass_kg").log().alias("_lm"))
        .with_columns(
            zscore_within("_lm", by).alias("f_log_mass"),
            *[zscore_within(f"resid_{c}", by).alias(f"f_{c}") for c in feats],
            *[zscore_within(f"resid_{c}", by).alias(f"d_{c}") for c in descs],
        )
        .drop("_lm")
    )

    f_cols = ["f_log_mass", *[f"f_{c}" for c in feats]]
    features = df.select("player_id", *f_cols).sort("player_id")
    context = df.select(*KEY_COLS, *feats, *descs, *[f"d_{c}" for c in descs]).sort("player_id")
    return (
        features,
        context,
        pl.DataFrame(fits, schema_overrides={"curvature": pl.Float64}).select(
            "role", "column", "slope", "curvature", "n", "r_log_mass", "resid_sd"
        ),
    )


def main() -> None:
    from eeg import config
    from eeg.config import PROJECT_ROOT

    processed = PROJECT_ROOT / "data" / "processed"
    features, context, fits = build_features(
        pl.read_parquet(processed / "tee.parquet"),
        pl.read_parquet(PROJECT_ROOT / "data" / "interim" / "forty_fits.parquet"),
        config.load(),
    )
    features.write_parquet(processed / "features.parquet")
    context.write_parquet(processed / "features_context.parquet")
    fits.write_csv(PROJECT_ROOT / "reports" / "feature_fits.csv", float_precision=4)
    print(f"{features.height} prospects -> data/processed/features.parquet")
    print(context.group_by("pos_group").len().sort("pos_group"))
    print(fits)
    f = features.drop("player_id")
    print("feature correlations:")
    print(f.corr().with_columns(pl.all().round(2)).insert_column(0, pl.Series("", f.columns)))


if __name__ == "__main__":
    main()
