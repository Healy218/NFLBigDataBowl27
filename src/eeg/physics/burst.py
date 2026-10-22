"""Lab burst: which Combine measure of the start is reliable, and which one shows up in games.

    uv run python -m eeg.physics.burst

Candidates per prospect, from the 40:
- tracked, per well-fit rep (`sprint.max_fit_rmse_mps`), then the geometric mean over reps: τ and a0 = v_max/τ
  from the mono-exponential fit, the sustained peak of a_s (`lab_a_peak`, the field burst estimator), and the
  tracked 10-yard time t10_trk = crossing of the 10-yard line − fitted start t0;
- official: the 10-yard split (`ten_split_s`, electronic, best attempt), expressed physically as the mean
  acceleration over the first 10 yd from rest, a10 = 2 d10 / t10² (m/s²; a monotone transform, same ranking),
  and the idealized constant-force power over them, p10 = m a10 · d10 / t10 (W).

Checks (logs, within position group; groups only read results):
- rep-to-rep reliability of one tracked rep (first two well-fit reps) and of their mean (Spearman–Brown);
  the official split is a single value, so it has no rep reliability;
- validity against games: r with the prospect's field burst (`burst_a_top3`) and peak acceleration
  (`a_peak_top3`), geometric mean over qualified seasons, and with DL get-off (`get_off`, qualified seasons);
- does the start add beyond top speed: standardized betas of the split with the rest of the 40
  (fly30_off = forty − split) or with v_max, and of τ with v_max (the clustering pair).

Writes data/processed/lab_burst.parquet (not committed) and reports/lab_burst.md (aggregates only).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.ingest import units as u
from eeg.physics.sprint import TEN_YD_LINE_YD, crossing_time

D10_M = 10 * u.yd_to_m
TRACKED = ["v_max", "tau", "lab_a0", "lab_a_peak", "t10_trk"]


def tracked_t10(drills: pl.DataFrame, fits: pl.DataFrame) -> pl.DataFrame:
    """Per rep: time from the fitted start t0 to the 10-yard line crossing."""
    rows = []
    forty = drills.filter(pl.col("drill") == "forty").join(fits.select("rep_id"), on="rep_id").sort("rep_id", "t_s")
    for (rep_id,), r in forty.group_by(["rep_id"], maintain_order=True):
        rows.append((rep_id, crossing_time(r["t_s"].to_numpy(), r["x_yd"].to_numpy(), TEN_YD_LINE_YD)))
    t20 = pl.DataFrame(rows, schema=["rep_id", "t_ten_line"], orient="row")
    return (
        fits.select("rep_id", "player_id", "v_max", "tau", "t0")
        .join(t20, on="rep_id")
        .with_columns((pl.col("t_ten_line") - pl.col("t0")).fill_nan(None).alias("t10_trk"))
    )


def official_start(prospects: pl.DataFrame) -> pl.DataFrame:
    """Official 10-yard split as mean acceleration from rest (a10) and idealized constant-force power (p10)."""
    t = pl.col("ten_split_s")
    return prospects.select(
        "player_id",
        "ten_split_s",
        (pl.col("forty_s") - t).alias("fly30_off"),
        (2 * D10_M / t**2).alias("a10"),
        (pl.col("mass_kg") * 2 * D10_M / t**2 * D10_M / t).alias("p10_w"),
    ).filter(t.is_not_null())


def per_prospect(reps: pl.DataFrame, cols=TRACKED) -> pl.DataFrame:
    """Geometric mean over a prospect's well-fit reps, plus the rep count."""
    return reps.group_by("player_id").agg(
        pl.len().alias("n_reps"), *(pl.col(c).log().mean().exp().alias(f"{c}_mean") for c in cols)
    )


def _within_logs(df: pl.DataFrame, cols) -> pl.DataFrame:
    d = df.select("group", *(pl.col(c).log().alias(c) for c in cols)).drop_nulls()
    return d.with_columns((pl.col(c) - pl.col(c).mean().over("group")) for c in cols)


def _r(d: pl.DataFrame, a: str, b: str) -> float:
    return float(np.corrcoef(d[a].to_numpy(), d[b].to_numpy())[0, 1]) if d.height > 2 else float("nan")


def rep_reliability(reps: pl.DataFrame, groups: pl.DataFrame, cols=TRACKED) -> pl.DataFrame:
    """Within-group r of log value between the first two well-fit reps (one rep), and the mean of two."""
    two = reps.sort("player_id", "rep_id").group_by("player_id", maintain_order=True).head(2)
    two = two.filter(pl.len().over("player_id") == 2).with_columns(pl.int_range(2).over("player_id").alias("k"))
    pairs = two.filter(pl.col("k") == 0).join(two.filter(pl.col("k") == 1), on="player_id", suffix="_b")
    pairs = pairs.join(groups, on="player_id")
    rows = []
    for c in cols:
        r = _r(_within_logs(pairs, [c, f"{c}_b"]), c, f"{c}_b")
        rows.append(
            dict(measure=c, n=pairs.drop_nulls([c, f"{c}_b"]).height, rel_one_rep=r, rel_mean_of_2=2 * r / (1 + r))
        )
    return pl.DataFrame(rows)


def validity(lab: pl.DataFrame, targets: dict[str, tuple[pl.DataFrame, str | None]], cols) -> pl.DataFrame:
    """Within-group r of each log lab measure with each log target. targets: name -> (per-prospect frame with
    player_id and the target column of the same name, restrict to this group or None)."""
    rows = []
    for c in cols:
        row = dict(measure=c)
        for name, (frame, grp) in targets.items():
            d = lab.join(frame, on="player_id")
            if grp is not None:
                d = d.filter(pl.col("group") == grp)
            w = _within_logs(d, [c, name])
            row |= {f"r_{name}": _r(w, c, name), f"n_{name}": w.height}
        rows.append(row)
    return pl.DataFrame(rows)


def betas(df: pl.DataFrame, y: str, xs, grp: str | None = None) -> dict:
    """Standardized betas of log xs on log y, within group (centred on group means), and R²."""
    if grp is not None:
        df = df.filter(pl.col("group") == grp)
    d = _within_logs(df, [y, *xs])
    x = np.column_stack([d[c].to_numpy() for c in xs])
    yv = d[y].to_numpy()
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yv)), x]), yv, rcond=None)
    pred = np.column_stack([np.ones(len(yv)), x]) @ coef
    r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return dict(
        target=y,
        n=d.height,
        **{f"beta_{c}": b * x[:, i].std() / yv.std() for i, (c, b) in enumerate(zip(xs, coef[1:]))},
        r2=r2,
    )


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT
    from eeg.ngs.channels import describe, drag_coef, lab_channels

    cfg = config.load()
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    cs = pl.read_parquet(interim / "combine_si.parquet")
    groups = cs.select("player_id", pl.col("pos_group").alias("group"))
    drills = pl.read_parquet(interim / "timed_drills.parquet")
    fits = pl.read_parquet(interim / "forty_fits.parquet").filter(
        pl.col("t40_model").is_not_nan() & (pl.col("rmse") <= cfg["sprint"]["max_fit_rmse_mps"])
    )
    lab_reps = lab_channels(
        drills, fits, drag_coef(cs.select("player_id", "ht_m", "mass_kg"), cfg["sprint"]), cfg["games"], True
    )
    reps = tracked_t10(drills, fits).join(lab_reps.select("rep_id", "lab_a0", "lab_a_peak"), on="rep_id")

    best = pl.read_parquet(processed / "lab_channels.parquet").select(
        "player_id", pl.col("lab_a_peak").alias("lab_a_peak_best"), pl.col("lab_a0").alias("lab_a0_best")
    )
    lab = (
        groups.join(official_start(cs), on="player_id", how="left")
        .join(per_prospect(reps), on="player_id", how="left")
        .join(best, on="player_id", how="left")
        .filter(pl.col("ten_split_s").is_not_null() | pl.col("n_reps").is_not_null())
        .sort("player_id")
    )
    lab.drop("group").write_parquet(processed / "lab_burst.parquet")

    fs = pl.read_parquet(processed / "field_seasons.parquet").filter("qualified")
    field = fs.group_by("player_id").agg(
        pl.col("burst_a_top3").log().mean().exp().alias("field_burst"),
        pl.col("a_peak_top3").log().mean().exp().alias("field_a_peak"),
    )
    prod = pl.read_parquet(processed / "production_seasons.parquet").filter("qualified")
    get_off = prod.drop_nulls("get_off").group_by("player_id").agg(pl.col("get_off").log().mean().exp())
    cands = [
        "lab_a_peak_best",
        "lab_a_peak_mean",
        "lab_a0_best",
        "lab_a0_mean",
        "tau_mean",
        "t10_trk_mean",
        "ten_split_s",
        "a10",
        "p10_w",
    ]
    valid = validity(
        lab.join(cs.select("player_id", "forty_s"), on="player_id"),
        {
            "field_burst": (field.select("player_id", "field_burst"), None),
            "field_a_peak": (field.select("player_id", "field_a_peak"), None),
            "get_off": (get_off, "DL"),
        },
        [*cands, "forty_s"],
    )
    both = (
        lab.join(field, on="player_id")
        .join(get_off, on="player_id", how="left")
        .join(cs.select("player_id", "mass_kg"), on="player_id")
    )
    beta = pl.DataFrame(
        [
            betas(both, "field_burst", ["ten_split_s", "fly30_off"]),
            betas(both, "field_a_peak", ["ten_split_s", "fly30_off"]),
            betas(both, "get_off", ["ten_split_s", "fly30_off"], "DL"),
            betas(both, "get_off", ["ten_split_s", "fly30_off", "mass_kg"], "DL"),
            betas(both, "field_burst", ["tau_mean", "v_max_mean"]),
            betas(both, "get_off", ["tau_mean", "v_max_mean"], "DL"),
            betas(both, "get_off", ["ten_split_s", "v_max_mean"], "DL"),
        ],
        infer_schema_length=None,
    )

    report = f"""# Lab burst: reliable, and visible in games?

Generated by `uv run python -m eeg.physics.burst`. Per-prospect values in `data/processed/lab_burst.parquet` (not
committed). Aggregates only; all correlations on logs, within position group.

- Prospects: {lab.height}; with an official 10-yard split {lab["ten_split_s"].is_not_null().sum()}; with well-fit tracked 40 reps
  {lab["n_reps"].is_not_null().sum()} (2 reps: {(lab["n_reps"] == 2).sum()}).
- a10 = 2 d10 / t10² is the mean acceleration over the first 10 yd from rest (a transform of the split);
  p10 = m a10 d10 / t10 is the idealized constant-force power over them (W).

{_md(describe(lab, ["ten_split_s", "a10", "p10_w", "tau_mean", "lab_a_peak_mean", "t10_trk_mean"]), 2)}

## Reliability of the tracked start

First two well-fit reps of each prospect. The official split is a single best-attempt value (no reps).

{_md(rep_reliability(reps, groups), 2)}

## Which start measure shows up in games

Within-group r with field burst (`burst_a_top3`) and field peak acceleration (`a_peak_top3`), geometric mean over
qualified seasons, and with DL get-off (lower is faster, so a good start measure correlates negatively if it
grows with explosiveness, positively if it is a time). `forty_s` for comparison.

{_md(valid, 2)}

## Does the start add beyond top speed?

Standardized betas fitted together within group: the official split with the rest of the 40 (fly30_off = forty −
split) or with tracked v_max, and the profile pair (τ with v_max; the clustering uses both):

{_md(beta, 2)}
"""
    (reports / "lab_burst.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
