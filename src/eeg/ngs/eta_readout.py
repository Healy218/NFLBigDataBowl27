"""Reading η (post hoc): shrunk leaderboards per position group, and η against the lab engine.

    uv run python -m eeg.ngs.eta_readout

Position groups are used here only to read results: the shrinkage target is the
group mean, and every correlation is within group (centred on the group mean).

- Shrinkage (empirical Bayes on log η_KE, per prospect): season values scatter
  around a prospect's true value with variance σ²_s (pooled from prospects with
  ≥ 2 qualified seasons: game noise plus real change between seasons); true
  values scatter around the group mean with variance τ² (method of moments:
  between-prospect variance of mean log η minus the average σ²_s / n). A
  prospect with n seasons is pulled toward the group mean by
  B = τ² / (τ² + σ²_s / n), posterior SD √(B σ²_s / n).
- Credibly above / below: the 90% interval excludes the group mean.
- Correlates: within-group r (95% CI, Fisher z) of shrunk log η_KE with the lab
  side (40 speed, profile axes, Kinetic Anomaly, mass, TEE) and NGS scores.
- Engine × efficiency: within-group terciles of `ax_explosive` against terciles
  of shrunk η_KE ("Combine warrior" = top engine, bottom η).

Writes data/processed/eta_leaderboard.parquet (named, not committed),
reports/eta_readout.md (aggregates only) and reports/figures/eta_readout.png.
Prints the named leaderboard to the terminal only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

Z90 = 1.645
TERCILE = ("low", "mid", "high")


def variance_components(seasons: pl.DataFrame, value: str, group: str = "group") -> dict:
    """σ²_s (between seasons of a prospect) and τ² (between prospects within group) of a log value."""
    d = seasons.select("player_id", group, pl.col(value).alias("x")).drop_nulls()
    d = d.with_columns(pl.col("x").mean().over("player_id").alias("m"), pl.len().over("player_id").alias("n"))
    multi = d.filter(pl.col("n") >= 2)
    df_within = multi.height - multi["player_id"].n_unique()
    sigma2 = float(((multi["x"] - multi["m"]) ** 2).sum() / df_within)
    p = d.group_by("player_id", group).agg(pl.col("m").first(), pl.col("n").first())
    p = p.with_columns((pl.col("m") - pl.col("m").mean().over(group)).alias("dev"))
    between = float((p["dev"] ** 2).sum() / (p.height - p[group].n_unique()))
    tau2 = max(between - float((sigma2 / p["n"]).mean()), 1e-6)
    return dict(sigma2=sigma2, tau2=tau2, prospects=p.height, multi_season=multi["player_id"].n_unique())


def shrink(seasons: pl.DataFrame, value: str, vc: dict, group: str = "group") -> pl.DataFrame:
    """Per prospect: mean log value over seasons, group mean, shrinkage factor B, shrunk value and 90% interval."""
    p = (
        seasons.drop_nulls(value)
        .group_by("player_id", group)
        .agg(pl.len().alias("n"), pl.col(value).mean().alias("raw"))
        .with_columns(pl.col("raw").mean().over(group).alias("group_mean"))
        .with_columns((vc["tau2"] / (vc["tau2"] + vc["sigma2"] / pl.col("n"))).alias("B"))
        .with_columns(
            (pl.col("group_mean") + pl.col("B") * (pl.col("raw") - pl.col("group_mean"))).alias("shrunk"),
            (pl.col("B") * vc["sigma2"] / pl.col("n")).sqrt().alias("post_sd"),
        )
        .with_columns(
            (pl.col("shrunk") - Z90 * pl.col("post_sd")).alias("lo90"),
            (pl.col("shrunk") + Z90 * pl.col("post_sd")).alias("hi90"),
        )
    )
    verdict = (
        pl.when(pl.col("lo90") > pl.col("group_mean"))
        .then(pl.lit("above"))
        .when(pl.col("hi90") < pl.col("group_mean"))
        .then(pl.lit("below"))
        .otherwise(pl.lit("—"))
    )
    rank = pl.col("shrunk").rank("ordinal", descending=True).over(group)
    return p.with_columns(verdict.alias("credible"), rank.alias("rank")).sort(group, "rank")


def within_r(df: pl.DataFrame, x: str, y: str, group: str = "group") -> dict:
    """Pearson r of x and y after centring both on their group means, with a 95% Fisher-z interval."""
    d = df.select(group, x, y).drop_nulls()
    d = d.with_columns((pl.col(c) - pl.col(c).mean().over(group)).alias(c) for c in (x, y))
    n = d.height
    r = float(np.corrcoef(d[x].to_numpy(), d[y].to_numpy())[0, 1])
    half = 1.96 / np.sqrt(n - 3 - (d[group].n_unique() - 1))
    lo, hi = np.tanh(np.arctanh(r) - half), np.tanh(np.arctanh(r) + half)
    return dict(variable=x, n=n, r=r, lo95=float(lo), hi95=float(hi))


def terciles(df: pl.DataFrame, col: str, group: str = "group") -> pl.Expr:
    """Within-group tercile label of col (low / mid / high)."""
    q = (pl.col(col).rank("average").over(group) - 0.5) / pl.col(col).count().over(group)
    return (
        pl.when(q < 1 / 3)
        .then(pl.lit(TERCILE[0]))
        .when(q < 2 / 3)
        .then(pl.lit(TERCILE[1]))
        .otherwise(pl.lit(TERCILE[2]))
    )


def plot_readout(board: pl.DataFrame, order, path) -> None:
    """Left: shrunk η_KE with 90% intervals per prospect, by group. Right: engine against shrunk η, within group."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6))
    x0 = 0
    for i, g in enumerate(order):
        b = board.filter(pl.col("group") == g).sort("shrunk")
        xs = np.arange(x0, x0 + b.height)
        ax0.errorbar(xs, np.exp(b["shrunk"]), yerr=[np.exp(b["shrunk"]) - np.exp(b["lo90"]),
                     np.exp(b["hi90"]) - np.exp(b["shrunk"])], fmt="o", ms=2, lw=0.6, color=f"C{i}", ecolor=f"C{i}", alpha=0.7, label=g)  # fmt: skip
        ax0.hlines(np.exp(b["group_mean"][0]), x0, x0 + b.height - 1, color="k", lw=0.8)
        x0 += b.height + 8
    ax0.set_xticks([])
    ax0.set_ylabel("shrunk η_KE (90% interval)")
    ax0.set_title("η_KE per prospect, ordered within group")
    ax0.legend(fontsize=8, ncol=5, loc="lower center")
    d = board.drop_nulls("ax_explosive").with_columns(
        (pl.col(c) - pl.col(c).mean().over("group")).alias(c) for c in ("ax_explosive", "shrunk")
    )
    for i, g in enumerate(order):
        s = d.filter(pl.col("group") == g)
        ax1.scatter(s["ax_explosive"], s["shrunk"], s=10, alpha=0.6, color=f"C{i}", label=g)
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.axvline(0, color="grey", lw=0.6)
    ax1.set_xlabel("engine (ax_explosive), centred within group")
    ax1.set_ylabel("shrunk log η_KE, centred within group")
    ax1.set_title("Engine against efficiency")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg.clustering.audit import _md
    from eeg.clustering.readout import POS_ORDER
    from eeg.config import PROJECT_ROOT

    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    combine = pl.read_parquet(interim / "combine_si.parquet").select(
        "player_id", "name", pl.col("pos_group").alias("group"), "pos", "mass_kg",
        "ngs_athleticism_score", "ngs_college_production_score",
    )  # fmt: skip
    groups = combine.select("player_id", "group")
    eta = pl.read_parquet(processed / "eta.parquet").join(groups, on="player_id")
    eta = eta.with_columns(pl.col("eta_ke").log().alias("log_eta_ke"))
    career = pl.read_parquet(processed / "eta_career.parquet")
    profiles = pl.read_parquet(processed / "profiles.parquet")
    lab = pl.read_parquet(processed / "lab_channels.parquet").select("player_id", "lab_s_peak")
    tee = pl.read_parquet(processed / "tee.parquet").select("player_id", "TEE_z_rel")

    vc = variance_components(eta, "log_eta_ke")
    board = (
        shrink(eta, "log_eta_ke", vc)
        .join(combine.drop("group"), on="player_id")
        .join(career.select("player_id", "first_season", "eta_ke_career", "eta_ke_first"), on="player_id")
        .join(lab, on="player_id")
        .join(profiles.select("player_id", "ax_explosive", "ax_speed_vs_bounce", "ax_up_vs_out", "anomaly_d2"),
              on="player_id", how="left")
        .join(tee, on="player_id", how="left")
        .with_columns(pl.col("group").cast(pl.Enum(POS_ORDER)))
        .with_columns(
            pl.col("shrunk").exp().alias("eta_ke_shrunk"),
            pl.col("lab_s_peak").log().alias("log_lab_speed"),
            pl.col("mass_kg").log().alias("log_mass"),
        )
    )  # fmt: skip
    has_axis = board.filter(pl.col("ax_explosive").is_not_null())
    has_axis = has_axis.with_columns(
        terciles(has_axis, "ax_explosive").alias("engine"), terciles(has_axis, "shrunk").alias("efficiency")
    )
    board = board.join(has_axis.select("player_id", "engine", "efficiency"), on="player_id", how="left")
    board = board.sort("group", "rank")
    board.write_parquet(processed / "eta_leaderboard.parquet")
    plot_readout(board, POS_ORDER, reports / "figures" / "eta_readout.png")

    sd_s, tau = np.sqrt(vc["sigma2"]), np.sqrt(vc["tau2"])
    b_by_n = pl.DataFrame(
        [dict(seasons=n, B=vc["tau2"] / (vc["tau2"] + vc["sigma2"] / n),
              post_sd=np.sqrt(vc["tau2"] * vc["sigma2"] / n / (vc["tau2"] + vc["sigma2"] / n))) for n in (1, 2, 3)]
    )  # fmt: skip
    summary = (
        board.group_by("group")
        .agg(
            pl.len().alias("prospects"),
            pl.col("group_mean").first().exp().alias("group_eta_ke"),
            pl.col("eta_ke_shrunk").quantile(0.1).alias("shrunk_p10"),
            pl.col("eta_ke_shrunk").quantile(0.9).alias("shrunk_p90"),
            (pl.col("credible") == "above").sum().alias("credibly_above"),
            (pl.col("credible") == "below").sum().alias("credibly_below"),
        )
        .sort("group")
    )
    covars = {
        "log_lab_speed": "40 peak speed (log; the η denominator)",
        "ax_explosive": "profile axis 1: overall engine",
        "ax_speed_vs_bounce": "profile axis 2: speed vs bounce",
        "ax_up_vs_out": "profile axis 3: vertical vs broad",
        "anomaly_d2": "Kinetic Anomaly (Mahalanobis d²)",
        "TEE_z_rel": "TEE relative (J/kg composite)",
        "log_mass": "body mass (log)",
        "ngs_athleticism_score": "NGS athleticism score",
        "ngs_college_production_score": "NGS college production score",
    }
    corr = pl.DataFrame(
        [
            within_r(board, c, "shrunk") | {"meaning": m, "r_within_pos": within_r(board, c, "shrunk", "pos")["r"]}
            for c, m in covars.items()
        ]
    )
    sub = (
        board.group_by("group", "pos")
        .agg(pl.len().alias("prospects"), pl.col("eta_ke_shrunk").median().alias("median_eta_ke"),
             pl.col("mass_kg").median().alias("median_mass_kg"))
        .filter(pl.col("prospects") >= 3)
        .sort("group", "pos")
    )  # fmt: skip
    quad = has_axis.group_by("engine", "efficiency").len().pivot(on="efficiency", index="engine", values="len")
    quad = quad.with_columns(pl.col("engine").cast(pl.Enum(TERCILE))).sort("engine").select("engine", *TERCILE)
    first_vs_career = within_r(board.with_columns(pl.col("eta_ke_first").log()), "eta_ke_first", "shrunk")

    report = f"""# Reading η (post hoc)

Generated by `uv run python -m eeg.ngs.eta_readout`. Shrunk η_KE per prospect from `data/processed/eta.parquet`;
the named leaderboard is `data/processed/eta_leaderboard.parquet` (not committed). Position groups only read
results: the shrinkage target is the group mean and every correlation is within group. Aggregates only.

## Shrinkage (log η_KE)

- Between seasons of a prospect: SD {sd_s:.3f} (pooled over {vc["multi_season"]} prospects with ≥ 2 seasons).
- Between prospects within a group (true spread): SD {tau:.3f} ({vc["prospects"]} prospects).
- So one season is a weak read: a prospect's value is pulled toward the group mean by a factor B.

{_md(b_by_n, 3)}

## Per group

η_KE: group mean, spread of shrunk values (p10–p90), and prospects whose 90% interval excludes the group mean.

{_md(summary, 3)}

By listed position (groups of ≥ 3): sub-roles inside a group differ little in median η_KE.

{_md(sub, 3)}

## What goes with efficiency (within group, shrunk log η_KE)

`r_within_pos`: the same r centred on the listed position instead (finer role control).

{_md(corr.select("variable", "meaning", "n", "r", "lo95", "hi95", "r_within_pos"), 2)}

First-season η_KE against the shrunk career value: r {first_vs_career["r"]:.2f} (n {first_vs_career["n"]}); the career
value contains the first season and half the prospects have only one, so this is not a retest (that is r 0.42,
`reports/eta.md`).

## Engine × efficiency (profiled prospects, within-group terciles)

Rows: engine (`ax_explosive`). Columns: shrunk η_KE. Independence would put ~{has_axis.height / 9:.0f} in each cell.
"Combine warriors" are high engine × low efficiency; the opposite corner gets the most from a modest engine.

{_md(quad, 0)}

Figure: `reports/figures/eta_readout.png`.
"""
    (reports / "eta_readout.md").write_text(report)
    print(report)

    print("\n--- Named leaderboard (terminal only; not written to reports) ---")
    cols = ["rank", "name", "pos", "n", "eta_ke_career", "eta_ke_shrunk", "credible", "engine"]
    with pl.Config(tbl_rows=40, tbl_cols=12, float_precision=3, tbl_hide_dataframe_shape=True):
        for g in POS_ORDER:
            b = board.filter(pl.col("group") == g).select(cols)
            print(f"\n{g}: top 5 / bottom 5 of {b.height}")
            print(pl.concat([b.head(5), b.tail(5)]))
        print("\nCombine warriors (high engine, low efficiency):")
        print(board.filter((pl.col("engine") == "high") & (pl.col("efficiency") == "low")).select("group", *cols[1:]))


if __name__ == "__main__":
    main()
