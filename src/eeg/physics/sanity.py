"""Sanity and sensitivity pass over TEE.

    uv run python -m eeg.physics.sanity

Outliers: a value is flagged when |z| > pipeline.outlier_sd, either within its
combine_year cohort (the z columns of tee.parquet) or within its pos_group (pooled
over years: a single group-year has 13-50 prospects, too few to ever reach 4 SD).
pos_group is used here only to find data errors, never as a feature.

Sensitivity: every swept parameter (config.sweep_params) is moved one at a time to
its low and high bound, plus all-low and all-high corners, and TEE is rebuilt from
the same rep tables. Each channel and composite is rank-correlated with the default
build, pooled and within each pos_group (pooled ranks of absolute channels are mostly
mass, so the within-group ρ is the stricter test). A case passes when every ρ is
>= pipeline.spearman_min.

Writes data/processed/tee_outliers.parquet, reports/sensitivity.csv and
reports/figures/sanity_*.png.
"""

from __future__ import annotations

import polars as pl

from eeg import config
from eeg.physics.tee import CHANNELS, REL_CHANNELS, build_tee

COMPOSITES = ("TEE_z_abs", "TEE_z_rel")
CHECKED = (*CHANNELS, *REL_CHANNELS, *COMPOSITES)
GROUPS = ("WR", "TE", "DB", "OL", "DL")


def outlier_flags(tee: pl.DataFrame, sd: float) -> pl.DataFrame:
    """Long table of (player_id, channel, scope, z) with |z| > sd."""
    cols = (*CHANNELS, *REL_CHANNELS)
    z_group = tee.select(
        "player_id",
        *[((pl.col(c) - pl.col(c).mean().over("pos_group")) / pl.col(c).std().over("pos_group")).alias(c) for c in cols],
    )
    z_cohort = tee.select("player_id", *[pl.col(f"z_{c}").alias(c) for c in cols])
    long = pl.concat(
        [
            z.unpivot(index="player_id", variable_name="channel", value_name="z").with_columns(pl.lit(scope).alias("scope"))
            for scope, z in (("combine_year", z_cohort), ("pos_group", z_group))
        ]
    )
    return (
        long.filter(pl.col("z").abs() > sd)
        .join(tee.select("player_id", "combine_year", "pos_group", "mass_kg"), on="player_id")
        .sort("player_id", "channel", "scope")
    )


def spearman(a: pl.Series, b: pl.Series) -> float:
    """Spearman ρ over rows where both are non-null.

    Values are rounded to 8 significant figures first: tied raw results (e.g. broad
    jumps recorded to the inch) give per-kg values that differ by float noise,
    and a uniform rescale would otherwise reorder those spurious near-ties.
    """
    df = pl.DataFrame({"a": a, "b": b}).drop_nulls().select(pl.all().round_sig_figs(8))
    if df.height < 3:
        return float("nan")
    return df.select(pl.corr("a", "b", method="spearman")).item()


def rank_stability(base: pl.DataFrame, alt: pl.DataFrame, cols=CHECKED) -> pl.DataFrame:
    """Per column: pooled ρ and the minimum within-pos_group ρ against the base build."""
    j = base.select("player_id", "pos_group", *cols).join(alt.select("player_id", *cols), on="player_id", suffix="_alt")
    rows = []
    for c in cols:
        within = {g: spearman(d[c], d[f"{c}_alt"]) for (g,), d in j.group_by(["pos_group"])}
        g_min = min(within, key=lambda g: within[g])
        rows.append(
            dict(column=c, rho_pooled=spearman(j[c], j[f"{c}_alt"]), rho_group_min=within[g_min], worst_group=g_min)
        )
    return pl.DataFrame(rows)


def sweep_cases(raw: dict) -> list[tuple[str, str, dict]]:
    """(param, bound, raw config) for one-at-a-time bounds and the all-low / all-high corners.

    A bound equal to the default (e.g. broad.theta_deg high = 45) is skipped.
    """
    params = list(config.sweep_params(raw))
    cases = []
    for p in params:
        for bound, v in (("low", p.low), ("high", p.high)):
            if v != p.value:
                cases.append((p.key, bound, config.with_override(raw, p.key, v)))
    for bound in ("low", "high"):
        corner = raw
        for p in params:
            corner = config.with_override(corner, p.key, getattr(p, bound))
        cases.append(("all", bound, corner))
    return cases


def sensitivity(prospects, forty_reps, cod_reps, raw: dict) -> pl.DataFrame:
    """Rank stability of every channel and composite for each sweep case."""
    base = build_tee(prospects, forty_reps, cod_reps, config.values(raw))
    rho_min = config.values(raw)["pipeline"]["spearman_min"]
    out = []
    for key, bound, cfg in sweep_cases(raw):
        alt = build_tee(prospects, forty_reps, cod_reps, config.values(cfg))
        out.append(rank_stability(base, alt).with_columns(pl.lit(key).alias("param"), pl.lit(bound).alias("bound")))
    return (
        pl.concat(out)
        .with_columns((pl.min_horizontal("rho_pooled", "rho_group_min") >= rho_min).alias("pass"))
        .select("param", "bound", "column", "rho_pooled", "rho_group_min", "worst_group", "pass")
    )


def plots(tee: pl.DataFrame, outdir) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(7, 5))
    for g in GROUPS:
        d = tee.filter((pl.col("pos_group") == g) & pl.col("E_40").is_not_null())
        ax.scatter(d["mass_kg"], d["E_40"] / 1000, s=12, alpha=0.7, label=g)
    ax.set(xlabel="Combine mass [kg]", ylabel="E_40 [kJ]", title="40-yd energy vs mass")
    ax.legend()
    paths.append(outdir / "sanity_e40_vs_mass.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.flat, (*REL_CHANNELS, *COMPOSITES)):
        data = [tee.filter(pl.col("pos_group") == g)[c].drop_nulls().to_numpy() for g in GROUPS]
        ax.boxplot(data, tick_labels=GROUPS, showfliers=True)
        ax.set_title(c, fontsize=10)
    fig.suptitle("Per-kg channels and composites by position group")
    fig.tight_layout()
    paths.append(outdir / "sanity_by_group.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    from eeg.config import PROJECT_ROOT

    interim, processed = PROJECT_ROOT / "data" / "interim", PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    raw = config.load_raw()
    cfg = config.values(raw)
    prospects = pl.read_parquet(interim / "combine_si.parquet")
    forty = pl.read_parquet(interim / "forty_fits.parquet")
    cod_reps = pl.read_parquet(interim / "cod_reps.parquet")
    tee = build_tee(prospects, forty, cod_reps, cfg)

    flags = outlier_flags(tee, cfg["pipeline"]["outlier_sd"])
    flags.write_parquet(processed / "tee_outliers.parquet")
    print(f"{flags.height} values beyond {cfg['pipeline']['outlier_sd']} SD -> data/processed/tee_outliers.parquet")
    if flags.height:
        print(flags)

    sens = sensitivity(prospects, forty, cod_reps, raw)
    sens.write_csv(reports / "sensitivity.csv", float_precision=4)
    print(f"sensitivity: {sens['pass'].sum()}/{sens.height} pass (rho >= {cfg['pipeline']['spearman_min']}) -> reports/sensitivity.csv")
    print(
        sens.group_by("param", "bound", maintain_order=True).agg(
            pl.col("rho_pooled").min(), pl.col("rho_group_min").min(), pl.col("pass").all()
        )
    )
    with pl.Config(tbl_rows=-1):
        print(sens.sort("rho_group_min").head(10))

    for p in plots(tee, reports / "figures"):
        print(f"wrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
