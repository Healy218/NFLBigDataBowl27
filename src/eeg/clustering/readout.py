"""Post-hoc read of the kinetic profiles against position groups.

    uv run python -m eeg.clustering.readout

Positions enter here for the first time, only to read results that were fit without them:

- Axes by position group: means, SDs, and η² (share of each axis's variance
  that position explains), with a one-way ANOVA p-value.
- Nearest-neighbour position match: how often a prospect's nearest neighbour has
  the same position, in engine space (the profile axes) and, for contrast, in the
  full feature space with log mass; against a label-permutation null.
- Engine twins: each prospect's nearest neighbour with a *different* position;
  mutual pairs are twins. Counted by position pair.
- Kinetic Anomalies by position group and size class, against the χ² expectation;
  engine spread (axis SDs) by position group; and, as a robustness check, a
  size-relative anomaly score (Mahalanobis distance under each size class's own
  covariance, weighted by membership probability).
- Descriptors (E_40, bench, shuttle, cone mass-residual z-scores) against the
  axes: pooled r and within-position r (both demeaned by position group).
- The middle-mass bump: power-law (degree-1) residuals of vert, broad and v_max
  averaged by mass decile.

Writes reports/phase3_readout.md (aggregate statistics only),
data/processed/engine_twins.parquet and data/processed/anomalies.parquet
(per-prospect, not committed), and reports/figures/readout_axes_by_position.png,
readout_mass_bump.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import chi2, f_oneway
from sklearn.neighbors import NearestNeighbors

from eeg.clustering.features import mass_residual

POS_ORDER = ("WR", "DB", "TE", "DL", "OL")
N_PERM = 1000
SEED = 0


def axes_by_position(df: pl.DataFrame, axes: list[str], pos: str = "pos_group") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-position mean / SD of each axis, and per-axis η² with its ANOVA p-value."""
    summary = (
        df.group_by(pos)
        .agg(pl.len().alias("n"), *(pl.col(a).mean().alias(f"{a}_mean") for a in axes))
        .sort(pl.col(pos).replace_strict({p: i for i, p in enumerate(POS_ORDER)}, default=len(POS_ORDER)))
    )
    rows = []
    for a in axes:
        y = df[a].to_numpy()
        groups = [g[a].to_numpy() for _, g in df.group_by(pos)]
        between = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
        rows.append(dict(axis=a, eta_sq=between / ((y - y.mean()) ** 2).sum(), anova_p=float(f_oneway(*groups).pvalue)))
    return summary, pl.DataFrame(rows)


def nn_same_label_rate(X: np.ndarray, labels: np.ndarray) -> float:
    """Share of points whose nearest other point has the same label."""
    _, idx = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
    return float((labels[idx[:, 1]] == labels).mean())


def nn_position_test(X: np.ndarray, labels: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Observed same-position nearest-neighbour rate vs labels permuted over the same points."""
    rng = np.random.default_rng(seed)
    _, idx = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
    nn = idx[:, 1]
    null = np.array([(p[nn] == p).mean() for p in (rng.permutation(labels) for _ in range(n_perm))])
    obs = float((labels[nn] == labels).mean())
    return dict(
        observed=obs, null_mean=float(null.mean()), null_sd=float(null.std()), z=(obs - null.mean()) / null.std()
    )


def engine_twins(X: np.ndarray, ids: np.ndarray, labels: np.ndarray) -> pl.DataFrame:
    """For each point, its nearest neighbour with a different label; `mutual` when the choice is reciprocal."""
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    D[labels[:, None] == labels[None, :]] = np.inf
    j = D.argmin(axis=1)
    mutual = j[j] == np.arange(len(X))
    return pl.DataFrame(
        {
            "player_id": ids,
            "pos_group": labels,
            "twin_id": ids[j],
            "twin_pos_group": labels[j],
            "distance": D[np.arange(len(X)), j],
            "mutual": mutual,
        }
    )


def twin_pairs(twins: pl.DataFrame) -> pl.DataFrame:
    """Mutual twin pairs counted by (unordered) position pair."""
    m = twins.filter("mutual").with_columns(
        pl.concat_str(
            pl.min_horizontal("pos_group", "twin_pos_group"),
            pl.max_horizontal("pos_group", "twin_pos_group"),
            separator="–",
        ).alias("pair")
    )
    return m.group_by("pair").agg((pl.len() // 2).alias("n_pairs")).sort("n_pairs", descending=True)


def anomaly_table(df: pl.DataFrame, by: str, alpha: float) -> pl.DataFrame:
    """Prospects with anomaly_p < alpha per group, against alpha × group size."""
    return (
        df.group_by(by)
        .agg(pl.len().alias("n"), (pl.col("anomaly_p") < alpha).sum().alias("observed"))
        .with_columns((pl.col("n") * alpha).alias("expected"))
        .sort(by)
    )


def size_relative_anomaly(S: np.ndarray, membership: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Squared Mahalanobis distance under each size class's own mean and covariance, weighted by P(class).

    S: axis scores (n × d); membership: P(size class) (n × k). Returns the distance and its χ²_d tail probability.
    """
    d2 = np.zeros(len(S))
    for w in membership.T:
        mu = (w[:, None] * S).sum(axis=0) / w.sum()
        D = S - mu
        cov = (w[:, None] * D).T @ D / w.sum()
        d2 += w * np.einsum("ij,jk,ik->i", D, np.linalg.inv(cov), D)
    return d2, chi2.sf(d2, S.shape[1])


def descriptor_correlations(
    df: pl.DataFrame, descriptors: list[str], axes: list[str], pos: str = "pos_group"
) -> pl.DataFrame:
    """r of each descriptor with each axis: pooled, and within position (both sides demeaned by position)."""
    rows = []
    for d in descriptors:
        sub = df.drop_nulls(d)
        within = sub.with_columns((pl.col(c) - pl.col(c).mean().over(pos)).alias(c) for c in (d, *axes))
        row = dict(descriptor=d, n=sub.height)
        for a in axes:
            row[f"{a}_r"] = sub.select(pl.corr(d, a)).item()
            row[f"{a}_r_within"] = within.select(pl.corr(d, a)).item()
        rows.append(row)
    return pl.DataFrame(rows)


def mass_bump(context: pl.DataFrame, cols: list[str], n_bins: int = 10) -> pl.DataFrame:
    """Binned mean power-law residual (degree 1) of each column by log-mass decile."""
    d = context.with_columns(
        *(mass_residual(context, c, degree=1)[0].alias(c) for c in cols),
        ((pl.col("mass_kg").rank("ordinal") - 1) * n_bins // pl.len()).alias("bin"),
    )
    return (
        d.group_by("bin")
        .agg(
            pl.col("mass_kg").median().alias("mass_kg"),
            pl.len().alias("n"),
            *(pl.col(c).mean().alias(c) for c in cols),
            *((pl.col(c).std() / pl.len().sqrt()).alias(f"{c}_se") for c in cols),
        )
        .sort("mass_kg")
    )


def plot_axes_by_position(df: pl.DataFrame, axes: list[str], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax_list = plt.subplots(1, len(axes), figsize=(4.2 * len(axes), 3.6), sharey=True)
    for ax, a in zip(ax_list, axes):
        data = [df.filter(pl.col("pos_group") == p)[a].to_numpy() for p in POS_ORDER]
        ax.boxplot(data, tick_labels=POS_ORDER, showfliers=False)
        for i, y in enumerate(data):
            ax.scatter(
                np.full(len(y), i + 1) + np.random.default_rng(i).uniform(-0.15, 0.15, len(y)), y, s=5, alpha=0.4
            )
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(a)
    ax_list[0].set_ylabel("axis score (SD)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mass_bump(bump: pl.DataFrame, cols: list[str], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in cols:
        ax.errorbar(bump["mass_kg"], bump[c], yerr=bump[f"{c}_se"], fmt="o-", capsize=3, label=c)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("body mass (kg; decile medians)")
    ax.set_ylabel("log residual from power law")
    ax.set_title("Output for size peaks at middle mass")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT

    cfg = config.load()
    processed, reports = PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "reports"
    profiles = pl.read_parquet(processed / "profiles.parquet")
    context = pl.read_parquet(processed / "features_context.parquet")
    features = pl.read_parquet(processed / "features.parquet")
    axes = [c for c in profiles.columns if c.startswith("ax_")]
    descriptors = [f"d_{c}" for c in cfg["clustering"]["descriptors"]]
    df = profiles.join(context.select("player_id", "pos_group", "combine_year", *descriptors), on="player_id")
    labels = df["pos_group"].to_numpy()

    summary, eta = axes_by_position(df, axes)
    X_engine = df.select(axes).to_numpy()
    X_full = df.select("player_id").join(features, on="player_id").drop("player_id").to_numpy()
    nn = pl.DataFrame(
        [
            dict(space="engine axes (size excluded)", **nn_position_test(X_engine, labels)),
            dict(space="all features (with log mass)", **nn_position_test(X_full, labels)),
        ]
    )

    twins = engine_twins(X_engine, df["player_id"].to_numpy(), labels)
    twins.write_parquet(processed / "engine_twins.parquet")
    pairs = twin_pairs(twins)

    anomalies = df.filter(pl.col("anomaly_p") < 0.05).select(
        "player_id", "pos_group", "size_class", *axes, "anomaly_d2", "anomaly_p"
    )
    anomalies.sort("anomaly_p").write_parquet(processed / "anomalies.parquet")
    a_pos = anomaly_table(df, "pos_group", 0.05)
    a_size = anomaly_table(df, "size_class", 0.05)
    driver = (
        anomalies.with_columns(
            pl.Series("driver", np.array(axes)[np.abs(anomalies.select(axes).to_numpy()).argmax(axis=1)])
        )
        .group_by("driver")
        .agg(pl.len().alias("n"))
        .sort("driver")
    )
    spread = summary.select("pos_group").join(
        df.group_by("pos_group").agg(
            *(pl.col(a).std().alias(f"{a}_sd") for a in axes), pl.col("anomaly_d2").mean().alias("mean_anomaly_d2")
        ),
        on="pos_group",
        maintain_order="left",
    )
    _, p_size = size_relative_anomaly(df.select(axes).to_numpy(), df.select(pl.col("^p_size_.*$")).to_numpy())
    by_size = df.with_columns(pl.Series("p_size_rel", p_size))
    robust = a_pos.select("pos_group", "n", pl.col("observed").alias("pooled")).join(
        by_size.group_by("pos_group").agg((pl.col("p_size_rel") < 0.05).sum().alias("size_relative")),
        on="pos_group",
        maintain_order="left",
    )
    rho_size = by_size.select(pl.corr("anomaly_p", "p_size_rel", method="spearman")).item()

    desc = descriptor_correlations(df, descriptors, axes)
    bump_cols = ["E_vert", "E_broad", "v_max"]
    bump = mass_bump(context, bump_cols)
    plot_axes_by_position(df, axes, reports / "figures" / "readout_axes_by_position.png")
    plot_mass_bump(bump, bump_cols, reports / "figures" / "readout_mass_bump.png")

    report = f"""# Phase 3 readout: kinetic profiles by position group

Generated by `uv run python -m eeg.clustering.readout` on {df.height} prospects. Positions were not used to fit
anything read here. Per-prospect tables (not committed): `data/processed/engine_twins.parquet`,
`data/processed/anomalies.parquet`.

## Axes by position group

Mean axis score (SD units; every axis has SD 1 over all prospects):

{_md(summary)}

η²: share of each axis's variance explained by position group.

{_md(eta, 3)}

## Does position predict your engine neighbour?

Share of prospects whose nearest neighbour has the same position, against {N_PERM} label permutations.

{_md(nn, 3)}

## Engine twins (mutual nearest neighbours across positions)

{twins["mutual"].sum() // 2} mutual cross-position pairs. Median distance to the nearest other-position prospect:
{twins["distance"].median():.2f} SD (axis space).

{_md(pairs)}

## Kinetic Anomalies (p < 0.05)

{anomalies.height} prospects. By position group and size class, against 5 % of each group:

{_md(a_pos, 1)}

{_md(a_size, 1)}

Axis with the largest |score| per anomaly:

{_md(driver)}

Engine spread by position group (axis SDs; mean anomaly_d2 is {len(axes)} under the χ² model):

{_md(spread)}

Robustness: anomalies at p < 0.05 when each size class is scored against its own covariance (size-relative),
vs the pooled score. Rank agreement of the two p-values: Spearman {rho_size:.2f}.

{_md(robust)}

## Descriptors against the axes

r pooled and r within position (descriptor and axis both demeaned by position group). n = prospects with the drill.

{_md(desc)}

## Middle-mass bump

Mean power-law (degree-1) residual by mass decile (log units; 0.05 ≈ 5 % above the power law). See
`reports/figures/readout_mass_bump.png`.

{_md(bump.select("mass_kg", "n", *bump_cols), 3)}
"""
    (reports / "phase3_readout.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
