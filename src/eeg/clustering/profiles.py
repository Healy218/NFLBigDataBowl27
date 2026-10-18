"""Continuous kinetic profiles: engine axes, Kinetic Anomaly score, size class.

    uv run python -m eeg.clustering.profiles

The GMM test (clustering/gmm.py) found the engine features to be one Gaussian
cloud, independent of the size classes. So each prospect is described by:

- **Engine axes:** principal axes of the correlation matrix of the engine features
  (log mass excluded), as standardized scores (SD 1). The first
  `clustering.profile_axes` are kept and named from the reference loadings:
    1. explosive        vert, broad, v_max (and weakly τ) together: output for your size
    2. speed_vs_bounce  τ, v_max against the jumps: top speed that builds late vs quick, bouncy power
    3. up_vs_out        vert (and τ) against broad, v_max: jumping up vs out
- **Kinetic Anomaly score:** the sum of squared kept-axis scores, i.e. squared
  Mahalanobis distance in that subspace, with its χ² tail probability.
- **Size class** and its probabilities, from data/processed/gmm_membership.parquet.

Stability and reliability:
- bootstrap over prospects: Tucker congruence of each axis with the reference
  (axes matched and sign-aligned), and Spearman r of every prospect's axis scores
  and anomaly score under the resampled fit against the reference
- leave-one-combine-year-out congruence
- score reliability from 40 rep-to-rep noise: residual reliability of v_max and τ
  from players with two well-fit reps, combined through the loadings (the jumps
  have one recorded value each, so their reliability is taken as 1: an upper bound)

Writes data/processed/profiles.parquet, reports/profiles_report.md (aggregate
statistics only) and reports/figures/profiles_map.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2, spearmanr

from eeg.clustering.audit import pca

AXIS_NAMES = ("explosive", "speed_vs_bounce", "up_vs_out")
N_BOOT = 200
SEED = 0


def fit_axes(X: np.ndarray) -> dict:
    """Correlation-matrix PCA of X: eigenvalues, loadings and the standardization used."""
    ev, vec = pca(X)
    return dict(ev=ev, vec=vec, mean=X.mean(axis=0), sd=X.std(axis=0, ddof=1))


def scores(X: np.ndarray, fit: dict) -> np.ndarray:
    """Standardized axis scores (SD 1 on the fitting sample), one column per axis."""
    return ((X - fit["mean"]) / fit["sd"]) @ fit["vec"] / np.sqrt(fit["ev"])


def anomaly(S: np.ndarray, n_axes: int) -> tuple[np.ndarray, np.ndarray]:
    """Squared Mahalanobis distance over the first n_axes scores and its χ² tail probability."""
    d2 = (S[:, :n_axes] ** 2).sum(axis=1)
    return d2, chi2.sf(d2, n_axes)


def match_axes(ref: dict, fit: dict) -> tuple[dict, np.ndarray]:
    """Reorder and sign-flip fit's axes to best match ref's; returns the matched fit and |congruence|.

    Loadings are unit vectors, so Tucker's congruence is their dot product.
    """
    phi = ref["vec"].T @ fit["vec"]
    _, col = linear_sum_assignment(-np.abs(phi))
    sign = np.sign(phi[np.arange(len(col)), col])
    matched = dict(fit, ev=fit["ev"][col], vec=fit["vec"][:, col] * sign)
    return matched, np.abs(phi[np.arange(len(col)), col])


def bootstrap_stability(X: np.ndarray, n_axes: int, n_boot: int = N_BOOT, seed: int = SEED) -> pl.DataFrame:
    """Per draw and axis: congruence with the reference, and Spearman r of all prospects' scores.

    axis = "anomaly" rows hold the Spearman r of the anomaly score (no congruence).
    """
    ref = fit_axes(X)
    S_ref = scores(X, ref)
    d2_ref, _ = anomaly(S_ref, n_axes)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        fit, cong = match_axes(ref, fit_axes(X[rng.integers(0, len(X), len(X))]))
        S = scores(X, fit)
        for j in range(X.shape[1]):
            rows.append(dict(draw=b, axis=str(j + 1), congruence=cong[j], spearman=spearmanr(S_ref[:, j], S[:, j])[0]))
        rows.append(dict(draw=b, axis="anomaly", congruence=None, spearman=spearmanr(d2_ref, anomaly(S, n_axes)[0])[0]))
    return pl.DataFrame(rows)


def cohort_stability(X: np.ndarray, years: np.ndarray) -> pl.DataFrame:
    """Congruence of each axis when one combine year is left out of the fit."""
    ref = fit_axes(X)
    rows = []
    for y in np.unique(years):
        _, cong = match_axes(ref, fit_axes(X[years != y]))
        rows.append(dict(left_out=str(y), **{f"axis_{j + 1}": float(c) for j, c in enumerate(cong)}))
    return pl.DataFrame(rows)


def residual_reliability(forty_reps: pl.DataFrame, resid_sd: dict[str, float], cols=("v_max", "tau")) -> dict:
    """1 − (single-rep error variance / residual variance) of log value, from players' first two reps.

    The mass trend is the same for both reps, so the error is half the variance of
    the log difference between them.
    """
    two = (
        forty_reps.sort("player_id", "rep_id")
        .group_by("player_id", maintain_order=True)
        .head(2)
        .filter(pl.len().over("player_id") == 2)
    )
    out = {}
    for c in cols:
        diff = two.group_by("player_id").agg((pl.col(c).log().first() - pl.col(c).log().last()).alias("d"))["d"]
        out[c] = dict(n=diff.len(), reliability=1 - diff.var() / 2 / resid_sd[c] ** 2)
    return out


def score_reliability(fit: dict, reliability: np.ndarray) -> np.ndarray:
    """Reliability of each standardized axis score given per-feature reliabilities (independent errors)."""
    return 1 - (fit["vec"] ** 2 * (1 - reliability)[:, None]).sum(axis=0) / fit["ev"]


def build_profiles(features: pl.DataFrame, membership: pl.DataFrame, engine: list[str], n_axes: int) -> pl.DataFrame:
    X = features.select(engine).to_numpy()
    S = scores(X, fit_axes(X))
    d2, p = anomaly(S, n_axes)
    out = features.select("player_id").with_columns(
        *(pl.Series(f"ax_{AXIS_NAMES[j]}", S[:, j]) for j in range(n_axes)),
        pl.Series("anomaly_d2", d2),
        pl.Series("anomaly_p", p),
    )
    return out.join(membership, on="player_id", how="left")


def plot_map(profiles: pl.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (x, y) in zip(axes, [("ax_explosive", "ax_speed_vs_bounce"), ("ax_explosive", "ax_up_vs_out")]):
        for k, g in profiles.sort("size_class").group_by("size_class", maintain_order=True):
            ax.scatter(g[x], g[y], s=12, alpha=0.7, label=f"size class {k[0]}")
        top = profiles.filter(pl.col("anomaly_p") < 0.01)
        ax.scatter(top[x], top[y], facecolors="none", edgecolors="red", s=60, label="anomaly p < 0.01")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT

    cfg = config.load()
    n_axes = cfg["clustering"]["profile_axes"]
    raw = cfg["clustering"]["features"]
    engine = [f"f_{c}" for c in raw]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    features = pl.read_parquet(processed / "features.parquet")
    years = features.join(
        pl.read_parquet(processed / "features_context.parquet").select("player_id", "combine_year"), on="player_id"
    )["combine_year"].to_numpy()
    X = features.select(engine).to_numpy()

    profiles = build_profiles(features, pl.read_parquet(processed / "gmm_membership.parquet"), engine, n_axes)
    profiles.write_parquet(processed / "profiles.parquet")
    plot_map(profiles, reports / "figures" / "profiles_map.png")

    ref = fit_axes(X)
    axis_labels = [f"{j + 1} {AXIS_NAMES[j]}" if j < n_axes else f"{j + 1} (dropped)" for j in range(len(engine))]
    loadings = pl.DataFrame(ref["vec"], schema=axis_labels, orient="row").insert_column(0, pl.Series("feature", engine))

    reps = pl.read_parquet(interim / "forty_fits.parquet").filter(
        (pl.col("rmse") <= cfg["sprint"]["max_fit_rmse_mps"])
        & pl.col("player_id").is_in(features["player_id"].implode())
    )
    fits = pl.read_csv(reports / "feature_fits.csv").filter(pl.col("role") == "feature")
    rr = residual_reliability(reps, dict(zip(fits["column"], fits["resid_sd"])))
    feat_rel = np.array([rr[c]["reliability"] if c in rr else 1.0 for c in raw])
    axes = pl.DataFrame(
        {
            "axis": axis_labels,
            "eigenvalue": ref["ev"],
            "var_share": ref["ev"] / ref["ev"].sum(),
            "score_reliability": score_reliability(ref, feat_rel),
        }
    )

    boot = bootstrap_stability(X, n_axes)
    stab = (
        boot.group_by("axis")
        .agg(
            pl.col("congruence").median().alias("congruence_median"),
            pl.col("congruence").quantile(0.05).alias("congruence_p5"),
            pl.col("spearman").median().alias("score_rho_median"),
            pl.col("spearman").quantile(0.05).alias("score_rho_p5"),
        )
        .sort("axis")
    )
    loyo = cohort_stability(X, years)
    calib = pl.DataFrame(
        [
            dict(tail=f"p < {a}", observed=int((profiles["anomaly_p"] < a).sum()), expected=a * profiles.height)
            for a in (0.05, 0.01, 0.001)
        ]
    )

    report = f"""# Phase 3 kinetic profiles

Generated by `uv run python -m eeg.clustering.profiles` on {features.height} prospects. Engine features:
{", ".join(engine)} (mass-residual z-scores; log mass excluded). Kept axes: {n_axes} (`clustering.profile_axes`).
Per-prospect table: `data/processed/profiles.parquet`.

## Axes (correlation-matrix PCA)

{_md(axes)}

Score reliability: 1 − error share of each standardized score, from 40 rep-to-rep noise.
Residual reliability of v_max {rr["v_max"]["reliability"]:.2f} and τ {rr["tau"]["reliability"]:.2f}
({rr["tau"]["n"]} players with two reps); jumps taken as 1 (one recorded value each), so these are upper bounds.

{_md(loadings)}

## Bootstrap stability ({N_BOOT} resamples of prospects)

Congruence: Tucker's φ between resampled and reference loadings, axes matched and sign-aligned (≥ 0.95: same
axis). Score ρ: Spearman r between every prospect's score under the resampled fit and the reference score.

{_md(stab)}

## Leave one combine year out (congruence)

{_md(loyo, 3)}

## Anomaly calibration (χ²{n_axes} tail)

{_md(calib, 1)}
"""
    (reports / "profiles_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
