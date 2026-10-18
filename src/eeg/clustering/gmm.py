"""Gaussian mixture test for engine archetypes: BIC over k (from 1) × covariance type.

    uv run python -m eeg.clustering.gmm

On data/processed/features.parquet, three feature sets are scored:
- all: log mass + the four mass-residual engine features
- engine: the four engine features only (column 0, log mass, dropped)
- mass: log mass only

plus a **separable** model: a mixture on log mass times one Gaussian for the
engine features, independent of it. If the separable model has the lowest BIC,
the joint mixture's components are size classes and the engine is one cloud.

The engine-only choice of k is bootstrapped (fraction of resamples choosing each k).
Soft membership is exported for the size classes (the mass mixture), ordered from
lightest to heaviest.

Writes reports/gmm_selection.csv (every set × covariance × k), reports/gmm_report.md,
data/processed/gmm_membership.parquet and reports/figures/gmm_bic.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture

K_RANGE = range(1, 9)
COV_TYPES = ("full", "tied", "diag", "spherical")
N_INIT = 10
N_BOOT = 50
SEED = 0


def fit_gmm(X: np.ndarray, k: int, cov: str = "full", seed: int = SEED, n_init: int = N_INIT) -> GaussianMixture:
    return GaussianMixture(k, covariance_type=cov, n_init=n_init, random_state=seed).fit(X)


def bic_grid(X: np.ndarray, ks=K_RANGE, covs=COV_TYPES, seed: int = SEED, n_init: int = N_INIT) -> pl.DataFrame:
    """BIC, log-likelihood and parameter count for every covariance type × k, best first."""
    rows = []
    for cov in covs:
        for k in ks:
            g = fit_gmm(X, k, cov, seed, n_init)
            rows.append(
                dict(cov=cov, k=k, bic=float(g.bic(X)), loglik=float(g.score(X) * len(X)), n_params=g._n_parameters())
            )
    return pl.DataFrame(rows).sort("bic").with_columns((pl.col("bic") - pl.col("bic").min()).alias("delta_bic"))


def best(grid: pl.DataFrame) -> dict:
    return grid.sort("bic").row(0, named=True)


def separable_bic(X: np.ndarray, mass_grid: pl.DataFrame) -> dict:
    """BIC of the best mixture on column 0 times a single full Gaussian on the other columns."""
    m = best(mass_grid)
    engine = fit_gmm(X[:, 1:], 1)
    loglik = m["loglik"] + float(engine.score(X[:, 1:]) * len(X))
    n_params = m["n_params"] + engine._n_parameters()
    return dict(
        cov=f"mass {m['cov']} × engine full",
        k=m["k"],
        bic=-2 * loglik + n_params * np.log(len(X)),
        loglik=loglik,
        n_params=n_params,
    )


def bootstrap_k(X: np.ndarray, n_boot: int = N_BOOT, ks=range(1, 6), covs=COV_TYPES, seed: int = SEED) -> pl.DataFrame:
    """How often each k has the lowest BIC (over covariance types) across bootstrap resamples."""
    rng = np.random.default_rng(seed)
    chosen = []
    for b in range(n_boot):
        Z = X[rng.integers(0, len(X), len(X))]
        chosen.append(best(bic_grid(Z, ks, covs, seed=b, n_init=3))["k"])
    counts = np.bincount(chosen, minlength=max(ks) + 1)
    return pl.DataFrame({"k": list(ks), "share": [counts[k] / n_boot for k in ks]})


def size_membership(mass: np.ndarray, k: int, cov: str) -> tuple[np.ndarray, np.ndarray]:
    """P(size class) per prospect and the class means, classes ordered by mean log mass."""
    g = fit_gmm(mass[:, None], k, cov)
    order = np.argsort(g.means_[:, 0])
    return g.predict_proba(mass[:, None])[:, order], g.means_[order, 0]


def plot_bic(grids: dict[str, pl.DataFrame], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(grids), figsize=(5 * len(grids), 3.6))
    for ax, (name, grid) in zip(axes, grids.items()):
        for cov in COV_TYPES:
            g = grid.filter(pl.col("cov") == cov).sort("k")
            ax.plot(g["k"], g["bic"], "o-", label=cov)
        ax.set_title(f"BIC: {name}")
        ax.set_xlabel("k")
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT

    processed, reports = PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "reports"
    features = pl.read_parquet(processed / "features.parquet")
    cols = features.columns[1:]
    X = features.drop("player_id").to_numpy()

    grids = {"all": bic_grid(X), "engine": bic_grid(X[:, 1:]), "mass": bic_grid(X[:, :1])}
    sep = separable_bic(X, grids["mass"])
    pl.concat(g.with_columns(pl.lit(name).alias("features")) for name, g in grids.items()).select(
        "features", "cov", "k", "bic", "delta_bic", "loglik", "n_params"
    ).sort("features", "cov", "k").write_csv(reports / "gmm_selection.csv", float_precision=2)

    b_all, b_mass = best(grids["all"]), best(grids["mass"])
    joint = fit_gmm(X, b_all["k"], b_all["cov"])
    order = np.argsort(joint.means_[:, 0])
    means = (
        pl.DataFrame(joint.means_[order], schema=cols, orient="row")
        .insert_column(0, pl.Series("component", [f"{i + 1}" for i in range(b_all["k"])]))
        .with_columns(pl.Series("weight", joint.weights_[order]))
    )
    candidates = pl.DataFrame(
        [
            dict(
                model=f"joint mixture ({b_all['cov']}, k = {b_all['k']})", bic=b_all["bic"], n_params=b_all["n_params"]
            ),
            dict(
                model=f"separable: mass k = {sep['k']} × one engine Gaussian", bic=sep["bic"], n_params=sep["n_params"]
            ),
        ]
    ).with_columns((pl.col("bic") - pl.col("bic").min()).alias("delta_bic"))
    boot = bootstrap_k(X[:, 1:])

    probs, class_means = size_membership(X[:, 0], b_mass["k"], b_mass["cov"])
    membership = pl.DataFrame({"player_id": features["player_id"], "size_class": probs.argmax(axis=1) + 1})
    membership = membership.with_columns(pl.Series(f"p_size_{i + 1}", probs[:, i]) for i in range(probs.shape[1]))
    membership.write_parquet(processed / "gmm_membership.parquet")
    # post hoc only: which positions fall in each size class
    positions = pl.read_parquet(processed / "features_context.parquet").select("player_id", "pos_group")
    by_pos = (
        membership.join(positions, on="player_id")
        .pivot(on="size_class", index="pos_group", values="player_id", aggregate_function="len", sort_columns=True)
        .fill_null(0)
        .sort("pos_group")
    )
    plot_bic(grids, reports / "figures" / "gmm_bic.png")

    report = f"""# Phase 3 GMM test

Generated by `uv run python -m eeg.clustering.gmm` on {features.height} prospects. BIC over k = 1..{max(K_RANGE)}
× covariance type ({", ".join(COV_TYPES)}); lower is better. Full grid: `reports/gmm_selection.csv`.

## Best model per feature set

{_md(pl.DataFrame([dict(features=name, **{c: best(g)[c] for c in ("cov", "k", "bic")}) for name, g in grids.items()]), 1)}

## Joint mixture vs separable (size classes × one engine cloud)

{_md(candidates, 1)}

Joint mixture component means (z-scores, ordered by log mass):

{_md(means)}

## Engine-only k across {N_BOOT} bootstrap resamples

{_md(boot)}

## Size classes (mass mixture, {b_mass["cov"]}, k = {b_mass["k"]})

Class means of z(log mass): {", ".join(f"{m:.2f}" for m in class_means)}. Membership: `data/processed/gmm_membership.parquet`.

Prospects per size class by position group (read after fitting):

{_md(by_pos)}
"""
    (reports / "gmm_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
