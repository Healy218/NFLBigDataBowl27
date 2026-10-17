"""K-Means baseline on the Phase 3 feature matrix.

    uv run python -m eeg.clustering.kmeans

For k = 2..10 on data/processed/features.parquet (already z-scored):
- inertia (within-cluster sum of squares, for the elbow)
- mean silhouette
- gap statistic (Tibshirani et al. 2001): reference data drawn uniformly in the
  box spanned by the data's principal axes; k_gap is the smallest k with
  gap(k) >= gap(k+1) - s(k+1)
- silhouette under a Gaussian null: column 0 (log mass, bimodal) kept as observed,
  the other features redrawn from a Gaussian with their observed covariance. A
  structureless cloud still gets silhouette ~0.2, so only the excess over this null
  (sil_excess_sd, in null SDs) is evidence of real clusters
- ARI against pos_group, read from features_context.parquet only after fitting:
  how much a solution just restates position (the features never see it)

Writes reports/kmeans_selection.csv, data/processed/kmeans_labels.parquet
(player_id, k, cluster for every k) and reports/figures/kmeans_selection.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

K_RANGE = range(2, 11)
N_INIT = 50
SEED = 0


def fit_kmeans(X: np.ndarray, k: int, seed: int = SEED, n_init: int = N_INIT) -> KMeans:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def log_wk(X: np.ndarray, k: int, seed: int = SEED, n_init: int = 10) -> float:
    """log of the pooled within-cluster sum of squares (gap statistic's W_k up to a constant)."""
    return float(np.log(fit_kmeans(X, k, seed, n_init).inertia_))


def gap_statistic(X: np.ndarray, ks, n_refs: int = 20, seed: int = SEED) -> pl.DataFrame:
    """gap(k) = mean_b log W*_kb - log W_k, with s(k) = sd_b · sqrt(1 + 1/B).

    Reference sets are uniform in the bounding box of X rotated onto its principal
    axes (Tibshirani's "b" variant), so correlated features don't inflate the box.
    """
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ vt.T
    lo, hi = Xp.min(axis=0), Xp.max(axis=0)
    refs = [rng.uniform(lo, hi, size=Xp.shape) @ vt for _ in range(n_refs)]
    rows = []
    for k in ks:
        ref = np.array([log_wk(R, k, seed) for R in refs])
        rows.append(
            dict(k=k, gap=float(ref.mean() - log_wk(X, k, seed)), gap_s=float(ref.std() * np.sqrt(1 + 1 / n_refs)))
        )
    return pl.DataFrame(rows)


def gap_choice(gaps: pl.DataFrame) -> int | None:
    """Smallest k with gap(k) >= gap(k+1) - s(k+1); None if no k qualifies."""
    g = gaps.sort("k")
    for i in range(g.height - 1):
        if g["gap"][i] >= g["gap"][i + 1] - g["gap_s"][i + 1]:
            return int(g["k"][i])
    return None


def null_silhouette(X: np.ndarray, k: int, n_draws: int = 20, seed: int = SEED) -> tuple[float, float]:
    """Mean and SD of the k-means silhouette when columns 1.. are Gaussian (observed covariance)."""
    rng = np.random.default_rng(seed)
    cov = np.cov(X[:, 1:].T)
    sil = []
    for b in range(n_draws):
        Z = X.copy()
        Z[:, 1:] = rng.multivariate_normal(X[:, 1:].mean(axis=0), cov, size=len(X))
        sil.append(silhouette_score(Z, fit_kmeans(Z, k, seed + b, n_init=10).labels_))
    return float(np.mean(sil)), float(np.std(sil))


def select_k(X: np.ndarray, positions: np.ndarray, ks=K_RANGE, n_refs: int = 20) -> tuple[pl.DataFrame, dict]:
    """Per-k metrics (plus a gap-only row for k = 1) and the label vector for each k."""
    rows, labels = [], {}
    for k in ks:
        km = fit_kmeans(X, k)
        labels[k] = km.labels_
        sizes = np.bincount(km.labels_)
        sil = float(silhouette_score(X, km.labels_))
        null_mu, null_sd = null_silhouette(X, k)
        rows.append(
            dict(
                k=k,
                inertia=float(km.inertia_),
                silhouette=sil,
                sil_null=null_mu,
                sil_excess_sd=(sil - null_mu) / null_sd,
                ari_pos_group=float(adjusted_rand_score(positions, km.labels_)),
                min_cluster=int(sizes.min()),
                max_cluster=int(sizes.max()),
            )
        )
    # the gap statistic also scores k = 1 (no clusters); the other metrics need k >= 2
    gaps = gap_statistic(X, [1, *ks], n_refs)
    metrics = gaps.join(pl.DataFrame(rows), on="k", how="left").select(*pl.DataFrame(rows).columns, "gap", "gap_s")
    return metrics, labels


def plot_selection(metrics: pl.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = metrics["k"].to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    axes[0].plot(k, metrics["inertia"], "o-")
    axes[0].set_title("Inertia (elbow)")
    axes[1].plot(k, metrics["silhouette"], "o-", label="data")
    axes[1].plot(k, metrics["sil_null"], "s--", label="Gaussian null")
    axes[1].set_title("Mean silhouette")
    axes[1].legend()
    axes[2].errorbar(k, metrics["gap"], yerr=metrics["gap_s"], fmt="o-", capsize=3)
    axes[2].set_title("Gap statistic (± s)")
    axes[3].plot(k, metrics["ari_pos_group"], "o-")
    axes[3].set_title("ARI vs position (lower = less positional)")
    for ax in axes:
        ax.set_xlabel("k")
        ax.set_xticks(k)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg.config import PROJECT_ROOT

    processed, reports = PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "reports"
    features = pl.read_parquet(processed / "features.parquet")
    positions = (
        features.select("player_id")
        .join(pl.read_parquet(processed / "features_context.parquet").select("player_id", "pos_group"), on="player_id")
        .get_column("pos_group")
        .to_numpy()
    )
    X = features.drop("player_id").to_numpy()
    metrics, labels = select_k(X, positions)
    metrics.write_csv(reports / "kmeans_selection.csv", float_precision=4)
    pl.concat(
        pl.DataFrame({"player_id": features["player_id"], "k": k, "cluster": lab.astype(np.int64)})
        for k, lab in labels.items()
    ).write_parquet(processed / "kmeans_labels.parquet")
    plot_selection(metrics, reports / "figures" / "kmeans_selection.png")
    with pl.Config(tbl_cols=-1, tbl_width_chars=160):
        print(metrics)
    print(f"gap statistic choice: k = {gap_choice(metrics)}")
    print(f"silhouette max: k = {metrics.drop_nulls('silhouette').sort('silhouette', descending=True)['k'][0]}")


if __name__ == "__main__":
    main()
