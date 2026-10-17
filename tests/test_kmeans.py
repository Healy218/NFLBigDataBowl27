import numpy as np
import polars as pl
import pytest

from eeg.clustering import kmeans as K


def _blobs(n=300, sep=6.0, seed=0):
    """Column 0 is noise (the 'mass' slot); columns 1-3 hold two well-separated blobs."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 4))
    X[:, 1] += sep * lab
    return X, lab


def test_gap_choice_rule():
    g = pl.DataFrame({"k": [1, 2, 3], "gap": [0.5, 0.9, 0.95], "gap_s": [0.02, 0.02, 0.1]})
    assert K.gap_choice(g) == 2  # gap(2) >= gap(3) - s(3)
    assert K.gap_choice(g.with_columns(pl.Series("gap", [1.0, 0.9, 0.8]))) == 1
    assert K.gap_choice(pl.DataFrame({"k": [1, 2], "gap": [0.1, 0.9], "gap_s": [0.0, 0.0]})) is None


def test_gap_finds_real_clusters_and_none_in_a_gaussian():
    X, _ = _blobs()
    assert K.gap_choice(K.gap_statistic(X, [1, 2, 3, 4], n_refs=10)) == 2
    G = np.random.default_rng(1).multivariate_normal(np.zeros(4), np.eye(4) + 0.5, size=300)
    assert K.gap_choice(K.gap_statistic(G, [1, 2, 3, 4], n_refs=10)) == 1


def test_null_silhouette_separates_structure_from_a_cloud():
    X, _ = _blobs()
    from sklearn.metrics import silhouette_score

    sil = silhouette_score(X, K.fit_kmeans(X, 2, n_init=10).labels_)
    mu, sd = K.null_silhouette(X, 2, n_draws=5)
    assert (sil - mu) / sd > 5
    G = np.random.default_rng(2).normal(size=(300, 4))
    sil_g = silhouette_score(G, K.fit_kmeans(G, 2, n_init=10).labels_)
    mu_g, sd_g = K.null_silhouette(G, 2, n_draws=10)
    assert abs(sil_g - mu_g) < 4 * sd_g


def test_select_k_reports_position_agreement():
    X, lab = _blobs()
    positions = np.where(lab == 1, "OL", "WR")
    metrics, labels = K.select_k(X, positions, ks=[2, 3], n_refs=5)
    assert metrics["k"].to_list() == [1, 2, 3]
    row2 = metrics.filter(pl.col("k") == 2).row(0, named=True)
    assert row2["ari_pos_group"] == pytest.approx(1.0) and row2["min_cluster"] + row2["max_cluster"] == len(X)
    assert metrics.filter(pl.col("k") == 1)["silhouette"][0] is None
    assert set(labels) == {2, 3} and len(labels[2]) == len(X)
