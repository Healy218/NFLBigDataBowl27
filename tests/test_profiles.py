import numpy as np
import polars as pl
import pytest

from eeg.clustering import profiles as P


def _engine(n=600, seed=0):
    """4 features with a clear eigen-structure: a shared factor, a contrast, and noise."""
    rng = np.random.default_rng(seed)
    g, c = rng.normal(size=(2, n))
    X = np.column_stack([g + c, g + 0.5 * c, g - 0.5 * c, g - c]) + 0.6 * rng.normal(size=(n, 4))
    return X * [2.0, 1.0, 0.5, 3.0] + [10, 0, -5, 1]  # scale and location don't matter (correlation PCA)


def test_scores_are_standardized_and_uncorrelated():
    X = _engine()
    S = P.scores(X, P.fit_axes(X))
    assert np.allclose(S.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(np.cov(S.T), np.eye(4), atol=1e-10)


def test_anomaly_is_mahalanobis_distance():
    X = _engine()
    S = P.scores(X, P.fit_axes(X))
    d2, p = P.anomaly(S, 4)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    assert np.allclose(d2, np.einsum("ij,jk,ik->i", Z, np.linalg.inv(np.corrcoef(X.T)), Z))
    assert ((p > 0) & (p <= 1)).all() and abs((p < 0.05).mean() - 0.05) < 0.03
    assert (P.anomaly(S, 2)[0] <= d2).all()


def test_match_axes_undoes_order_and_sign():
    X = _engine()
    ref = P.fit_axes(X)
    shuffled = dict(ref, ev=ref["ev"][[2, 0, 3, 1]], vec=ref["vec"][:, [2, 0, 3, 1]] * [1, -1, -1, 1])
    matched, cong = P.match_axes(ref, shuffled)
    assert np.allclose(matched["vec"], ref["vec"]) and np.allclose(matched["ev"], ref["ev"])
    assert np.allclose(cong, 1)


def test_score_reliability():
    fit = dict(ev=np.ones(3), vec=np.eye(3))  # independent features: each axis is one feature
    assert np.allclose(P.score_reliability(fit, np.array([1.0, 0.5, 0.8])), [1.0, 0.5, 0.8])
    X = _engine()
    rel = P.score_reliability(P.fit_axes(X), np.ones(4))
    assert np.allclose(rel, 1)


def test_residual_reliability_recovers_noise_share():
    rng = np.random.default_rng(3)
    n = 3000
    true = rng.normal(0, 0.1, n)  # residual SD 0.1 of log value, rep noise SD 0.1 / sqrt(3) -> reliability 0.75
    noise_sd = 0.1 / np.sqrt(3)
    reps = pl.DataFrame(
        {
            "player_id": np.repeat(np.arange(n), 2),
            "rep_id": [f"{i}_{k}" for i in range(n) for k in "ab"],
            "tau": np.exp(np.repeat(true, 2) + rng.normal(0, noise_sd, 2 * n)),
        }
    )
    sd = float(np.sqrt(0.1**2 + noise_sd**2))
    rr = P.residual_reliability(reps, {"tau": sd}, cols=("tau",))
    assert rr["tau"]["n"] == n and rr["tau"]["reliability"] == pytest.approx(0.75, abs=0.03)


def test_bootstrap_and_cohort_stability():
    X = _engine()
    boot = P.bootstrap_stability(X, n_axes=2, n_boot=10)
    assert set(boot["axis"]) == {"1", "2", "3", "4", "anomaly"} and boot.height == 10 * 5
    first = boot.filter(pl.col("axis") == "1")
    assert first["congruence"].min() > 0.99 and first["spearman"].min() > 0.99
    assert boot.filter(pl.col("axis") == "anomaly")["spearman"].min() > 0.9
    loyo = P.cohort_stability(X, np.repeat([2023, 2024, 2025], 200))
    assert loyo["left_out"].to_list() == ["2023", "2024", "2025"] and loyo["axis_1"].min() > 0.99


def test_build_profiles_columns():
    X = _engine(n=50)
    features = pl.DataFrame({"player_id": np.arange(50), **{f"f_{i}": X[:, i] for i in range(4)}})
    membership = pl.DataFrame({"player_id": np.arange(50), "size_class": 1, "p_size_1": 1.0})
    prof = P.build_profiles(features, membership, [f"f_{i}" for i in range(4)], n_axes=3)
    assert prof.columns == [
        "player_id",
        "ax_explosive",
        "ax_speed_vs_bounce",
        "ax_up_vs_out",
        "anomaly_d2",
        "anomaly_p",
        "size_class",
        "p_size_1",
    ]
    assert prof["player_id"].to_list() == list(range(50))
