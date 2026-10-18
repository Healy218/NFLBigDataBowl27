import numpy as np
import pytest

from eeg.clustering import gmm as G


def _sized(n=400, engine_shift=0.0, seed=0):
    """Column 0 is bimodal 'mass'; columns 1-3 are one Gaussian, shifted by engine_shift in the heavy mode."""
    rng = np.random.default_rng(seed)
    heavy = rng.integers(0, 2, n)
    X = rng.multivariate_normal(np.zeros(4), np.eye(4) * 0.5 + 0.5, size=n)
    X[:, 0] = np.where(heavy == 1, 1.0, -1.0) + 0.25 * rng.normal(size=n)
    X[:, 1:] += engine_shift * heavy[:, None]
    return X, heavy


def test_bic_grid_finds_blobs_and_one_gaussian():
    X, _ = _sized(engine_shift=5.0)
    assert G.best(G.bic_grid(X[:, 1:], ks=range(1, 4), n_init=3))["k"] == 2
    grid = G.bic_grid(np.random.default_rng(1).normal(size=(400, 3)), ks=range(1, 4), n_init=3)
    assert G.best(grid)["k"] == 1
    assert grid["delta_bic"].min() == 0 and grid.height == 3 * len(G.COV_TYPES)


def test_separable_model_wins_only_when_engine_ignores_size_class():
    X, _ = _sized()
    mass = G.bic_grid(X[:, :1], ks=range(1, 4), n_init=3)
    assert G.best(mass)["k"] == 2
    joint = G.best(G.bic_grid(X, ks=range(1, 4), n_init=3))
    assert G.separable_bic(X, mass)["bic"] < joint["bic"]

    Xs, _ = _sized(engine_shift=2.0)
    mass_s = G.bic_grid(Xs[:, :1], ks=range(1, 4), n_init=3)
    assert G.separable_bic(Xs, mass_s)["bic"] > G.best(G.bic_grid(Xs, ks=range(1, 4), n_init=3))["bic"]


def test_bootstrap_k_shares():
    X, _ = _sized()
    boot = G.bootstrap_k(X[:, 1:], n_boot=8, ks=range(1, 4))
    assert boot["share"].sum() == pytest.approx(1.0)
    assert boot.filter(boot["k"] == 1)["share"][0] >= 0.75


def test_size_membership_is_ordered_by_mass():
    X, heavy = _sized()
    probs, means = G.size_membership(X[:, 0], 2, "full")
    assert np.allclose(probs.sum(axis=1), 1) and means[0] < means[1]
    assert ((probs.argmax(axis=1) == heavy).mean()) > 0.99
