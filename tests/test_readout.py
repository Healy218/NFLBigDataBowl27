import numpy as np
import polars as pl
import pytest

from eeg.clustering import readout as R


def _clouds(n=300, shift=0.0, seed=0):
    """3-D Gaussian points with a label; label 'B' is shifted by `shift` on axis 0."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(np.array(["A", "B", "C"]), n)
    X = rng.normal(size=(n, 3))
    X[:, 0] += shift * (labels == "B")
    return X, labels


def test_axes_by_position_eta_squared():
    X, labels = _clouds(shift=2.0)
    df = pl.DataFrame({"pos_group": labels, "a0": X[:, 0], "a1": X[:, 1]})
    summary, eta = R.axes_by_position(df, ["a0", "a1"])
    assert summary["n"].sum() == len(X)
    e = dict(zip(eta["axis"], eta["eta_sq"]))
    assert e["a0"] > 0.3 and e["a1"] < 0.05
    assert eta.filter(pl.col("axis") == "a0")["anova_p"][0] < 1e-10


def test_nn_position_test_null_and_signal():
    X, labels = _clouds()
    null = R.nn_position_test(X, labels, n_perm=200)
    assert abs(null["z"]) < 3 and null["null_mean"] == pytest.approx(1 / 3, abs=0.05)
    Xs, ls = _clouds(shift=8.0)
    assert R.nn_position_test(Xs, ls, n_perm=200)["z"] > 5


def test_engine_twins_cross_position_and_mutual():
    X, labels = _clouds(n=120)
    ids = np.arange(1000, 1120)
    twins = R.engine_twins(X, ids, labels)
    assert (twins["pos_group"] != twins["twin_pos_group"]).all()
    lookup = dict(zip(twins["player_id"], twins["twin_id"]))
    for a, mutual in zip(twins["player_id"], twins["mutual"]):
        assert mutual == (lookup[lookup[a]] == a)
    pairs = R.twin_pairs(twins)
    assert pairs["n_pairs"].sum() == twins["mutual"].sum() // 2
    assert all(p.split("–")[0] <= p.split("–")[1] for p in pairs["pair"])


def test_size_relative_anomaly_reduces_to_pooled_with_one_class():
    X, _ = _clouds()
    d2, p = R.size_relative_anomaly(X, np.ones((len(X), 1)))
    D = X - X.mean(axis=0)
    expected = np.einsum("ij,jk,ik->i", D, np.linalg.inv(np.cov(X.T, ddof=0)), D)
    assert np.allclose(d2, expected) and ((p > 0) & (p <= 1)).all()
    # a class with twice the spread no longer looks anomalous when scored against itself
    wide = np.r_[X[:150], 2 * X[150:]]
    member = np.zeros((len(X), 2))
    member[:150, 0], member[150:, 1] = 1, 1
    pooled = R.size_relative_anomaly(wide, np.ones((len(X), 1)))[1]
    by_class = R.size_relative_anomaly(wide, member)[1]
    assert (by_class[150:] < 0.05).sum() < (pooled[150:] < 0.05).sum()


def test_descriptor_correlations_within_removes_position_shift():
    rng = np.random.default_rng(4)
    n = 400
    labels = np.repeat(["A", "B"], n // 2)
    shift = np.where(labels == "B", 3.0, 0.0)
    df = pl.DataFrame({"pos_group": labels, "ax": rng.normal(size=n) + shift, "d": rng.normal(size=n) + shift})
    df = df.with_columns(pl.when(pl.int_range(pl.len()) % 5 == 0).then(None).otherwise(pl.col("d")).alias("d"))
    out = R.descriptor_correlations(df, ["d"], ["ax"]).row(0, named=True)
    assert out["n"] == n - n // 5
    assert out["ax_r"] > 0.6 and abs(out["ax_r_within"]) < 0.15


def test_mass_bump_finds_middle_peak():
    rng = np.random.default_rng(5)
    n = 500
    m = rng.uniform(80, 150, n)
    x = np.log(m) - np.log(m).mean()
    df = pl.DataFrame(
        {"combine_year": 2024, "mass_kg": m, "E_vert": np.exp(3 + 0.6 * x - 2.0 * x**2 + rng.normal(0, 0.01, n))}
    )
    bump = R.mass_bump(df, ["E_vert"], n_bins=5)
    assert bump.height == 5 and bump["n"].sum() == n
    v = bump["E_vert"].to_numpy()
    assert v.argmax() in (1, 2, 3) and v[0] < 0 and v[-1] < 0
