import math

import numpy as np
import polars as pl
import pytest

from eeg import config
from eeg.physics import sprint as sp
from eeg.physics.bench import bench_rom, work_bench
from eeg.physics.broad import energy_broad, takeoff_velocity
from eeg.physics.vertical import energy_vertical, sayers_peak_power

CFG = config.load()


# --- hand checks (HANDOFF step 6) ---------------------------------------------


def test_vertical_hand_check():
    assert energy_vertical(100.0, 1.0) == pytest.approx(980.665)


def test_broad_hand_check():
    assert energy_broad(100.0, 3.0, theta_deg=45.0) == pytest.approx(1470.9975)


def test_bench_hand_check():
    # 25 reps, ROM 0.40 m (α = 0.5 on a 0.80 m arm)
    assert bench_rom(0.80, alpha=0.5) == pytest.approx(0.40)
    assert work_bench(25, 0.80, alpha=0.5) == pytest.approx(25 * 102.05828 * 9.80665 * 0.40, rel=1e-6)
    assert work_bench(25, 0.80, alpha=0.5) == pytest.approx(10_009, abs=1)


def test_sayers_hand_check():
    v = CFG["vertical"]
    # 100 cm, 100 kg: 60.7*100 + 45.3*100 - 2055
    assert sayers_peak_power(100.0, 1.0, v["sayers_a"], v["sayers_b"], v["sayers_c"]) == pytest.approx(8545.0)


# --- physics properties --------------------------------------------------------


def test_broad_energy_matches_kinetic_energy():
    v0 = takeoff_velocity(3.0, theta_deg=40.0)
    assert 0.5 * 100.0 * v0**2 == pytest.approx(energy_broad(100.0, 3.0, theta_deg=40.0))


def test_broad_45_is_minimum_energy():
    e45 = energy_broad(100.0, 3.0, 45.0)
    assert all(energy_broad(100.0, 3.0, th) > e45 for th in (30.0, 35.0, 40.0, 50.0))
    # sweep bound: θ = 30° gives 1/sin(60°) ≈ 15.5% more
    assert energy_broad(100.0, 3.0, 30.0) / e45 == pytest.approx(1 / math.sin(math.radians(60)))


@pytest.mark.parametrize("theta", [0.0, 90.0, -10.0, 120.0])
def test_broad_rejects_bad_angle(theta):
    with pytest.raises(ValueError, match="takeoff angle"):
        energy_broad(100.0, 3.0, theta)


def test_linear_scaling():
    assert energy_vertical(200.0, 0.5) == pytest.approx(energy_vertical(100.0, 1.0))
    assert work_bench(20, 0.9, 0.5) == pytest.approx(2 * work_bench(10, 0.9, 0.5))


# --- input types: floats, numpy, polars Series and expressions -----------------


def test_vectorised_inputs():
    m = np.array([90.0, 140.0])
    h = np.array([0.9, 0.7])
    np.testing.assert_allclose(energy_vertical(m, h), m * 9.80665 * h)

    df = pl.DataFrame({"mass_kg": [90.0, 140.0], "broad_m": [3.2, 2.6], "arm_m": [0.80, 0.88], "reps": [15, 30]})
    out = df.select(
        e_broad=energy_broad(pl.col("mass_kg"), pl.col("broad_m"), 45.0),
        w_bench=work_bench(pl.col("reps"), pl.col("arm_m"), 0.5),
    )
    assert out["e_broad"].to_list() == pytest.approx([0.5 * 90 * 3.2 * 9.80665, 0.5 * 140 * 2.6 * 9.80665])
    assert out["w_bench"][1] == pytest.approx(30 * 102.05828 * 9.80665 * 0.44, rel=1e-6)


def test_nulls_propagate():
    df = pl.DataFrame({"mass_kg": [100.0, 100.0], "vertical_m": [1.0, None]})
    out = df.select(e=energy_vertical(pl.col("mass_kg"), pl.col("vertical_m")))["e"]
    assert out[0] == pytest.approx(980.665) and out[1] is None


# --- sprint (40-yard dash) -----------------------------------------------------

def _synthetic_rep(v_max=10.0, tau=1.3, t0=0.15, dt=0.1, x0_yd=10.6, noise=0.0, seed=0):
    t = np.arange(0.0, 6.0, dt)
    s = sp.speed_model(t, v_max, tau, t0)
    s = s + np.random.default_rng(seed).normal(0, noise, t.size)
    x_yd = x0_yd + sp.distance_model(t, v_max, tau, t0) / 0.9144
    return t, s, x_yd


def test_speed_and_distance_models_are_consistent():
    t = np.linspace(0, 6, 6001)
    v = sp.speed_model(t, 10.0, 1.3, 0.2)
    x = sp.distance_model(t, 10.0, 1.3, 0.2)
    assert np.trapezoid(v, t) == pytest.approx(x[-1], rel=1e-6)
    assert sp.speed_model(0.1, 10.0, 1.3, 0.2) == 0.0  # before t0


def test_time_to_distance_inverts_distance_model():
    T = sp.time_to_distance(36.576, 10.0, 1.3)
    assert sp.distance_model(T, 10.0, 1.3, 0.0) == pytest.approx(36.576, abs=1e-9)


@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_fit_trace_recovers_parameters(noise):
    t, s, _ = _synthetic_rep(noise=noise)
    fit = sp.fit_trace(t, s)
    tol = 1e-4 if noise == 0 else 0.1
    assert fit.v_max == pytest.approx(10.0, abs=tol * 2)
    assert fit.tau == pytest.approx(1.3, abs=tol)
    assert fit.t0 == pytest.approx(0.15, abs=tol)


def test_fit_splits_recovers_parameters():
    t10 = sp.time_to_distance(10 * 0.9144, 9.5, 1.1)
    t40 = sp.time_to_distance(40 * 0.9144, 9.5, 1.1)
    assert sp.fit_splits(t10, t40) == pytest.approx((9.5, 1.1), abs=1e-4)


def test_drag_work_closed_form_matches_numeric():
    v_max, tau, T, rho, cd, A = 10.0, 1.3, 4.9, 1.18, 0.9, 0.55
    t = np.linspace(0, T, 200_001)
    numeric = np.trapezoid(0.5 * rho * cd * A * sp.speed_model(t, v_max, tau, 0.0) ** 3, t)
    assert sp.drag_work(v_max, tau, T, rho, cd, A) == pytest.approx(numeric, rel=1e-6)


def test_drag_work_constant_speed_limit():
    # tau -> 0: v ≈ v_max throughout, W = ½ ρ C_d A v³ T
    assert sp.drag_work(10.0, 1e-6, 5.0, 1.18, 0.9, 0.6) == pytest.approx(0.5 * 1.18 * 0.9 * 0.6 * 1000 * 5.0, rel=1e-5)


def test_body_surface_and_frontal_area():
    assert sp.body_surface_area(1.80, 80.0) == pytest.approx(1.996, abs=2e-3)  # Du Bois reference
    # HANDOFF scale: ~0.54 m² (85 kg DB) to ~0.72 m² (140 kg OL)
    assert sp.frontal_area(1.83, 85.0, 0.266) == pytest.approx(0.55, abs=0.02)
    assert sp.frontal_area(1.96, 140.0, 0.266) == pytest.approx(0.72, abs=0.03)


def test_crossing_time():
    t = np.array([0.0, 0.1, 0.2])
    assert sp.crossing_time(t, np.array([49.0, 49.6, 50.4]), 50.0) == pytest.approx(0.15)
    assert math.isnan(sp.crossing_time(t, np.array([1.0, 2.0, 3.0]), 50.0))


def _frames(reps):
    rows = []
    for rep_id, pid, kw in reps:
        t, s, x = _synthetic_rep(**kw)
        rows += [dict(player_id=pid, draft_year=2024, drill="forty", rep_id=rep_id, t_s=ti, x_yd=xi, s_mps=si)
                 for ti, xi, si in zip(t, x, s)]  # fmt: skip
    return pl.DataFrame(rows)


def test_fit_forty_reps_and_best_rep():
    fr = _frames([("a1", 1, dict(v_max=9.8)), ("a2", 1, dict(v_max=10.2)), ("b1", 2, dict(v_max=8.5, tau=1.2))])
    reps = sp.fit_forty_reps(fr)
    assert reps.height == 3
    a2 = reps.filter(pl.col("rep_id") == "a2").row(0, named=True)
    assert a2["v_max"] == pytest.approx(10.2, abs=1e-3)
    # t40_model = time from t0 to cover 40 yd from the first frame's line offset
    expected = sp.time_to_distance((50.0 - 10.6) * 0.9144, 10.2, 1.3)
    assert a2["t40_model"] == pytest.approx(expected, abs=2e-3)

    best = sp.best_reps(reps, max_rmse=0.4)
    assert dict(zip(best["player_id"], best["rep_id"])) == {1: "a2", 2: "b1"}
    assert dict(zip(best["player_id"], best["n_reps"])) == {1: 2, 2: 1}
    assert sp.best_reps(reps.with_columns(rmse=pl.lit(1.0)), max_rmse=0.4).height == 0


def test_forty_energy():
    best = pl.DataFrame({"player_id": [1], "v_max": [10.0], "tau": [1.3], "t40_model": [4.9]})
    prospects = pl.DataFrame({"player_id": [1], "mass_kg": [90.0], "ht_m": [1.85]})
    cfg = CFG["sprint"]
    r = sp.forty_energy(best, prospects, cfg).row(0, named=True)
    area = sp.frontal_area(1.85, 90.0, cfg["frontal_area_bsa_factor"])
    assert r["KE_40"] == pytest.approx(4500.0)
    assert r["W_drag_40"] == pytest.approx(sp.drag_work(10.0, 1.3, 4.9, cfg["air_density"], cfg["drag_coefficient"], area))
    assert r["E_40"] == pytest.approx(r["KE_40"] + r["W_drag_40"])
    assert 600 < r["W_drag_40"] < 1000  # HANDOFF scale check
