import math

import numpy as np
import polars as pl
import pytest

from eeg import config
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
