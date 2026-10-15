import math

import polars as pl
import pytest

from eeg import config
from eeg.physics import sprint, tee
from eeg.physics.bench import work_bench
from eeg.physics.broad import energy_broad
from eeg.physics.vertical import energy_vertical, sayers_peak_power

RAW = config.load_raw()
CFG = config.values(RAW)

# 6 prospects over 2 combine years; player 6 has only two channels (vertical + broad)
PROSPECTS = pl.DataFrame(
    {
        "player_id": [1, 2, 3, 4, 5, 6],
        "combine_year": [2023, 2023, 2023, 2024, 2024, 2024],
        "pos_group": ["WR", "OL", "DB", "DL", "TE", "WR"],
        "mass_kg": [90.0, 140.0, 88.0, 130.0, 115.0, 92.0],
        "ht_m": [1.85, 1.95, 1.80, 1.93, 1.96, 1.83],
        "arm_m": [0.82, 0.88, 0.80, 0.86, 0.84, 0.81],
        "vertical_m": [0.95, 0.70, 0.98, 0.78, 0.85, 0.90],
        "broad_m": [3.2, 2.6, 3.3, 2.8, 3.0, 3.1],
        "bench_reps": [14.0, 30.0, None, 28.0, 20.0, None],
        "shuttle_s": [4.2, 4.7, None, 4.6, 4.4, None],
        "cone_s": [6.9, 7.6, None, 7.5, None, None],
    }
)
FORTY_REPS = pl.DataFrame(
    {
        "rep_id": ["1a", "1b", "2a", "3a", "4a", "5a"],
        "player_id": [1, 1, 2, 3, 4, 5],
        "v_max": [10.4, 10.2, 8.8, 10.6, 9.1, 9.6],
        "tau": [1.3, 1.3, 1.2, 1.3, 1.25, 1.3],
        "rmse": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "t40_model": [4.55, 4.60, 5.30, 4.50, 5.05, 4.80],
    }
)
COD_REPS = pl.DataFrame(
    {
        "rep_id": ["1s", "2s", "4s", "5s", "3s", "1c", "2c", "4c"],
        "player_id": [1, 2, 4, 5, 3, 1, 2, 4],
        "drill": ["shuttle"] * 5 + ["cone"] * 3,
        "valid": [True] * 8,
        "t_trk": [4.4, 4.9, 4.8, 4.6, 4.5, 7.0, 7.8, 7.6],
        "sum_dv2": [110.0, 85.0, 90.0, 95.0, 112.0, 100.0, 78.0, 80.0],
    }
)


@pytest.fixture(scope="module")
def out():
    return tee.build_tee(PROSPECTS, FORTY_REPS, COD_REPS, CFG)


def _row(df, pid):
    return df.filter(pl.col("player_id") == pid).row(0, named=True)


def test_channels_match_physics_modules(out):
    r = _row(out, 2)
    assert r["E_vert"] == pytest.approx(energy_vertical(140.0, 0.70))
    assert r["E_broad"] == pytest.approx(energy_broad(140.0, 2.6, CFG["broad"]["theta_deg"]))
    assert r["W_bench"] == pytest.approx(work_bench(30.0, 0.88, CFG["bench"]["rom_arm_fraction"]))
    v = CFG["vertical"]
    assert r["P_sayers"] == pytest.approx(sayers_peak_power(140.0, 0.70, v["sayers_a"], v["sayers_b"], v["sayers_c"]))
    e40 = sprint.forty_energy(FORTY_REPS.filter(pl.col("player_id") == 2), PROSPECTS, CFG["sprint"])["E_40"][0]
    assert r["E_40"] == pytest.approx(e40)
    assert r["W_cod_shuttle"] == pytest.approx(0.5 * 140.0 * 85.0)
    assert r["W_cod_cone"] == pytest.approx(0.5 * 140.0 * 78.0)


def test_best_forty_rep_is_used(out):
    e40 = sprint.forty_energy(FORTY_REPS.filter(pl.col("rep_id") == "1a"), PROSPECTS, CFG["sprint"])["E_40"][0]
    assert _row(out, 1)["E_40"] == pytest.approx(e40)


def test_per_kg_and_flags(out):
    r = _row(out, 5)
    for c in tee.CHANNELS:
        if r[c] is not None:
            assert r[f"{c}_per_kg"] == pytest.approx(r[c] / 115.0)
    assert r["W_cod_cone_missing"] and not r["W_cod_shuttle_missing"]
    assert _row(out, 2)["sayers_extrapolated"] and not _row(out, 1)["sayers_extrapolated"]
    # player 3 has a tracked shuttle but no official time
    assert _row(out, 3)["W_cod_shuttle_unofficial"] and not _row(out, 1)["W_cod_shuttle_unofficial"]


def test_zscores_within_combine_year(out):
    for _, g in out.group_by("combine_year"):
        z = g["z_E_vert"]
        assert z.mean() == pytest.approx(0.0, abs=1e-12)
        assert z.std() == pytest.approx(1.0)
    # bench nulls stay null
    assert _row(out, 3)["z_W_bench"] is None


def test_composite_needs_min_drills(out):
    assert CFG["pipeline"]["composite_min_drills"] == 3
    r6 = _row(out, 6)
    assert r6["n_drills"] == 2 and r6["TEE_z_abs"] is None and r6["TEE_z_rel"] is None
    r1 = _row(out, 1)
    assert r1["n_drills"] == 6
    assert r1["TEE_z_abs"] == pytest.approx(sum(r1[f"z_{c}"] for c in tee.CHANNELS) / 6)
    r5 = _row(out, 5)  # 5 channels: no cone
    zs = [r5[f"z_{c}_per_kg"] for c in tee.CHANNELS if r5[f"z_{c}_per_kg"] is not None]
    assert len(zs) == 5 and r5["TEE_z_rel"] == pytest.approx(sum(zs) / 5)


def test_override_changes_only_its_channel(out):
    alt = tee.build_tee(PROSPECTS, FORTY_REPS, COD_REPS, config.values(config.with_override(RAW, "broad.theta_deg", 30.0)))
    a, b = _row(out, 1), _row(alt, 1)
    assert b["E_broad"] / a["E_broad"] == pytest.approx(1 / math.sin(math.radians(60)))
    assert all(a[c] == b[c] for c in ("E_vert", "E_40", "W_cod_shuttle", "W_bench"))
