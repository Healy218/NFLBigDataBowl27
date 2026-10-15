import math

import numpy as np
import polars as pl
import pytest

from eeg import config
from eeg.ingest import units as u
from eeg.physics import cod

CFG = config.load()["cod"]
DT = 0.1


# --- synthetic reps --------------------------------------------------------------
# Each leg is (axis, sign, distance_m, duration_s). Every leg but the last is a full
# sine hump v = v_pk sin(π τ/T) (rest -> v_pk -> rest, a plant at each end); the last
# is a quarter wave rising to v_pk at the start line, then the run continues at v_pk.
# Both shapes cover d = 2 v_pk T / π, so v_pk = π d / (2T), and the exact
# Σ|Δ(v²)| = Σ 2 v_pk² (full legs) + v_pk² (last leg), with peaks on sample times.


def _rep(legs, drill="shuttle", idle_s=0.2, tail_s=0.3, rep_id="r1", player_id=1, x0_yd=95.0, y0_yd=15.0):
    t_edges = np.cumsum([idle_s] + [T for *_, T in legs])
    t = np.round(np.arange(0.0, t_edges[-1] + tail_s + 1e-9, DT), 10)
    pos, s, start = np.zeros((t.size, 2)), np.zeros(t.size), np.zeros(2)
    for k, (axis, sign, d, T) in enumerate(legs):
        vp, tau, last = math.pi * d / (2 * T), t - t_edges[k], k == len(legs) - 1
        w = 2 * T if last else T  # period scale: quarter wave on the last leg, half wave otherwise
        tt = np.clip(tau, 0.0, T)
        prog = vp * w / math.pi * (1 - np.cos(math.pi * tt / w))  # distance along the leg
        v = vp * np.sin(math.pi * tt / w)
        if last:  # carry on at v_pk through the line
            prog, v = prog + vp * np.maximum(tau - T, 0.0), np.where(tau > T, vp, v)
        active = (tau > 0) if last else (tau > 0) & (tau <= T + 1e-9)
        step = np.zeros(2)
        step[axis] = sign
        s = np.where(active, v, s)
        pos = np.where((tau > 0)[:, None], start + step * prog[:, None], pos)
        start = start + step * d
    x_m, y_m = x0_yd * u.yd_to_m + pos[:, 0], y0_yd * u.yd_to_m + pos[:, 1]
    return pl.DataFrame(
        dict(
            player_id=player_id, draft_year=2024, drill=drill, rep_id=rep_id, t_s=t,
            x_yd=x_m / u.yd_to_m, y_yd=y_m / u.yd_to_m, x_m=x_m, y_m=y_m, s_mps=s,
        )  # fmt: skip
    )


def _shuttle(sign=1.0, durations=(1.0, 1.6, 0.9), **kw):
    # 3.4 m out, 7.0 m across, 3.6 m back to (and through) the start line
    legs = [(0, sign, 3.4, durations[0]), (0, -sign, 7.0, durations[1]), (0, sign, 3.6, durations[2])]
    return _rep(legs, **kw), legs


def _expected_sum_dv2(legs):
    vps = [math.pi * d / (2 * T) for *_, d, T in legs]
    return sum(2 * v**2 for v in vps[:-1]) + vps[-1] ** 2


def _segment(df):
    return cod.segment_rep(df["drill"][0], df["t_s"], df["x_yd"], df["y_yd"], df["s_mps"], CFG)


# --- primitives --------------------------------------------------------------------


def test_ke_variation_counts_braking_as_positive():
    assert cod.ke_variation(np.array([0.0, 3.0, 0.0, 4.0])) == pytest.approx(9 + 9 + 16)


def test_crossing_after():
    t = np.arange(6) * 0.1
    v = np.array([0.0, -1.0, -2.0, -1.0, 1.0, 2.0])
    tc, i = cod.crossing_after(t, v, 2, rising=True)
    assert (tc, i) == (pytest.approx(0.35), 3)
    assert math.isnan(cod.crossing_after(t, v, 4, rising=True)[0])


def test_onset_time_interpolates():
    assert cod.onset_time(np.array([0.0, 0.1, 0.2]), np.array([0.2, 0.6, 1.4]), 1.0) == pytest.approx(0.15)


# --- segmentation --------------------------------------------------------------------


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_shuttle_valid_either_direction(sign):
    df, legs = _shuttle(sign)
    seg = _segment(df)
    assert seg["valid"] and seg["finish_rule"] == "crossing"
    assert seg["t_fin"] == pytest.approx(0.2 + 1.0 + 1.6 + 0.9, abs=1e-6)  # end of the last leg

    rep = cod.segment_cod_reps(df, CFG).row(0, named=True)
    assert rep["valid"] and rep["n_plants"] == 2
    assert rep["sum_dv2"] == pytest.approx(_expected_sum_dv2(legs), rel=1e-6)
    assert rep["path_m"] == pytest.approx(3.4 + 7.0 + 3.6, abs=0.05)


def test_shuttle_window_ending_short_of_line():
    df, _ = _shuttle()
    # window cut while the sensor is still ~0.4 yd short of the line -> window end is the finish
    cut = df.filter(pl.col("t_s") <= 3.6 + 1e-9)
    seg = _segment(cut)
    assert seg["valid"] and seg["finish_rule"] == "window_end" and seg["t_fin"] == pytest.approx(3.6)
    # cut ~2 yd short -> no finish
    assert _segment(df.filter(pl.col("t_s") <= 3.2 + 1e-9))["reason"] == "no_finish"


def test_shuttle_without_far_leg_is_rejected():
    df = _rep([(0, 1.0, 3.4, 1.0), (0, -1.0, 3.4, 1.0)])
    assert _segment(df)["reason"] == "no_far_leg"


def test_jogged_out_rep_is_rejected():
    legs = [(0, 1.0, 3.4, 1.0), (0, -1.0, 7.0, 1.6), (0, 1.0, 3.6, 3.0)]  # last leg at ~1.9 m/s
    rep = cod.segment_cod_reps(_rep(legs), CFG).row(0, named=True)
    assert not rep["valid"] and rep["reason"] == "no_finish_sprint"


def _cone(**kw):
    # out / back / out along +x (plants at both lines), down to cone 3 and back up,
    # then sprint back along -x through the start line
    legs = [
        (0, 1.0, 3.0, 1.0), (0, -1.0, 3.0, 1.0), (0, 1.0, 4.5, 1.2),
        (1, -1.0, 4.5, 1.2), (1, 1.0, 4.5, 1.2), (0, -1.0, 4.5, 1.0),
    ]  # fmt: skip
    return _rep(legs, drill="cone", **kw), legs


def test_cone_valid():
    df, legs = _cone()
    rep = cod.segment_cod_reps(df, CFG).row(0, named=True)
    assert rep["valid"] and rep["finish_rule"] == "crossing"
    assert rep["t_fin"] == pytest.approx(0.2 + 1.0 + 1.0 + 1.2 + 1.2 + 1.2 + 1.0, abs=1e-6)
    assert rep["sum_dv2"] == pytest.approx(_expected_sum_dv2(legs), rel=1e-6)


def test_cone_window_opening_late_is_rejected():
    df, _ = _cone()
    late = df.filter(pl.col("t_s") >= 5.0 - 1e-9).with_columns(pl.col("t_s") - 5.0)  # opens on the way back up
    rep = cod.segment_cod_reps(late, CFG).row(0, named=True)
    assert not rep["valid"] and rep["reason"] in {"truncated_start", "no_cone_loop"}


def test_count_plants():
    s = np.array([0.5, 3.0, 0.2, 4.0, 1.0, 5.0, 2.5, 6.0])  # minima 0.2, 1.0 (plants), 2.5 (not)
    assert cod.count_plants(s) == 2


# --- best rep and energy -------------------------------------------------------------


def test_best_rep_and_energy():
    fast, legs = _shuttle(durations=(0.8, 1.4, 0.8), rep_id="fast")
    slow, _ = _shuttle(rep_id="slow")
    jog = _rep([(0, 1.0, 3.4, 1.0), (0, -1.0, 7.0, 1.6), (0, 1.0, 3.6, 3.0)], rep_id="jog")
    reps = cod.segment_cod_reps(pl.concat([slow, fast, jog]), CFG)
    best = cod.best_reps(reps)
    assert best.height == 1 and best["rep_id"][0] == "fast" and best["n_reps"][0] == 3

    w = cod.cod_energy(best, pl.DataFrame({"player_id": [1], "mass_kg": [100.0]}))["W_cod"][0]
    assert w == pytest.approx(0.5 * 100.0 * _expected_sum_dv2(legs), rel=1e-6)
