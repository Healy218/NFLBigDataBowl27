from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from eeg.ingest.load_games import build_game_frames, summarize

CFG = {
    "games": dict(
        season_types=["REG", "POST"],
        frame_dt_s=0.1,
        max_speed_mps=11.0,
        max_accel_mps2=12.0,
        max_speed_mismatch_mps=1.0,
    )
}
GAMES = pl.DataFrame(
    {"game_id": [1, 2, 3], "season": [2024] * 3, "season_type": ["PRE", "REG", "POST"], "week": [1, 5, 19]}
)
T0 = datetime(2024, 10, 6, 13, 0, 0)


def _track(game_id, play_id, nfl_id, x, s=None, a=None, snap_at=2, dt=None):
    """One track along x (yd) at 10 Hz; s defaults to the true speed, a to 0."""
    n = len(x)
    t = np.cumsum([0.0, *(dt or [0.1] * (n - 1))])
    s = np.gradient(x, t) if s is None else s
    return pl.DataFrame(
        {
            "game_id": game_id, "play_id": play_id, "nfl_id": nfl_id,
            "time": [T0 + timedelta(seconds=float(ti)) for ti in t],
            "x": np.asarray(x, float), "y": 20.0, "s": np.asarray(s, float),
            "a": np.zeros(n) if a is None else np.asarray(a, float),
            "dis": 0.0, "o": 0.0, "dir": 90.0,
            "event": pl.Series([("ball_snap" if i == snap_at else None) for i in range(n)], dtype=pl.String),
        }
    )  # fmt: skip


def _build(*tracks):
    raw = pl.concat(tracks).sample(fraction=1.0, shuffle=True, seed=1)
    return build_game_frames(raw, GAMES, CFG)


def test_season_types_and_columns():
    x = 10 + 0.5 * np.arange(8)
    out = _build(_track(1, 1, 7, x), _track(2, 1, 7, x), _track(3, 1, 7, x))
    assert set(out["season_type"]) == {"REG", "POST"} and set(out["week"]) == {5, 19}
    assert "dis" not in out.columns and "o" not in out.columns
    assert out.filter(pl.col("game_id") == 2)["frame_id"].to_list() == list(range(8))


def test_constant_speed_units_and_snap_time():
    x = 10 + 0.5 * np.arange(8)  # 5 yd/s
    out = _build(_track(2, 1, 7, x))
    assert out["v_xy_mps"].to_list() == pytest.approx([5 * 0.9144] * 8)
    assert out["s_mps"].to_list() == pytest.approx([5 * 0.9144] * 8)
    assert out["a_t_mps2"].to_list() == pytest.approx([0.0] * 8, abs=1e-9)
    assert out["x_m"][0] == pytest.approx(9.144)
    assert out["t_snap_s"].to_list() == pytest.approx([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert not out["glitch_track"].any()


def test_acceleration_from_positions():
    t = 0.1 * np.arange(10)
    out = _build(_track(2, 1, 7, 10 + 0.5 * 4.0 * t**2))  # 4 yd/s^2 from rest
    inner = out["a_t_mps2"].to_numpy()[2:-2]
    assert inner == pytest.approx(4 * 0.9144, rel=1e-6)
    assert out["v_xy_mps"].to_numpy()[1:-1] == pytest.approx(4 * t[1:-1] * 0.9144, rel=1e-6)


def test_snap_time_null_without_snap_and_first_snap_used():
    x = 10 + 0.5 * np.arange(6)
    two = _track(2, 2, 7, x).with_columns(
        pl.when(pl.int_range(pl.len()).is_in([1, 4])).then(pl.lit("ball_snap")).alias("event")
    )
    out = _build(_track(2, 1, 7, x, snap_at=-1), two)
    assert out.filter(pl.col("play_id") == 1)["t_snap_s"].is_null().all()
    assert out.filter(pl.col("play_id") == 2)["t_snap_s"][1] == pytest.approx(0.0)


def test_glitch_flags():
    x = 10 + 0.5 * np.arange(8)
    jump = x.copy()
    jump[3:] += 3.0  # 30 yd/s step into frame 3
    fast = _track(2, 2, 8, x, s=np.r_[[5.0] * 5, 13.0, 5.0, 5.0])  # 13 yd/s reported, positions say 5
    accel = _track(2, 3, 8, x, a=np.r_[[0.0] * 6, 15.0, 0.0])
    gap = _track(2, 4, 8, 10 + 0.5 * np.arange(5), dt=[0.1, 0.2, 0.1, 0.1])
    out = _build(_track(2, 1, 7, jump), fast, accel, gap, _track(3, 1, 9, x))
    flags = {p: out.filter((pl.col("game_id") == 2) & (pl.col("play_id") == p)) for p in (1, 2, 3, 4)}
    assert flags[1]["flag_jump"].arg_true().to_list() == [3]
    assert flags[2]["flag_speed"].arg_true().to_list() == [5] and flags[2]["flag_mismatch"][5]
    assert flags[3]["flag_accel"].arg_true().to_list() == [6]
    assert flags[4]["flag_gap"].arg_true().to_list() == [2]
    assert all(f["glitch_track"].all() for f in flags.values())
    assert not out.filter(pl.col("game_id") == 3)["glitch_track"].any()

    summ = summarize(out)
    reg = summ.filter(pl.col("season_type") == "REG").row(0, named=True)
    assert reg["tracks"] == 4 and reg["glitch_tracks"] == 4 and reg["flag_jump"] == 1
    assert summ.filter(pl.col("season_type") == "POST")["glitch_tracks"][0] == 0
