import numpy as np
import polars as pl
import pytest

from eeg.ngs import channels as C

G = dict(sustain_frames=3, burst_window_s=1.5, burst_start_speed_mps=1.0, accel_window_frames=5, frame_dt_s=0.1)


def _track(play_id, s, a, t_snap0=-0.5, glitch=False, player_id=7, game_id=1, season=2024):
    n = len(s)
    return pl.DataFrame(
        {
            "player_id": player_id, "season": season, "season_type": "REG", "week": 1, "game_id": game_id,
            "play_id": play_id, "t_snap_s": np.round(t_snap0 + 0.1 * np.arange(n), 6),
            "s_mps": np.asarray(s, float), "a_s_mps2": np.asarray(a, float), "glitch_track": glitch,
        }
    )  # fmt: skip


NO_DRAG = pl.DataFrame({"player_id": [7, 8], "drag_k": [0.0, 0.0]})


def test_play_channels_burst_power_and_glitch():
    t = 0.1 * np.arange(25)
    after = np.clip(t - 0.5, 0, None)  # snap at frame 5, then 4 m/s^2 from rest
    s = 4.0 * after
    a = np.where(t >= 0.5, 4.0, 0.0)
    moving = _track(2, 3.0 + s, a)  # already at 3 m/s at the snap
    spike = _track(3, np.r_[[1.0] * 10, 9.0, [1.0] * 4], np.zeros(15))
    out = C.play_channels(
        pl.concat([_track(1, s, a), moving, spike, _track(4, s * 2, a, glitch=True)]), NO_DRAG, G
    ).sort("play_id")
    assert out["play_id"].to_list() == [1, 2, 3]  # glitch track dropped
    p1, p2, p3 = out.rows(named=True)
    assert p1["burst_eligible"] and p1["burst_a"] == pytest.approx(4.0) and p1["a_peak"] == pytest.approx(4.0)
    assert p1["s_peak"] == pytest.approx(4.0 * (t[-3] - 0.5))  # held 3 frames: the third-last frame
    assert p1["ke_peak"] == pytest.approx(0.5 * p1["s_peak"] ** 2)
    assert p1["p_peak"] == pytest.approx(4.0 * 4.0 * (t[-3] - 0.5))  # a v without drag
    assert not p2["burst_eligible"] and p2["burst_a"] is None
    assert p3["s_peak"] == pytest.approx(1.0)  # one-frame spike ignored


def test_drag_adds_k_v_cubed():
    s = np.full(10, 8.0)
    drag = pl.DataFrame({"player_id": [7], "drag_k": [0.004]})
    out = C.play_channels(_track(1, s, np.zeros(10)), drag, G)
    assert out["p_peak"][0] == pytest.approx(0.004 * 8.0**3)


def test_drag_coef_formula():
    prospects = pl.DataFrame({"player_id": [1], "ht_m": [1.85], "mass_kg": [100.0]})
    sprint = dict(frontal_area_bsa_factor=0.266, air_density=1.18, drag_coefficient=0.9)
    bsa = 0.007184 * 100.0**0.425 * 185.0**0.725
    assert C.drag_coef(prospects, sprint)["drag_k"][0] == pytest.approx(0.5 * 1.18 * 0.9 * 0.266 * bsa / 100.0)


def _plays(rows):
    return pl.DataFrame(
        [dict(zip(["player_id", "season", "game_id", "play_id", "s_peak", "burst_a"], r)) for r in rows]
    ).with_columns(
        season_type=pl.lit("REG"), week=1, burst_eligible=pl.col("burst_a").is_not_null(),
        ke_peak=0.5 * pl.col("s_peak") ** 2, a_peak=pl.col("burst_a"), p_peak=pl.col("s_peak"),
    )  # fmt: skip


def test_game_and_season_statistics():
    plays = _plays([
        (7, 2024, 1, 1, 8.0, 5.0), (7, 2024, 1, 2, 9.0, None),
        (7, 2024, 2, 1, 7.0, 4.0), (7, 2024, 3, 1, 6.0, 6.0), (7, 2024, 4, 1, 5.0, None),
    ])  # fmt: skip
    games = C.game_channels(plays)
    assert games["s_peak"].to_list() == [9.0, 7.0, 6.0, 5.0] and games["plays"].to_list() == [2, 1, 1, 1]
    assert games["burst_plays"].to_list() == [1, 1, 1, 0]
    season = C.season_channels(games).row(0, named=True)
    assert season["games"] == 4 and season["plays"] == 5 and season["burst_games"] == 3
    assert season["s_peak_max"] == 9.0 and season["s_peak_game_med"] == 6.5
    assert season["s_peak_top3"] == pytest.approx((9 + 7 + 6) / 3)
    assert season["burst_a_top3"] == pytest.approx(5.0)  # only 3 games with a burst
    strict = C.season_channels(games, min_game_plays=2, min_plays=2, min_games=1).row(0, named=True)
    assert strict["games"] == 1 and strict["s_peak_max"] == 9.0 and strict["qualified"]
    assert not C.season_channels(games, min_games=5)["qualified"][0]


def test_lab_channels_same_estimators():
    t = 0.1 * np.arange(40)
    s = 9.0 * (1 - np.exp(-np.clip(t - 0.3, 0, None) / 1.0))
    drills = pl.DataFrame({"player_id": 7, "drill": "forty", "rep_id": "r", "t_s": t, "s_mps": s})
    best = pl.DataFrame({"rep_id": ["r"], "v_max": [9.0], "tau": [1.0], "t_finish": [t[-10]]})
    lab = C.lab_channels(drills, best, NO_DRAG, G).row(0, named=True)
    assert lab["lab_s_peak"] == pytest.approx(s[-12])  # sustained, up to the finish
    assert lab["lab_a0"] == 9.0 and lab["lab_p_model"] == pytest.approx(81 / 4)
    assert 0 < lab["lab_a_peak"] < 9.0 and 0 < lab["lab_p_peak"] < 81 / 4


def test_expected_top_matches_brute_force_over_subsets():
    from itertools import combinations

    v = np.array([9.0, 3.0, 7.0, 7.0, 1.0, 5.0, 8.0])
    for k in (3, 4, 7):
        brute = np.mean([np.sort(c)[-3:].mean() for c in combinations(v, k)])
        assert C.expected_top(v, k) == pytest.approx(brute)
    assert np.isnan(C.expected_top(v, 8))


def test_play_statistic_does_not_grow_with_exposure():
    """Two players with the same play distribution, one with 4x the plays: game-based top3 favours the busier
    player, top3_of{K} does not."""
    rng = np.random.default_rng(1)
    rows = []
    for pid, n_games in [(1, 4), (2, 16)]:
        for g in range(n_games):
            for p in range(20):
                rows.append((pid, 2024, g, p, float(rng.normal(8.0, 0.5)), None))
    plays = _plays(rows)
    games = C.game_channels(plays)
    season = C.season_channels(games).join(C.play_season_stats(plays, games, 20), on=["player_id", "season"])
    s = {r["player_id"]: r for r in season.iter_rows(named=True)}
    assert s[2]["s_peak_top3"] - s[1]["s_peak_top3"] > 0.2
    assert abs(s[2]["s_peak_top3_of20"] - s[1]["s_peak_top3_of20"]) < 0.15
    assert s[1]["burst_a_top3_of20"] is None  # no burst plays
