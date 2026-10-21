import numpy as np
import polars as pl
import pytest

from eeg.ngs import validate as V


def _frames(tracks):
    """tracks: list of (player_id, season, play_id, speeds, glitch)."""
    rows = []
    for pid, season, play, speeds, glitch in tracks:
        for i, s in enumerate(speeds):
            rows.append(dict(player_id=pid, season=season, game_id=1, play_id=play, frame_id=i, s_mps=float(s),
                             a_s_mps2=0.0, glitch_track=glitch))  # fmt: skip
    return pl.DataFrame(rows)


def test_field_peaks_sustain_and_glitch():
    f = _frames([
        (7, 2024, 1, [1, 2, 9, 2, 1], False),        # one-frame spike to 9
        (7, 2024, 2, [5, 6, 6, 6, 5], False),        # held at 6
        (7, 2024, 3, [12, 12, 12, 12, 12], True),    # glitch track
    ])  # fmt: skip
    p = V.field_peaks(f).row(0, named=True)
    assert p["plays"] == 2 and p["s_peak"] == 9 and p["s_peak_sust"] == 6
    assert V.field_peaks(f, keep_glitch=True)["s_peak"][0] == 12


def test_compare_ratio_and_share():
    df = pl.DataFrame({"f": [9.0, 8.0, 11.0, 7.0], "l": [10.0, 9.0, 10.5, 8.0]})
    c = V.compare(df, {"f": "l"}).row(0, named=True)
    assert c["n"] == 4 and c["field_above_lab"] == pytest.approx(0.25)
    assert c["ratio_median"] == pytest.approx(np.median([0.9, 8 / 9, 11 / 10.5, 7 / 8]))


def test_season_retest_pairs_consecutive_seasons():
    field = pl.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 3, 3],
        "season": [2023, 2024, 2025, 2023, 2025, 2024, 2025],
        "plays": [200, 200, 200, 200, 200, 50, 200],
        "x": [1.0, 2.0, 3.0, 5.0, 6.0, 0.0, 9.0],
    })  # fmt: skip
    r = V.season_retest(field, ["x"]).row(0, named=True)
    assert r["pairs"] == 2  # 2023-24 and 2024-25 for player 1; player 2 skips a season; player 3 under min plays


def test_lab_peaks_stop_at_finish():
    t = 0.1 * np.arange(30)
    s = 9.0 * (1 - np.exp(-t / 1.0))
    s[-5:] = 10.5  # after the finish: ignored
    drills = pl.DataFrame({"player_id": 7, "drill": "forty", "rep_id": "r1", "t_s": t, "s_mps": s})
    best = pl.DataFrame({"rep_id": ["r1"], "v_max": [9.0], "tau": [1.0], "t_finish": [t[-6]]})
    lab = V.lab_peaks(drills, best, window=5).row(0, named=True)
    assert lab["lab_v_max"] == 9.0 and lab["lab_a0"] == pytest.approx(9.0)
    assert lab["lab_s_peak"] == pytest.approx(s[-6]) and lab["lab_s_peak"] < 9.0
    assert 0 < lab["lab_a_peak"] < 9.0  # smoothed, sampled after t = 0


def test_band_labels():
    df = pl.DataFrame({"v": [0.5, 2.0, 7.9, 20.0]})
    out = df.select(V._band("v", [2, 4, 8], ["a", "b", "c", "d"]).alias("band"))["band"].to_list()
    assert out == ["a", "b", "c", "d"]
