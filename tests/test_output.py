import numpy as np
import polars as pl
import pytest

from eeg.outcomes import output as O


def test_draft_features_undrafted_and_bands():
    out = pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "draft_year": [2023, 2023, 2024],
            "draft_overall_pick": [5, 150, None],
            "undrafted": [False, False, True],
            "round_band": ["1", "4-5", "UDFA"],
        }
    )
    f = O.draft_features(out).sort("player_id")
    assert f["pick"].to_list() == [5.0, 150.0, float(O.UDFA_PICK)]
    assert f["udfa"].to_list() == [0.0, 0.0, 1.0]
    assert f["band_4-5"].to_list() == [0.0, 1.0, 0.0] and f["band_2-3"].sum() == 0


def test_next_season_pairs_first_eta_season_with_the_following_one():
    eta = pl.DataFrame(
        {
            "player_id": [1, 1, 2, 3],
            "season": [2023, 2024, 2024, 2025],
            "eta_ke": [0.5, 0.6, 0.4, 0.7],
            "lab_ke": [30.0, 30.0, 25.0, 28.0],
            "field_ke": [15.0, 18.0, 10.0, 19.6],
            "plays": [200, 300, 150, 400],
        }
    )
    career = pl.DataFrame({"player_id": [1, 2, 3], "first_season": [2023, 2024, 2025]})
    seasons = pl.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2, 3],
            "season": [2023, 2024, 2025, 2024, 2025, 2025],
            "plays": [200, 300, 0, 150, 0, 400],
        }
    )
    ns = O.next_season(eta, career, seasons).sort("player_id")
    assert ns["player_id"].to_list() == [1, 2]  # prospect 3's next season is beyond the data
    assert ns["log_eta"][0] == pytest.approx(np.log(0.5))  # first season, not the later one
    assert ns["pt_next"].to_list() == pytest.approx([np.log1p(300), 0.0])  # zeros kept
    assert ns["log_plays"][1] == pytest.approx(np.log(150))


def test_career_production_uses_each_groups_measure_on_seasons_with_eta():
    prod = pl.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "season": [2023, 2024, 2023, 2024],
            "group": ["WR", "WR", "DL", "DL"],
            "plays": [100, 400, 200, 200],
            "yprr": [1.0, 3.0, None, None],
            "pressure_rate": [None, None, 0.1, 0.3],
            "get_off": [None, None, 0.8, 0.9],
            "qualified": [True, True, True, False],
        }
    )
    eta = pl.DataFrame(
        {"player_id": [1, 1, 2, 2], "season": [2023, 2024, 2023, 2024], "eta_ke": [0.5] * 4, "lab_ke": [30.0] * 4}
    )
    cp = O.career_production(prod, eta, {"WR": "yprr", "DL": "pressure_rate"}).sort("player_id")
    assert cp["production"].to_list() == pytest.approx([2.0, 0.1])  # DL 2024 is not qualified
    assert cp["seasons"].to_list() == [2, 1]
    assert cp["log_get_off"][1] == pytest.approx(np.log(0.8))


def test_varying_drops_predictors_constant_within_cells():
    d = pl.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 5.0], "c": [0.0, 0.0, 1.0, 1.0]})
    assert O._varying(d, ["x", "c"], "g") == ["x"]
