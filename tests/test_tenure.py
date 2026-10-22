import numpy as np
import polars as pl
import pytest

from eeg.survival import tenure as T


def _prospects():
    """1: 2023 debut, still playing (censored at 3). 2: 2023 debut, last play 2024 (exit at 2). 3: 2024 debut, exit
    at 1. 4: 2025 debut (no later season to miss). 5: never played."""
    return pl.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "seasons_active": [3, 2, 1, 1, 0],
            "first_active": [2023, 2023, 2024, 2025, None],
            "last_active": [2025, 2024, 2024, 2025, None],
            "exited": [False, True, True, False, False],
        }
    )


def test_at_risk_excludes_last_season_debuts_and_non_players():
    d = T.at_risk(_prospects(), 2025).sort("player_id")
    assert d["player_id"].to_list() == [1, 2, 3]
    assert d["duration"].to_list() == [3, 2, 1] and d["event"].to_list() == [0, 1, 1]


def test_km_table_matches_hand_count():
    d = T.at_risk(_prospects(), 2025).with_columns(pl.lit("all").alias("g"))
    row = T.km_table(d, "g").row(0, named=True)
    assert row["stay_after_1"] == pytest.approx(2 / 3) and row["stay_after_2"] == pytest.approx(2 / 3 * 1 / 2)


def test_oe_test_detects_a_tail_that_never_exits():
    rng = np.random.default_rng(0)
    n = 200
    tail = np.zeros(n, bool)
    tail[: n // 5] = True
    event = np.where(tail, 0, rng.random(n) < 0.4).astype(int)
    d = pl.DataFrame(
        {"cell": ["a|2023"] * n, "m": rng.normal(size=n), "tail": tail, "event": event, "expected": np.full(n, 0.32)}
    )
    r = T.oe_test(d, "m", n_perm=500)
    assert r["observed"] == 0 and r["oe"] == 0 and r["p_perm"] < 0.01
