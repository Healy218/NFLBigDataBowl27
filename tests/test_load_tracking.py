from datetime import datetime, timedelta

import polars as pl
import pytest

from eeg.ingest.load_tracking import build_timed_drills


def _raw():
    t0 = datetime(2025, 3, 1, 18, 0, 0)
    rows = []
    for drill, attempt, n in [("FORTY_YARD_DASH", 1, 3), ("SHORT_SHUTTLE", 2, 2), ("SKILL_DRILLS_WR", 1, 2)]:
        for i in range(n):
            rows.append(dict(draft_year=2025, event_id="ev", nfl_id=7, entity_type="PLAYER", time=t0 + timedelta(seconds=100 * attempt + 0.1 * i),
                             drill_type=drill, drill_name=drill, attempt=attempt, x=10.0 + i, y=20.0, s=10.0, a=1.0, dis=0.5, dir=90.0))  # fmt: skip
    rows.append({**rows[0], "entity_type": "BALL"})
    return pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=1)


def test_filters_and_keys():
    out = build_timed_drills(_raw())
    assert set(out["drill"]) == {"forty", "shuttle"}  # skill drills and BALL dropped
    assert out.height == 5
    assert set(out["rep_id"]) == {"7_ev_1", "7_ev_2"}
    assert "dis" not in out.columns


def test_time_and_units():
    out = build_timed_drills(_raw()).filter(pl.col("drill") == "forty")
    assert out["t_s"].to_list() == pytest.approx([0.0, 0.1, 0.2])
    assert out["x_yd"].to_list() == [10.0, 11.0, 12.0]
    assert out["x_m"][0] == pytest.approx(9.144)
    assert out["s_mps"][0] == pytest.approx(9.144)
    assert out["a_mps2"][0] == pytest.approx(0.9144)
