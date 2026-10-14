import polars as pl
import pytest

from eeg.ingest.load_combine import build_combine_si

RESULT_COLS = {
    "draft_year": pl.Int64, "nfl_id": pl.Int64, "combine_position": pl.String,
    "combine_height": pl.Float64, "combine_weight": pl.Int64, "hand_size": pl.Float64,
    "arm_length": pl.Float64, "wing_span": pl.Float64, "ten_yd_split": pl.Float64,
    "forty": pl.Float64, "vertical": pl.Float64, "broad_jump": pl.Int64,
    "three_cone": pl.Float64, "short_shuttle": pl.Float64, "bench_reps": pl.Int64,
    "ngs_athleticism_score": pl.Int64, "ngs_college_production_score": pl.Int64,
    "ngs_final_score": pl.Int64,
}  # fmt: skip


def _results(rows):
    base = {c: None for c in RESULT_COLS}
    return pl.DataFrame([{**base, **r} for r in rows], schema=RESULT_COLS)


def _players(ids):
    return pl.DataFrame({"nfl_id": ids, "display_name": [f"P{i}" for i in ids], "nfl_position": ["CB"] * len(ids)})


@pytest.fixture
def out():
    res = _results(
        [
            dict(draft_year=2024, nfl_id=1, combine_position="DB", combine_height=74.0, combine_weight=200,
                 arm_length=32.0, wing_span=78.0, hand_size=9.5, ten_yd_split=1.55, forty=4.40,
                 vertical=40.0, broad_jump=130, bench_reps=15, three_cone=6.9, short_shuttle=4.1),
            dict(draft_year=2023, nfl_id=2, combine_position="OL", combine_height=77.5, combine_weight=310,
                 arm_length=34.5, wing_span=82.0, hand_size=10.0, vertical=28.0),
        ]
    )  # fmt: skip
    return build_combine_si(res, _players([1, 2]))


def test_si_conversion(out):
    r = out.filter(pl.col("player_id") == 1).row(0, named=True)
    assert r["ht_m"] == pytest.approx(1.8796)
    assert r["mass_kg"] == pytest.approx(200 * 0.45359237)
    assert r["arm_m"] == pytest.approx(0.8128)
    assert r["vertical_m"] == pytest.approx(1.016)
    assert r["broad_m"] == pytest.approx(3.302)
    assert (r["forty_s"], r["ten_split_s"], r["bench_reps"]) == (4.40, 1.55, 15)
    assert (r["name"], r["pos"], r["pos_group"]) == ("P1", "CB", "DB")


def test_missingness(out):
    full = out.filter(pl.col("player_id") == 1).row(0, named=True)
    sparse = out.filter(pl.col("player_id") == 2).row(0, named=True)
    assert full["n_drills_raw"] == 6
    assert not any(full[f"{k}_missing"] for k in ["forty", "vertical", "broad", "bench", "cone", "shuttle"])
    assert sparse["n_drills_raw"] == 1
    assert sparse["forty_missing"] and not sparse["vertical_missing"]


def test_sorted_by_year(out):
    assert out["combine_year"].to_list() == [2023, 2024]


def test_rejects_unknown_group():
    res = _results([dict(draft_year=2024, nfl_id=1, combine_position="QB")])
    with pytest.raises(ValueError, match="combine_position"):
        build_combine_si(res, _players([1]))


def test_rejects_duplicate_ids():
    res = _results([dict(draft_year=2024, nfl_id=1, combine_position="WR")] * 2)
    with pytest.raises(ValueError, match="not unique"):
        build_combine_si(res, _players([1]))
