import numpy as np
import polars as pl
import pytest

from eeg.outcomes import robustness as R


def test_knn_expectation_leaves_the_prospect_out():
    d = pl.DataFrame({"cell": ["a"] * 5 + ["b"] * 3, "pick": [1, 2, 3, 50, 51, 1, 2, 3], "y": [5, 6, 7, 0, 1, 9, 9, 3]})
    e = R.knn_expectation(d, "y", k=2)
    a = e.filter(pl.col("cell") == "a")["y_exp"].to_list()
    assert a[0] == pytest.approx((6 + 7) / 2)  # nearest picks 2 and 3, not itself
    assert a[3] == pytest.approx((1 + 7) / 2)  # 51 and 3
    b = e.filter(pl.col("cell") == "b")
    assert b["y_exp"][2] == pytest.approx(9.0) and b["y_resid"][2] == pytest.approx(-6.0)  # cells don't mix


def test_results_composite_needs_two_of_jumps_and_bench(tmp_path):
    tee = pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "z_E_40": [1.0, 0.0, None],
            "z_E_vert": [1.0, 2.0, None],
            "z_E_broad": [3.0, None, 1.0],
            "z_W_bench": [None, None, None],
            "z_W_cod_shuttle": [None] * 3,
            "z_W_cod_cone": [None] * 3,
        }
    )
    tee.write_parquet(tmp_path / "tee.parquet")
    out = R.tee_channels(tmp_path).sort("player_id")
    assert out["z_results"][0] == pytest.approx(2.0)
    assert out["z_results"][1] is None and out["z_results"][2] is None
    assert not np.isnan(out["z_E_40"][0])
