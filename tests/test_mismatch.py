import numpy as np
import polars as pl
import pytest

from eeg.outcomes import mismatch as M


def _frame(n_per=30, seed=0, effect=0.0):
    """Two groups × two classes; y falls with pick; the top 20% of `m` within a cell get `effect` extra."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in ("WR", "OL"):
        for yr in (2023, 2024):
            pick = rng.uniform(1, 257, n_per)
            m = rng.normal(size=n_per)
            top = m >= np.quantile(m, 0.8)
            y = 6 - 0.015 * pick + 0.5 * (yr == 2024) + rng.normal(0, 0.5, n_per) + effect * top
            rows += [dict(group=g, draft_year=yr, pick=p, udfa=0.0, m=mm, y=yy) for p, mm, yy in zip(pick, m, y)]
    d = pl.DataFrame(rows)
    return d.with_columns(pl.concat_str("group", pl.col("draft_year").cast(pl.String), separator="|").alias("cell"))


def test_slot_expectation_is_leave_one_out_with_class_intercepts():
    d = M.slot_expectation(_frame(), "y", ["pick", "udfa"]).sort("group", "draft_year", "pick")
    x = d.filter(pl.col("group") == "WR")
    i = 7
    rest = x.with_row_index().filter(pl.col("index") != i)
    X = np.column_stack(
        [(rest["draft_year"] == 2023).cast(float), (rest["draft_year"] == 2024).cast(float), rest["pick"]]
    )
    b, *_ = np.linalg.lstsq(X, rest["y"].to_numpy(), rcond=None)
    row = x.row(i, named=True)
    pred = b @ [row["draft_year"] == 2023, row["draft_year"] == 2024, row["pick"]]
    assert row["y_exp"] == pytest.approx(np.clip(pred, x["y"].min(), x["y"].max()))
    assert row["y_resid"] == pytest.approx(row["y"] - row["y_exp"])


def test_tail_flag_takes_the_top_share_per_cell():
    t = M.tail_flag(_frame(), "m")
    assert t.group_by("cell").agg(pl.col("tail").sum())["tail"].to_list() == [6] * 4


def test_wilson_interval():
    lo, hi = M.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)


def test_tail_test_finds_a_planted_tail_and_not_a_null_one():
    def run(effect):
        d = M.slot_expectation(_frame(n_per=60, effect=effect), "y", ["pick", "udfa"])
        return M.tail_test(d, "m", "y_resid", n_perm=500)

    hit, null = run(1.0), run(0.0)
    assert hit["beat_tail"] > 0.8 and hit["p_share"] < 0.01 and hit["ratio"] > 2
    assert null["p_share"] > 0.05
