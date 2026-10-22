import numpy as np
import polars as pl
import pytest

from eeg.ngs import eta as E


def _seasons():
    """Two prospects; prospect 2 has a second season and an unqualified one; prospect 3 has no lab."""
    rows = [
        (1, 2023, True, 200),
        (2, 2023, True, 100),
        (2, 2024, True, 300),
        (2, 2025, False, 50),
        (3, 2024, True, 150),
    ]
    s = pl.DataFrame(rows, schema=["player_id", "season", "qualified", "plays"], orient="row")
    speed = [8.0, 7.0, 7.5, 6.0, 9.0]
    return s.with_columns(
        games=pl.lit(8),
        s_peak_top3=pl.Series(speed),
        ke_peak_top3=0.5 * pl.Series(speed) ** 2,
        burst_a_top3=pl.lit(5.0),
        p_peak_top3=pl.lit(20.0),
    )


def _lab():
    return pl.DataFrame(
        {
            "player_id": [1, 2],
            "lab_s_peak": [10.0, 9.0],
            "lab_ke_peak": [50.0, 40.5],
            "lab_a_peak": [4.0, 5.0],
            "lab_p_peak": [25.0, 20.0],
        }
    )


def test_season_eta_keeps_qualified_with_lab_and_ke_is_speed_squared():
    eta = E.season_eta(_seasons(), _lab(), "top3", "ke")
    assert eta.select("player_id", "season").rows() == [(1, 2023), (2, 2023), (2, 2024)]
    assert eta["eta_speed"].to_list() == pytest.approx([0.8, 7 / 9, 7.5 / 9])
    assert eta["eta_ke"].to_numpy() == pytest.approx(eta["eta_speed"].to_numpy() ** 2)
    assert eta["eta_burst"].to_list() == pytest.approx([1.25, 1.0, 1.0])
    assert eta["gap"].to_numpy() == pytest.approx(1 - eta["eta_ke"].to_numpy())


def test_career_eta_is_weighted_geometric_mean_and_keeps_first_season():
    eta = E.season_eta(_seasons(), _lab(), "top3", "ke")
    car = E.career_eta(eta, "plays", "ke").filter(pl.col("player_id") == 2).row(0, named=True)
    assert car["seasons"] == 2 and car["first_season"] == 2023 and car["plays"] == 400
    expected = np.exp((100 * np.log(7 / 9) + 300 * np.log(7.5 / 9)) / 400)
    assert car["eta_speed_career"] == pytest.approx(expected)
    assert car["eta_ke_career"] == pytest.approx(expected**2)  # geometric mean keeps η_ke = η_speed²
    assert car["eta_speed_first"] == pytest.approx(7 / 9)
    assert car["gap_career"] == pytest.approx(1 - expected**2)


def test_role_table_separates_denominator_driven_ratio():
    rng = np.random.default_rng(0)
    n = 600
    group = np.where(np.arange(n) % 2, "A", "B")
    shift = np.where(group == "A", 0.0, -0.2)
    # speed: field = lab × an efficiency independent of the lab; burst: field unrelated to a noisy lab
    lab_s = np.exp(rng.normal(2.2, 0.05, n))
    field_s = lab_s * np.exp(shift + rng.normal(-0.15, 0.05, n))
    lab_b = np.exp(rng.normal(1.5, 0.15, n))
    field_b = np.exp(rng.normal(1.5, 0.1, n))
    eta = pl.DataFrame({"player_id": np.arange(n)})
    for name, f, lab in [("speed", field_s, lab_s), ("burst", field_b, lab_b)]:
        eta = eta.with_columns(**{f"field_{name}": f, f"lab_{name}": lab, f"eta_{name}": f / lab})
    for name in ("ke", "power"):
        eta = eta.with_columns(**{f"{p}_{name}": pl.col(f"{p}_speed") for p in ("field", "lab", "eta")})
    groups = pl.DataFrame({"player_id": np.arange(n), "group": group})
    t = {r["channel"]: r for r in E.role_table(eta, groups).iter_rows(named=True)}
    assert t["speed"]["between_share"] > 0.8 and abs(t["speed"]["r_lab_within"]) < 0.1
    assert t["burst"]["between_share"] < 0.05 and t["burst"]["r_lab_within"] < -0.7


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")  # constant channels in the fixture
def test_season_retest_pairs_consecutive_seasons():
    eta = E.season_eta(_seasons(), _lab(), "top3", "ke")
    extra = eta.with_columns(pl.col("player_id") + 10, pl.col("eta_speed") * 1.01)  # more pairs for a correlation
    both = pl.concat([eta, extra, eta.with_columns(pl.col("player_id") + 20, pl.col("eta_speed") * 0.9)])
    groups = pl.DataFrame({"player_id": [1, 2, 11, 12, 21, 22], "group": ["A"] * 6})
    rt = E.season_retest(both, groups).filter(pl.col("channel") == "speed").row(0, named=True)
    assert rt["pairs"] == 3  # only prospect 2 (and its copies) has consecutive qualified seasons
