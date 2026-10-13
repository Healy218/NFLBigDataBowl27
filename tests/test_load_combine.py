import polars as pl
import pytest

from eeg.ingest.load_combine import POS_GROUPS, build_combine_si, parse_height_in


def _raw(rows):
    cols = {
        "season": pl.Int32, "draft_year": pl.Float64, "draft_team": pl.String,
        "draft_round": pl.Float64, "draft_ovr": pl.Float64, "pfr_id": pl.String,
        "cfb_id": pl.String, "player_name": pl.String, "pos": pl.String, "school": pl.String,
        "ht": pl.String, "wt": pl.Float64, "forty": pl.Float64, "bench": pl.Float64,
        "vertical": pl.Float64, "broad_jump": pl.Float64, "cone": pl.Float64,
        "shuttle": pl.Float64,
    }  # fmt: skip
    base = {c: None for c in cols}
    return pl.DataFrame([{**base, **r} for r in rows], schema=cols)


@pytest.fixture
def out():
    raw = _raw(
        [
            dict(season=2019, player_name="A. Receiver", school="Ohio State", pos="WR",
                 pfr_id="RecA00", cfb_id="a-receiver-1", ht="6-2", wt=200.0, forty=4.40,
                 vertical=40.0, broad_jump=130.0, bench=15.0, cone=6.9, shuttle=4.1),
            dict(season=2021, player_name="B. Edge", school="Texas A&M", pos="EDGE",
                 pfr_id="EdgB00", ht="6-4", wt=260.0, forty=4.7),
            # Two different people sharing a pfr_id; one is out of scope (ILB).
            dict(season=2007, player_name="C. Same", school="FSU", pos="WR", pfr_id="SamC99", ht="6-0"),
            dict(season=2007, player_name="C. Same", school="FSU", pos="ILB", pfr_id="SamC99", ht="6-1"),
            dict(season=2010, player_name="D. Kicker", school="X", pos="K", ht="6-0"),
        ]
    )  # fmt: skip
    return build_combine_si(raw)


def test_parse_height():
    df = pl.DataFrame({"ht": ["6-2", "5-11", "6-10", " 6-0 ", "74", None, "6-"]})
    assert df.select(parse_height_in())["ht_in"].to_list() == [74.0, 71.0, 82.0, 72.0, None, None, None]


def test_scope_and_groups(out):
    assert out.height == 3  # ILB and K dropped
    assert set(out["pos"]) == {"WR", "EDGE"}
    assert dict(zip(out["pos"], out["pos_group"])) == {"WR": "WR", "EDGE": "DL"}
    assert set(POS_GROUPS.values()) == {"WR", "TE", "DB", "OL", "DL"}


def test_si_conversion(out):
    r = out.filter(pl.col("player_name") == "A. Receiver").row(0, named=True)
    assert r["ht_m"] == pytest.approx(74 * 0.0254)
    assert r["mass_kg"] == pytest.approx(200 * 0.45359237)
    assert r["vertical_m"] == pytest.approx(1.016)
    assert r["broad_m"] == pytest.approx(3.302)
    assert r["forty_s"] == 4.40 and r["bench_reps"] == 15.0
    assert r["n_drills_raw"] == 6
    assert not any(r[f"{k}_missing"] for k in ["ht", "wt", "forty", "vertical", "broad", "bench", "cone", "shuttle"])
    assert r["arm_missing"] and r["arm_m"] is None


def test_flags(out):
    edge = out.filter(pl.col("pos") == "EDGE").row(0, named=True)
    assert edge["test_setting"] == "pro_day"
    assert edge["n_drills_raw"] == 1 and edge["vertical_missing"]

    same = out.filter(pl.col("player_name") == "C. Same").row(0, named=True)
    assert same["pfr_id_conflict"]
    assert same["test_setting"] == "combine"


def test_player_id(out):
    assert out["player_id"].is_unique().all()
    assert "2021_b-edge_texas-a-m_EDGE" in out["player_id"].to_list()
