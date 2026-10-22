"""2026 for fun: the explosion of the 2026 rookie class (public Combine results, our physics) and week 1 of 2026.

    uv run python fun/week1_2026/week1_2026.py            # downloads the public files on the first run

Public data (not the competition's), downloaded to data/external/nflverse/ (gitignored), from the nflverse data
releases (https://github.com/nflverse/nflverse-data): Combine results, the 2026 draft, the 2026 roster, and the
weekly Next Gen Stats tables (passing, rushing, receiving). Player-level 10 Hz game tracking for 2026 is not public,
and the NGS fastest-ball-carrier leaderboard needs a browser (its API is signed), so in-game speeds come only from
published reports, listed with sources in published_top_speeds.csv (add rows to extend it).

1. The 2026 class in joules, with the project's physics (src/eeg/physics):
   - 40: v_max from the official 40 time with the sprint model v(t) = v_max (1 − e^(−t/τ)), τ fixed at the median of
     the tracked 40 fits for the position group (competition data, 2023–2025 classes); E_40 = ½ m v_max² + drag work.
     The project measured v_max from the 10 Hz trace; here it is reconstructed from the time, so it is an estimate.
   - Vertical: m g h. Broad: take-off energy at 45°. Bench is left out (no arm length in the public data, and only
     45 of 319 benched).
   - Total explosive energy: the mean of each channel's z-score within the 2026 class (WR, TE, DB, OL, DL, the
     project's groups), at least two channels. Other positions are scored against the same reference.
2. Week 1 of 2026 from the Next Gen Stats tables: how the most explosive rookies did, and the week's NGS leaders.
3. Published week 1 top speeds as kinetic energy, ½ m v², with roster weight.

Writes fun/week1_2026/report.md.
"""

from __future__ import annotations

import urllib.request

import polars as pl
from scipy.optimize import brentq

from eeg import config
from eeg.clustering.audit import _md
from eeg.config import PROJECT_ROOT
from eeg.ingest.units import G
from eeg.physics.broad import energy_broad
from eeg.physics.sprint import best_reps, drag_work, frontal_area, kinetic_energy, time_to_distance
from eeg.physics.vertical import energy_vertical

EXT = PROJECT_ROOT / "data/external/nflverse"
HERE = PROJECT_ROOT / "fun/week1_2026"
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {
    "combine.csv": "combine/combine.csv",
    "draft_picks.csv": "draft_picks/draft_picks.csv",
    "roster_2026.csv": "rosters/roster_2026.csv",
    "ngs_passing.csv.gz": "nextgen_stats/ngs_passing.csv.gz",
    "ngs_rushing.csv.gz": "nextgen_stats/ngs_rushing.csv.gz",
    "ngs_receiving.csv.gz": "nextgen_stats/ngs_receiving.csv.gz",
}
SEASON = 2026
GROUP = {
    "WR": "WR", "TE": "TE", "CB": "DB", "SAF": "DB", "S": "DB", "DB": "DB",
    "OT": "OL", "G": "OL", "C": "OL", "OL": "OL", "EDGE": "DL", "DT": "DL", "DE": "DL", "DL": "DL",
}  # fmt: skip
PFR_TEAMS = {"TAM": "TB", "SFO": "SF", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO", "LVR": "LV", "LAR": "LAR"}
LB_TO_KG, IN_TO_M, MPH_TO_MPS, D40_M = 0.45359237, 0.0254, 0.44704, 40 * 0.9144


def download() -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        if not (EXT / name).exists():
            urllib.request.urlretrieve(f"{BASE}/{path}", EXT / name)


def _read(name: str) -> pl.DataFrame:
    return pl.read_csv(EXT / name, infer_schema_length=0)


def tau_by_group(cfg: dict) -> dict[str, float]:
    """Median τ of the best tracked 40 rep per position group (competition data, 2023–2025)."""
    groups = pl.read_parquet(PROJECT_ROOT / "data/interim/combine_si.parquet").select("player_id", "pos_group")
    fits = best_reps(
        pl.read_parquet(PROJECT_ROOT / "data/interim/forty_fits.parquet"), cfg["sprint"]["max_fit_rmse_mps"]
    )
    t = fits.join(groups, on="player_id").group_by("pos_group").agg(pl.col("tau").median())
    out = dict(zip(t["pos_group"], t["tau"]))
    out["other"] = float(fits["tau"].median())
    return out


def v_max_from_40(t40: float, tau: float) -> float:
    """v_max such that the sprint model covers 40 yd in t40 seconds."""
    return brentq(lambda v: time_to_distance(D40_M, v, tau) - t40, 3.0, 15.0)


def class_joules(cfg: dict) -> pl.DataFrame:
    """The 2026 Combine class with E_40, E_vert, E_broad (J), class z-scores and total explosive energy."""
    c = _read("combine.csv").filter(pl.col("season") == str(SEASON))
    ht = pl.col("ht").str.split("-")
    c = c.with_columns(
        (pl.col("wt").cast(pl.Float64) * LB_TO_KG).alias("mass_kg"),
        ((ht.list.get(0).cast(pl.Float64) * 12 + ht.list.get(1).cast(pl.Float64)) * IN_TO_M).alias("ht_m"),
        pl.col("forty").cast(pl.Float64),
        (pl.col("vertical").cast(pl.Float64) * IN_TO_M).alias("vert_m"),
        (pl.col("broad_jump").cast(pl.Float64) * IN_TO_M).alias("broad_m"),
        pl.col("pos").replace_strict(GROUP, default="other").alias("group"),
    )
    taus, s = tau_by_group(cfg), cfg["sprint"]
    v, e40 = [], []
    for r in c.iter_rows(named=True):
        if r["forty"] is None or r["mass_kg"] is None:
            v.append(None), e40.append(None)
            continue
        tau = taus.get(r["group"], taus["other"])
        vm = v_max_from_40(r["forty"], tau)
        area = frontal_area(r["ht_m"], r["mass_kg"], s["frontal_area_bsa_factor"])
        w = drag_work(vm, tau, r["forty"], s["air_density"], s["drag_coefficient"], area)
        v.append(vm), e40.append(float(kinetic_energy(r["mass_kg"], vm) + w))
    c = c.with_columns(
        pl.Series("v_max_est", v, dtype=pl.Float64),
        pl.Series("E_40", e40, dtype=pl.Float64),
        energy_vertical(pl.col("mass_kg"), pl.col("vert_m"), G).alias("E_vert"),
        energy_broad(pl.col("mass_kg"), pl.col("broad_m"), cfg["broad"]["theta_deg"]).alias("E_broad"),
    )
    ref = c.filter(pl.col("group") != "other")
    chans = ["E_40", "E_vert", "E_broad"]
    c = c.with_columns(((pl.col(ch) - ref[ch].mean()) / ref[ch].std()).alias(f"z_{ch}") for ch in chans)
    n = pl.sum_horizontal(pl.col(f"z_{ch}").is_not_null() for ch in chans)
    return c.with_columns(
        pl.when(n >= 2).then(pl.mean_horizontal(*(f"z_{ch}" for ch in chans))).otherwise(None).alias("explosion_z")
    )


def attach(c: pl.DataFrame) -> pl.DataFrame:
    """Draft pick and team (2026 draft), and the gsis id for the NGS tables (draft or roster)."""
    d = _read("draft_picks.csv").filter(pl.col("season") == str(SEASON)).select(
        pl.col("pfr_player_id").alias("pfr_id"), pl.col("pick").cast(pl.Int64), pl.col("team").replace(PFR_TEAMS).alias("draft_team"),
        pl.col("gsis_id").alias("gsis_draft"),
    )  # fmt: skip
    r = _read("roster_2026.csv").select(
        "pfr_id", pl.col("team").alias("team_now"), pl.col("gsis_id").alias("gsis_roster")
    )
    r = r.filter(pl.col("pfr_id").is_not_null()).unique("pfr_id")
    return (
        c.drop("draft_year", "draft_team", "draft_round", "draft_ovr")  # empty for 2026 in the Combine file
        .join(d, on="pfr_id", how="left")
        .join(r, on="pfr_id", how="left")
        .with_columns(pl.coalesce("gsis_draft", "gsis_roster").alias("gsis_id"))
    )


def ngs_week(name: str, week: int = 1) -> pl.DataFrame:
    return _read(name).filter(
        (pl.col("season") == str(SEASON)) & (pl.col("season_type") == "REG") & (pl.col("week") == str(week))
    )


def week1_line(c: pl.DataFrame) -> pl.DataFrame:
    """Each 2026 rookie's week 1 NGS line, if any: rushing, receiving or passing, as one text column."""
    rush = ngs_week("ngs_rushing.csv.gz").select(
        pl.col("player_gsis_id").alias("gsis_id"),
        pl.format("{} rushes, {} yds, RYOE {}, {} s to LOS", "rush_attempts", "rush_yards",
                  pl.col("rush_yards_over_expected").cast(pl.Float64).round(0), pl.col("avg_time_to_los").cast(pl.Float64).round(2)).alias("rush"),
    )  # fmt: skip
    rec = ngs_week("ngs_receiving.csv.gz").select(
        pl.col("player_gsis_id").alias("gsis_id"),
        pl.format("{} rec / {} tgt, {} yds, sep {} yd, YAC+ {}", "receptions", "targets", "yards",
                  pl.col("avg_separation").cast(pl.Float64).round(1), pl.col("avg_yac_above_expectation").cast(pl.Float64).round(1)).alias("rec"),
    )  # fmt: skip
    pas = ngs_week("ngs_passing.csv.gz").select(
        pl.col("player_gsis_id").alias("gsis_id"),
        pl.format("{} att, {} yds, CPOE {}", "attempts", "pass_yards",
                  pl.col("completion_percentage_above_expectation").cast(pl.Float64).round(1)).alias("pass"),
    )  # fmt: skip
    out = c.join(rush, on="gsis_id", how="left").join(rec, on="gsis_id", how="left").join(pas, on="gsis_id", how="left")
    return out.with_columns(pl.concat_str("rush", "rec", "pass", separator="; ", ignore_nulls=True).alias("week1_ngs"))


def leaders() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Week 1 NGS leaders: rush yards over expected (≥ 8 rushes), YAC above expectation (≥ 3 receptions)."""
    rush = (
        ngs_week("ngs_rushing.csv.gz")
        .filter(pl.col("rush_attempts").cast(pl.Int64) >= 8)
        .select(
            pl.col("player_display_name").alias("player"), pl.col("team_abbr").alias("team"),
            pl.col("rush_attempts").cast(pl.Int64).alias("rushes"), pl.col("rush_yards").cast(pl.Int64).alias("yards"),
            pl.col("rush_yards_over_expected").cast(pl.Float64).alias("RYOE"),
            pl.col("avg_time_to_los").cast(pl.Float64).alias("s_to_LOS"),
        )
        .sort("RYOE", descending=True)
        .head(10)
    )  # fmt: skip
    rec = (
        ngs_week("ngs_receiving.csv.gz")
        .filter(pl.col("receptions").cast(pl.Int64) >= 3)
        .select(
            pl.col("player_display_name").alias("player"), pl.col("team_abbr").alias("team"),
            pl.col("receptions").cast(pl.Int64).alias("rec"), pl.col("yards").cast(pl.Int64),
            pl.col("avg_yac_above_expectation").cast(pl.Float64).alias("YAC_above_exp"),
            pl.col("avg_separation").cast(pl.Float64).alias("separation_yd"),
        )
        .sort("YAC_above_exp", descending=True)
        .head(10)
    )  # fmt: skip
    return rush, rec


def published_speeds() -> pl.DataFrame:
    """Published week 1 top speeds with roster weight: ½ m v² in joules."""
    s = pl.read_csv(HERE / "published_top_speeds.csv")
    r = _read("roster_2026.csv").select(pl.col("full_name").alias("player"), "team", pl.col("weight").cast(pl.Float64))
    return (
        s.join(r.unique(["player", "team"]), on=["player", "team"], how="left")
        .with_columns((pl.col("mph") * MPH_TO_MPS).alias("m_per_s"), (pl.col("weight") * LB_TO_KG).alias("kg"))
        .with_columns((0.5 * pl.col("kg") * pl.col("m_per_s") ** 2).round(0).alias("joules"))
        .select("player", "team", "position", "mph", "m_per_s", "weight", "joules", "play")
    )


def main() -> None:
    download()
    cfg = config.load()
    c = week1_line(attach(class_joules(cfg)))
    ranked = c.filter(pl.col("explosion_z").is_not_null()).sort("explosion_z", descending=True)
    show = [
        pl.col("player_name").alias("player"), "pos",
        pl.col("pick").cast(pl.String).fill_null("UDFA").alias("pick"),
        pl.coalesce("draft_team", "team_now").fill_null("—").alias("team"),
        pl.col("wt").alias("lb"), "forty", pl.col("vertical").alias("vert_in"), pl.col("broad_jump").alias("broad_in"),
        pl.col("v_max_est").round(2).alias("v_max_est"), (pl.col("E_40") / 1000).round(2).alias("E_40_kJ"),
        pl.col("explosion_z").round(2), pl.col("week1_ngs").fill_null(""),
    ]  # fmt: skip
    top = ranked.head(15).select(show)
    by_group = (
        ranked.filter(pl.col("group") != "other")
        .with_columns(pl.col("explosion_z").rank("ordinal", descending=True).over("group").alias("rk"))
        .filter(pl.col("rk") <= 3)
        .sort("group", "rk")
        .select("group", *show[:-1])
    )
    played = c.filter(pl.col("week1_ngs").is_not_null() & (pl.col("week1_ngs") != ""))
    rush, rec = leaders()
    speeds = published_speeds()

    report = f"""# 2026 for fun: the rookie class in joules, and week 1

Generated by `uv run python fun/week1_2026/week1_2026.py`. Public data from the
[nflverse data releases](https://github.com/nflverse/nflverse-data) (Combine results, 2026 draft and roster,
weekly Next Gen Stats tables), not the competition data. Method in the script's docstring.

**What is and isn't public for 2026:** player-level 10 Hz game tracking is not, so there are no game joules per
starter as in `fun/week1_explosion/`. The NGS weekly tables (passing, rushing, receiving) are, and so are the
Combine results. In-game top speeds come only from published reports (`published_top_speeds.csv`, with sources).

## The 2026 class: most explosive by total energy

{ranked.height} of {c.height} Combine participants with at least two of the 40, vertical and broad. `v_max_est`:
top speed reconstructed from the 40 time (m/s); `E_40_kJ`: ½ m v² plus drag work; `explosion_z`: mean of the
three channels' z-scores within the class (raw joules, so size counts). `week1_ngs`: their week 1 NGS line, if they
had one.

{_md(top, 2)}

Top 3 in each of the project's position groups:

{_md(by_group, 2)}

## The 2026 rookies in week 1 (Next Gen Stats)

{played.height} rookies from the Combine list have a week 1 line in the NGS tables (only ball carriers, receivers
and passers have one):

{_md(played.sort("explosion_z", descending=True, nulls_last=True).select(show[0], "pos", show[3], pl.col("explosion_z").round(2), "week1_ngs"), 2)}

## Week 1 2026 Next Gen Stats leaders

Rush yards over expected (at least 8 rushes); `s_to_LOS` is the average time to reach the line of scrimmage, the
closest NGS has to a burst number:

{_md(rush, 2)}

Yards after catch above expectation (at least 3 receptions):

{_md(rec, 2)}

## Published week 1 top speeds, as kinetic energy

From published reports only (sources in `published_top_speeds.csv`); ½ m v² with the 2026 roster weight.

{_md(speeds, 2)}
"""
    (HERE / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
