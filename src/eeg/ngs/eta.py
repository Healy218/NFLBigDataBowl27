"""Explosion efficiency η = field / lab on matching channels, per player-season and per prospect.

    uv run python -m eeg.ngs.eta

Per qualified player-season with a lab 40 (field_seasons.parquet `qualified`,
lab_channels.parquet), with the season value `games.season_statistic`:
- η_speed = field s_peak / lab s_peak
- η_ke    = field ke_peak / lab ke_peak = η_speed²   (mass cancels)
- η_burst = field burst_a / lab a_peak               (both from near rest)
- η_power = field p_peak / lab p_peak                (W/kg)
- gap     = 1 − η of `eta.headline` (ke): the Explosion Efficiency Gap.

η is stored raw (`eta.role`): position groups are only used to read it, and
Iteration 4 stratifies by them. Per prospect: the first qualified season's η
and the career η, exp of the `eta.career_weight`-weighted mean of season log η
(geometric, so η_ke stays η_speed²).

Diagnostics (all on log η, within position = centred on the group mean): the
share of variance between groups, within-group r with the lab and the field
value (a ratio whose variance comes from the lab side measures the lab, not
transfer), correlation between channels, and retest across consecutive seasons
(the lab value is shared, so this retests the field side only).

Writes data/processed/eta.parquet (player-season), data/processed/eta_career.parquet
(prospect) (not committed), reports/eta.md (aggregates only) and reports/figures/eta.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.ngs.channels import LAB

KEY = ["player_id", "season"]
ETA = {"speed": "s_peak", "ke": "ke_peak", "burst": "burst_a", "power": "p_peak"}  # η name -> field channel


def season_eta(seasons: pl.DataFrame, lab: pl.DataFrame, stat: str, headline: str) -> pl.DataFrame:
    """η per qualified player-season with a lab value: field, lab and η per channel, plus gap = 1 − η_headline."""
    cols = []
    for name, ch in ETA.items():
        f, lab_col = pl.col(f"{ch}_{stat}"), pl.col(LAB[ch])
        cols += [f.alias(f"field_{name}"), lab_col.alias(f"lab_{name}"), (f / lab_col).alias(f"eta_{name}")]
    return (
        seasons.filter("qualified")
        .join(lab, on="player_id")
        .select(*KEY, "games", "plays", *cols)
        .with_columns((1 - pl.col(f"eta_{headline}")).alias("gap"))
        .sort(KEY)
    )


def career_eta(eta: pl.DataFrame, weight: str, headline: str) -> pl.DataFrame:
    """Per prospect: seasons, first-season η and career η (weighted geometric mean over seasons) per channel."""
    w = pl.col(weight)
    first = [pl.col(f"eta_{n}").sort_by("season").first().alias(f"eta_{n}_first") for n in ETA]
    career = [
        ((pl.col(f"eta_{n}").log() * w).sum() / w.filter(pl.col(f"eta_{n}").is_not_null()).sum())
        .exp()
        .alias(f"eta_{n}_career")
        for n in ETA
    ]
    return (
        eta.group_by("player_id")
        .agg(
            pl.len().alias("seasons"),
            pl.col("season").min().alias("first_season"),
            pl.col("plays").sum(),
            *first,
            *career,
        )
        .with_columns(
            (1 - pl.col(f"eta_{headline}_first")).alias("gap_first"),
            (1 - pl.col(f"eta_{headline}_career")).alias("gap_career"),
        )
        .sort("player_id")
    )


def _within(df: pl.DataFrame, cols, by: str = "group") -> pl.DataFrame:
    df = df.drop_nulls(cols)
    return df.with_columns((pl.col(c) - pl.col(c).mean().over(by)).alias(c) for c in cols)


def _corr(df: pl.DataFrame, a: str, b: str) -> float:
    return float(np.corrcoef(df[a].to_numpy(), df[b].to_numpy())[0, 1])


def role_table(eta: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """Per channel: SD of log η pooled and within group, share of variance between groups, and the within-group
    r of log η with log lab and log field."""
    d = eta.join(groups, on="player_id")
    rows = []
    for n in ETA:
        x = d.select("group", *(pl.col(f"{p}_{n}").log().alias(p) for p in ("eta", "lab", "field")))
        x = x.drop_nulls()
        w = _within(x, ["eta", "lab", "field"])
        rows.append(
            dict(
                channel=n,
                n=x.height,
                sd_pooled=x["eta"].std(),
                sd_within=w["eta"].std(),
                between_share=1 - w["eta"].var() / x["eta"].var(),
                r_lab_within=_corr(w, "eta", "lab"),
                r_field_within=_corr(w, "eta", "field"),
            )
        )
    return pl.DataFrame(rows)


def channel_corr(eta: pl.DataFrame, groups: pl.DataFrame, names=("speed", "burst", "power")) -> pl.DataFrame:
    """Within-group correlation of log η between channels."""
    cols = [f"eta_{n}" for n in names]
    d = _within(eta.join(groups, on="player_id").select("group", *(pl.col(c).log() for c in cols)), cols)
    m = np.corrcoef(d.select(cols).to_numpy().T)
    return pl.DataFrame(m, schema=list(names)).insert_column(0, pl.Series("", list(names)))


def season_retest(eta: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """r of log η between consecutive seasons of a prospect, pooled and within group."""
    a = eta.select(*KEY, *(pl.col(f"eta_{n}").log().alias(n) for n in ETA))
    pairs = a.join(a.with_columns(pl.col("season") - 1), on=KEY, suffix="_next").join(groups, on="player_id")
    rows = []
    for n in ETA:
        p = pairs.drop_nulls([n, f"{n}_next"])
        w = _within(p, [n, f"{n}_next"])
        rows.append(
            dict(channel=n, pairs=p.height, r_pooled=_corr(p, n, f"{n}_next"), r_within=_corr(w, n, f"{n}_next"))
        )
    return pl.DataFrame(rows)


def plot_eta(eta: pl.DataFrame, groups: pl.DataFrame, order, path) -> None:
    """Left: η_ke per player-season by position group. Right: within-group log η against log lab for speed and
    burst (a ratio driven by its denominator slopes down)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = eta.join(groups, on="player_id")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    data = [d.filter(pl.col("group") == g)["eta_ke"].to_numpy() for g in order]
    ax0.boxplot(data, tick_labels=list(order), showfliers=False)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        ax0.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=6, alpha=0.4)
    ax0.set_ylabel("η_KE (field / lab kinetic energy)")
    ax0.set_title("η_KE per player-season")
    for n, colour in [("speed", "C0"), ("burst", "C3")]:
        x = d.select("group", pl.col(f"eta_{n}").log().alias("eta"), pl.col(f"lab_{n}").log().alias("lab"))
        w = _within(x, ["eta", "lab"])
        ax1.scatter(w["lab"], w["eta"], s=6, alpha=0.4, color=colour, label=f"{n} (r {_corr(w, 'eta', 'lab'):.2f})")
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.set_xlabel("log lab value, centred within position")
    ax1.set_ylabel("log η, centred within position")
    ax1.set_title("Is η driven by its denominator?")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.clustering.readout import POS_ORDER
    from eeg.config import PROJECT_ROOT
    from eeg.ngs.channels import describe

    cfg = config.load()
    stat, e = cfg["games"]["season_statistic"], cfg["eta"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    groups = pl.read_parquet(interim / "combine_si.parquet").select("player_id", pl.col("pos_group").alias("group"))
    seasons = pl.read_parquet(processed / "field_seasons.parquet")
    lab = pl.read_parquet(processed / "lab_channels.parquet")
    profiled = pl.read_parquet(processed / "profiles.parquet").select("player_id")

    eta = season_eta(seasons, lab, stat, e["headline"])
    career = career_eta(eta, e["career_weight"], e["headline"])
    eta.write_parquet(processed / "eta.parquet")
    career.write_parquet(processed / "eta_career.parquet")

    qual = seasons.filter("qualified")
    no_lab = qual.join(lab, on="player_id", how="anti")
    eta_cols = [f"eta_{n}" for n in ETA] + ["gap"]
    by_pos = eta.join(groups, on="player_id").with_columns(pl.col("group").cast(pl.Enum(POS_ORDER)))
    by_pos = by_pos.sort("group").rename({"group": "pos_group"})
    car_pos = career.join(groups, on="player_id").with_columns(pl.col("group").cast(pl.Enum(POS_ORDER)))
    car_pos = car_pos.sort("group").rename({"group": "pos_group"})
    seasons_per = career["seasons"].value_counts().sort("seasons")
    plot_eta(eta, groups, POS_ORDER, reports / "figures" / "eta.png")

    report = f"""# Explosion efficiency η

Generated by `uv run python -m eeg.ngs.eta`. η = field season value (`{stat}`) / lab 40 value, same estimator on
both sides. Gap = 1 − η_{e["headline"]} (`eta.headline`). η is raw (`eta.role` = {e["role"]}): position groups only
read it. Per-season and per-prospect tables are in `data/processed/eta.parquet` and `eta_career.parquet` (not
committed). Aggregates only.

- Player-seasons with η: {eta.height} of {qual.height} qualified ({no_lab.height} seasons of
  {no_lab["player_id"].n_unique()} prospects have no well-fit 40); prospects: {career.height}, of which
  {career.join(profiled, on="player_id").height} have a kinetic profile.
- Qualified seasons per prospect: {", ".join(f"{r['seasons']}: {r['count']}" for r in seasons_per.iter_rows(named=True))}.

## η per player-season

{_md(describe(eta, eta_cols), 3)}

By position group (post hoc reading):

{_md(describe(by_pos, ["eta_ke", "eta_speed", "eta_burst", "eta_power"], by="pos_group"), 3)}

## Role and what drives each ratio (log η)

`between_share`: variance between position groups. `r_lab_within` / `r_field_within`: within-group r of log η with
log lab and log field. A ratio that tracks its field side measures game output against a fixed engine; one that
tracks (minus) its lab side mostly measures the lab value.

{_md(role_table(eta, groups), 3)}

Within-group correlation between channels:

{_md(channel_corr(eta, groups), 2)}

## Retest across consecutive seasons

The lab value is shared by both seasons, so this retests the field side: change between seasons is real
change (role, development, injury) plus game noise.

{_md(season_retest(eta, groups), 3)}

## Per prospect

`first`: first qualified season (usable at that point in a career). `career`: {e["career_weight"]}-weighted
geometric mean over qualified seasons (uses later seasons: look-ahead for survival).

{_md(describe(car_pos, ["eta_ke_first", "eta_ke_career", "gap_career"], by="pos_group"), 3)}

Figure: `reports/figures/eta.png`.
"""
    (reports / "eta.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
