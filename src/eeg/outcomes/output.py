"""Potential and η against regular-season output, within group, beyond the draft.

    uv run python -m eeg.outcomes.output

Outcomes (`outcomes` in config/physics.yaml): playing time for every group (plays per league season, zeros kept,
log1p) and one production measure per group (WR / TE YPRR, DL pressure rate, OL peak pressure probability allowed;
DB none). DL get-off is a realization channel next to η.

1. Draft control (`outcomes.draft_control`, open decision 2): cross-validated R² of playing time on each candidate
   form of the draft slot, within group × draft class. The form in use is the control in everything below.
2. Combine potential → playing time, every prospect with the predictors: draft alone, the Combine alone (profile
   axes, a10, mass, NGS college production), and both. Does the Combine add anything the draft has not priced?
3. η → next season's playing time. η of a prospect's first qualified season (season s) against plays in s + 1, with
   the draft and plays in s as controls, within group × season. Nothing from s + 1 enters a predictor. The engine
   (log lab KE) and η enter together: log field KE = log lab KE + log η_KE, so equal betas would mean only the game
   value matters, and a gap between them says which side carries the signal. DL get-off is tested the same way.
4. η and production (concurrent): per prospect, mean production over qualified seasons with η against career η
   and the engine, controls draft and log plays per season, within group. Same-season measures, so association,
   not prediction.
5. The DL chain: Combine start (a10) → game get-off → pressure rate, per DL prospect over qualified seasons, beyond
   the draft and log plays, centred on DL and on NFL position (edge and interior rushers differ in build, start
   and pressure rate, so the position-centred fit is the role check).

η uses the exposure-free season value (`games.season_statistic` = top3_of50); the earlier game-based value grew
with plays and would have credited η with playing time (reports/field_reliability.md).

Every statistic is within position group (values centred on group or group × class / season means); groups only
read results. Writes reports/output.md (aggregates only) and reports/figures/output.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.ngs.eta_readout import within_r
from eeg.ngs.transfer import COMPACT, cv_r2, fit_within

UDFA_PICK = 258  # one past the last pick; the undrafted indicator absorbs the difference
BANDS = ("2-3", "4-5", "6-7")  # round 1 is the reference; UDFA is `udfa`
DRAFT_FORMS = {
    "pick + undrafted": ["pick", "udfa"],
    "log pick + undrafted": ["log_pick", "udfa"],
    "round bands": [*(f"band_{b}" for b in BANDS), "udfa"],
    "log pick + round bands": ["log_pick", *(f"band_{b}" for b in BANDS), "udfa"],
}


def draft_features(outcomes: pl.DataFrame) -> pl.DataFrame:
    """Per prospect: draft class and every candidate draft-control column."""
    pick = pl.col("draft_overall_pick").fill_null(UDFA_PICK).cast(pl.Float64)
    return outcomes.select(
        "player_id",
        "draft_year",
        pick.alias("pick"),
        pick.log().alias("log_pick"),
        pl.col("undrafted").cast(pl.Float64).alias("udfa"),
        *((pl.col("round_band").cast(pl.String) == b).cast(pl.Float64).alias(f"band_{b}") for b in BANDS),
    )


def playing_time(seasons: pl.DataFrame) -> pl.DataFrame:
    """Per prospect: log1p of mean plays per league season (zeros kept) and seasons available."""
    return seasons.group_by("player_id").agg(
        pl.col("plays").mean().log1p().alias("pt"), pl.len().alias("seasons_available")
    )


def next_season(eta: pl.DataFrame, career: pl.DataFrame, seasons: pl.DataFrame) -> pl.DataFrame:
    """First qualified η season s of each prospect paired with plays in s + 1 (prospects whose s + 1 is in the
    data only). log η_KE, log lab and field KE, log plays in s, log1p plays in s + 1."""
    first = eta.join(career.select("player_id", pl.col("first_season").alias("season")), on=["player_id", "season"])
    nxt = seasons.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("plays").alias("plays_next"))
    return first.join(nxt, on=["player_id", "season"]).select(
        "player_id",
        "season",
        pl.col("eta_ke").log().alias("log_eta"),
        pl.col("lab_ke").log().alias("log_lab_ke"),
        pl.col("field_ke").log().alias("log_field_ke"),
        pl.col("plays").log().alias("log_plays"),
        pl.col("plays_next").log1p().alias("pt_next"),
    )


def career_production(prod: pl.DataFrame, eta: pl.DataFrame, measures: dict[str, str]) -> pl.DataFrame:
    """Per prospect, over qualified seasons with η: mean of the group's production measure (`production`),
    mean log η_KE and log lab KE, mean log plays per season, and mean log get-off (DL)."""
    rows = prod.filter("qualified").join(
        eta.select("player_id", "season", "eta_ke", "lab_ke"), on=["player_id", "season"]
    )
    value = pl.coalesce(
        *(pl.when(pl.col("group") == g).then(pl.col(m)) for g, m in measures.items())
    )  # fmt: skip
    return rows.group_by("player_id", "group").agg(
        value.mean().alias("production"),
        pl.col("eta_ke").log().mean().alias("log_eta"),
        pl.col("lab_ke").log().mean().alias("log_lab_ke"),
        pl.col("plays").log().mean().alias("log_plays"),
        pl.col("get_off").log().mean().alias("log_get_off"),
        pl.len().alias("seasons"),
    )


def dl_chain(prod: pl.DataFrame, prospects: pl.DataFrame) -> pl.DataFrame:
    """Per DL prospect over qualified seasons: mean pressure rate, mean log get-off, mean log plays, joined to
    prospect columns (NFL position, a10, draft)."""
    q = prod.filter(pl.col("qualified") & (pl.col("group") == "DL"))
    return (
        q.group_by("player_id")
        .agg(
            pl.col("pressure_rate").mean(),
            pl.col("get_off").log().mean().alias("log_get_off"),
            pl.col("plays").log().mean().alias("log_plays"),
        )
        .join(prospects, on="player_id")
    )


def _varying(d: pl.DataFrame, xs, group: str) -> list[str]:
    """Predictors that vary within at least one cell (a constant column has no within-group slope)."""
    return [x for x in xs if d.select((pl.col(x) - pl.col(x).mean().over(group)).abs().max()).item() > 1e-12]


def model_row(d: pl.DataFrame, y: str, xs, group: str, label: str, keep=()) -> dict:
    """fit_within and cv_r2 on rows complete in y and xs; betas of `keep` with 95% intervals."""
    sub = d.drop_nulls([y, *xs])
    xs = _varying(sub, xs, group)
    f = fit_within(sub, y, xs, group)
    out = dict(model=label, n=f["n"], r2=f["r2"], cv_r2=cv_r2(sub, y, xs, group))
    for c in keep:
        out |= {f"b_{c}": f.get(f"b_{c}"), f"lo_{c}": f.get(f"lo_{c}"), f"hi_{c}": f.get(f"hi_{c}")}
    return out


def betas_long(d: pl.DataFrame, y: str, xs, group: str, show, label: str) -> list[dict]:
    """Standardized betas of the `show` predictors (with 95% intervals) from one within-group fit."""
    sub = d.drop_nulls([y, *xs])
    xs = _varying(sub, xs, group)
    f = fit_within(sub, y, xs, group)
    return [
        dict(outcome=label, variable=c, n=f["n"], beta=f[f"b_{c}"], lo95=f[f"lo_{c}"], hi95=f[f"hi_{c}"])
        for c in show
        if c in xs
    ]


def plot_output(cv: pl.DataFrame, eff: pl.DataFrame, path) -> None:
    """Left: CV R² of playing-time models. Right: standardized betas of η and the engine on each outcome."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    y = np.arange(cv.height)
    ax0.barh(y, cv["cv_r2"], color=["C7" if "draft" in m and "+" not in m else "C0" for m in cv["model"]])
    ax0.set_yticks(y, cv["model"], fontsize=8)
    ax0.invert_yaxis()
    ax0.axvline(0, color="grey", lw=0.6)
    ax0.set_xlabel("cross-validated R² beyond the group mean")
    ax0.set_title("Playing time: the draft has priced the Combine")
    names = eff["outcome"] + " · " + eff["variable"]
    yy = np.arange(eff.height)
    colour = ["C3" if v == "log_eta" else ("C2" if v == "log_get_off" else "C0") for v in eff["variable"]]
    ax1.errorbar(
        eff["beta"], yy, xerr=[eff["beta"] - eff["lo95"], eff["hi95"] - eff["beta"]], fmt="none", ecolor=colour
    )
    ax1.scatter(eff["beta"], yy, c=colour, s=18, zorder=3)
    ax1.set_yticks(yy, names, fontsize=7)
    ax1.invert_yaxis()
    ax1.axvline(0, color="grey", lw=0.6)
    ax1.set_xlabel("standardized beta beyond the draft and current playing time (95% CI)")
    ax1.set_title("η (red), engine (blue), DL get-off (green)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT
    from eeg.ngs.transfer import prospect_table

    cfg = config.load()
    measures = cfg["outcomes"]["production"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    seasons = pl.read_parquet(processed / "outcomes_seasons.parquet")
    draft = draft_features(pl.read_parquet(processed / "outcomes.parquet"))
    d = (
        prospect_table(processed, interim, cfg["games"]["season_statistic"])
        .join(draft, on="player_id")
        .join(playing_time(seasons), on="player_id")
        .with_columns(pl.concat_str("group", pl.col("draft_year").cast(pl.String), separator="|").alias("cell"))
    )

    # 1. draft control
    forms = pl.DataFrame([model_row(d, "pt", xs, "cell", name) for name, xs in DRAFT_FORMS.items()])
    control = cfg["outcomes"]["draft_control"]
    rho = pl.DataFrame(
        [
            dict(group=g, n=x.height, r_pick=within_r(x, "pick", "pt", "draft_year")["r"])
            for (g,), x in d.group_by(["group"], maintain_order=True)
        ]
    ).sort("group")

    # 2. Combine potential -> playing time
    pot = d.drop_nulls([*COMPACT, *control])
    pt_models = pl.DataFrame(
        [
            model_row(pot, "pt", control, "cell", "draft"),
            model_row(pot, "pt", COMPACT, "cell", "Combine"),
            model_row(pot, "pt", [*control, *COMPACT], "cell", "draft + Combine"),
        ]
    )
    pt_betas = pl.DataFrame(
        betas_long(pot, "pt", COMPACT, "cell", COMPACT, "Combine alone")
        + betas_long(pot, "pt", [*control, *COMPACT], "cell", COMPACT, "beyond the draft")
    )

    # 3. first-season η -> next season's playing time
    eta = pl.read_parquet(processed / "eta.parquet")
    career = pl.read_parquet(processed / "eta_career.parquet")
    prod = pl.read_parquet(processed / "production_seasons.parquet")
    ns = (
        next_season(eta, career, seasons)
        .join(d.select("player_id", "group", *control), on="player_id")
        .join(
            prod.select("player_id", "season", pl.col("get_off").log().alias("log_get_off")),
            on=["player_id", "season"],
            how="left",
        )
        .with_columns(pl.concat_str("group", pl.col("season").cast(pl.String), separator="|").alias("cell"))
    )
    base = [*control, "log_plays"]
    ns_models = pl.DataFrame(
        [
            model_row(ns, "pt_next", ["log_eta"], "cell", "η alone", ["log_eta"]),
            model_row(ns, "pt_next", control, "cell", "draft"),
            model_row(ns, "pt_next", base, "cell", "draft + plays in s"),
            model_row(ns, "pt_next", [*base, "log_eta"], "cell", "+ η", ["log_eta"]),
            model_row(
                ns, "pt_next", [*base, "log_lab_ke", "log_eta"], "cell", "+ engine + η", ["log_lab_ke", "log_eta"]
            ),
        ]
    )
    dl_ns = ns.filter(pl.col("group") == "DL")
    dl_models = pl.DataFrame(
        [
            model_row(dl_ns, "pt_next", base, "cell", "DL: draft + plays in s"),
            model_row(dl_ns, "pt_next", [*base, "log_get_off"], "cell", "DL: + get-off", ["log_get_off"]),
            model_row(dl_ns, "pt_next", [*base, "log_get_off", "log_eta"], "cell", "DL: + get-off + η", ["log_get_off", "log_eta"]),
        ]
    )  # fmt: skip
    eta_checks = pl.DataFrame(
        [within_r(ns, x, "log_eta", "cell") for x in ["log_plays", "pick", "pt_next"]]
        + [within_r(ns, "log_lab_ke", "log_eta", "cell")]
    )

    # 4. η and production (concurrent, per prospect)
    cp = career_production(prod, eta, measures).join(draft.select("player_id", *control), on="player_id")
    cp_ctrl = [*control, "log_plays"]
    eff = []
    prod_rows = []
    for g, m in measures.items():
        x = cp.filter(pl.col("group") == g)
        r = within_r(x, "log_eta", "production")
        prod_rows.append(dict(group=g, measure=m, n=r["n"], r_eta=r["r"], lo95=r["lo95"], hi95=r["hi95"]))
        eff += betas_long(
            x, "production", [*cp_ctrl, "log_lab_ke", "log_eta"], "group", ["log_lab_ke", "log_eta"], f"{g} {m}"
        )
    dl_cp = cp.filter(pl.col("group") == "DL")
    eff += betas_long(
        dl_cp, "production", [*cp_ctrl, "log_get_off", "log_eta"], "group", ["log_get_off"], "DL pressure_rate"
    )
    eff_prod = pl.DataFrame(eff)
    eff_pt = pl.DataFrame(
        betas_long(ns, "pt_next", [*base, "log_lab_ke", "log_eta"], "cell", ["log_lab_ke", "log_eta"], "next-season plays")
        + betas_long(dl_ns, "pt_next", [*base, "log_get_off"], "cell", ["log_get_off"], "DL next-season plays")
    )  # fmt: skip
    # 5. DL chain
    chain = dl_chain(prod, d.select("player_id", "group", "pos", "log_a10", *control))
    chain_rows = []
    for centre, g in [("DL", "group"), ("NFL position", "pos")]:
        for xs, show, label in [
            (["log_a10"], ["log_a10"], "pressure ~ a10"),
            (["log_get_off"], ["log_get_off"], "pressure ~ get-off"),
            (["log_a10", "log_get_off"], ["log_a10", "log_get_off"], "pressure ~ a10 + get-off"),
        ]:
            chain_rows += [
                r | {"centred_on": centre}
                for r in betas_long(chain, "pressure_rate", [*control, "log_plays", *xs], g, show, label)
            ]
        chain_rows += [
            r | {"centred_on": centre}
            for r in betas_long(chain, "log_get_off", ["log_a10"], g, ["log_a10"], "get-off ~ a10")
        ]
    chain_tbl = pl.DataFrame(chain_rows).select("centred_on", "outcome", "variable", "n", "beta", "lo95", "hi95")
    plot_output(
        pl.concat([pt_models.select("model", "cv_r2"), ns_models.select("model", "cv_r2").with_columns(
            ("next season: " + pl.col("model")).alias("model"))]),
        pl.concat([eff_pt, eff_prod]),
        reports / "figures" / "output.png",
    )  # fmt: skip

    report = f"""# Potential and η against regular-season output

Generated by `uv run python -m eeg.outcomes.output`. Every statistic is within position group (centred on group ×
draft class, or group × season); groups only read results. Aggregates only. Playing time = log1p of plays per
league season (zeros kept; WR / TE plays are route plays). η is the exposure-free season value
(`{cfg["games"]["season_statistic"]}`).

## 1. Draft control (open decision 2)

Playing time on each form of the draft slot, within group × draft class ({d.height} prospects). `cv_r2`: out of
sample beyond the cell mean (5-fold × 20).

{_md(forms.select("model", "n", "r2", "cv_r2"), 3)}

In use (`outcomes.draft_control`): **{" + ".join(control)}**, the highest `cv_r2`. Within-class r of pick with playing time by group:

{_md(rho, 2)}

## 2. Combine potential → playing time

Prospects with every Combine predictor ({", ".join(f"`{c}`" for c in COMPACT)}) and the draft: {pot.height}.

{_md(pt_models.select("model", "n", "r2", "cv_r2"), 3)}

Standardized betas (95% CI), alone and beyond the draft:

{_md(pt_betas, 2)}

## 3. First-season η → next season's playing time

First qualified η season s, plays in s + 1 (zeros kept): {ns.height} prospects (s = 2023 or 2024). Centred on
group × season. Controls: the draft and log plays in s.

{_md(ns_models, 2)}

DL get-off (log, season s) as the realization channel ({dl_ns.drop_nulls("log_get_off").height} DL):

{_md(dl_models, 2)}

Checks, within group × season (r with log η_KE of season s): with plays in s (the exposure-bias check; the
game-based value gave +0.46), draft pick, plays in s + 1 and the engine (log lab KE, the η denominator):

{_md(eta_checks, 2)}

## 4. η and production (same seasons)

Per prospect, means over qualified seasons with η. Association, not prediction: production and η come from the
same seasons. Within-group r of production with log η_KE:

{_md(pl.DataFrame(prod_rows), 2)}

Standardized betas beyond the draft and log plays per season, engine (log lab KE) and η together; for DL,
get-off with η:

{_md(eff_prod, 2)}

## 5. The DL chain: Combine start → get-off → pressure

Per DL prospect over qualified seasons ({chain.height}; {chain.drop_nulls("log_a10").height} with an official 10-yard split). Pressure rate beyond
the draft and log plays; get-off on a10 alone. Centred on DL and, as the role check, on NFL position (DE, OLB, DT,
NT). If a10 shrinks once get-off is in the model, the Combine start reaches pressure through the game start.

{_md(chain_tbl, 2)}

Figure: `reports/figures/output.png`.
"""
    (reports / "output.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
