"""Tenure: does the explosion tail stay in the league longer, beyond the draft slot?

    uv run python -m eeg.survival.tenure

Secondary outcome (open decision 3). From `outcomes.parquet`: a prospect who played exits when they have no
scrimmage play in any later season through `outcomes.last_season` (2025). The 2023–2025 classes give at most three
seasons, so:
- At risk: prospects who played and debuted by last_season − 1 (a 2025 debut has no later season to miss).
- Time: seasons from the debut season through the last season with a play (1, 2, ...). Event: exited. Censored:
  still playing in last_season (the last observed season is then not an exit opportunity).
- A 2024 debut "exits" by missing one season (2025); a 2023 debut who missed 2024 and came back in 2025 is a gap,
  not an exit (6 prospects). So exits are early and partly provisional.

Analyses (position groups scope the study and read results; models stratify by group, never use it as a covariate):
1. Kaplan–Meier by round band and by the raw-joules tail (top `TAIL` within group × draft class, as in step 6).
2. Cox, stratified by position group (risk sets within a group, `eta.role`), draft control (`outcomes.draft_control`)
   plus one measure at a time: the tail flag, then the within-cell z-score. Hazard ratio < 1 = stays longer.
3. Observed vs expected exits in each measure's tail: expected from a draft-only stratified Cox, as the sum of each
   tail prospect's cumulative hazard at their own time (O/E, the one-sample log-rank idea), permutation p with the
   measure shuffled within group × class.

η is not used: it exists only for players who already play, and step 5 found it adds nothing. Writes
reports/tenure.md (aggregates only) and reports/figures/tenure.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.outcomes.mismatch import MEASURES, TAIL, tail_flag

TEST_MEASURES = ["TEE_z_abs", "ax_explosive", "a10", "mass_kg"]
N_PERM, SEED = 2000, 0


def at_risk(prospects: pl.DataFrame, last_season: int) -> pl.DataFrame:
    """Prospects who played and debuted before last_season, with `duration` (seasons from debut to last play) and
    `event` (exited)."""
    return prospects.filter((pl.col("seasons_active") > 0) & (pl.col("first_active") < last_season)).with_columns(
        (pl.col("last_active") - pl.col("first_active") + 1).alias("duration"),
        pl.col("exited").cast(pl.Int32).alias("event"),
    )


def within_z(d: pl.DataFrame, measure: str, cell: str = "cell") -> pl.Expr:
    """Within-cell z-score of a measure."""
    m = pl.col(measure)
    return ((m - m.mean().over(cell)) / m.std().over(cell)).alias(f"z_{measure}")


def km_table(d: pl.DataFrame, by: str) -> pl.DataFrame:
    """Kaplan–Meier share still in the league after 1 and 2 seasons, per level of `by`, with at-risk counts."""
    from lifelines import KaplanMeierFitter

    rows = []
    for (level,), x in d.group_by([by], maintain_order=True):
        km = KaplanMeierFitter().fit(x["duration"].to_numpy(), x["event"].to_numpy())
        s = km.survival_function_at_times([1, 2]).to_numpy()
        rows.append(
            dict(level=str(level), n=x.height, exits=int(x["event"].sum()), stay_after_1=s[0], stay_after_2=s[1])
        )
    return pl.DataFrame(rows).sort("level")


def cox(d: pl.DataFrame, covariates, strata: str = "group"):
    """Cox PH stratified by group on the given covariates (complete rows)."""
    from lifelines import CoxPHFitter

    sub = d.select("duration", "event", strata, *covariates).drop_nulls().to_pandas()
    return CoxPHFitter().fit(sub, "duration", "event", strata=[strata]), len(sub)


def cox_row(d: pl.DataFrame, control, x: str, label: str) -> dict:
    """Hazard ratio of x (95% CI, p) beyond the draft control, stratified by group."""
    fit, n = cox(d, [*control, x])
    s = fit.summary.loc[x]
    return dict(
        model=label,
        n=n,
        exits=int(d.drop_nulls(x)["event"].sum()),
        hr=float(s["exp(coef)"]),
        lo95=float(s["exp(coef) lower 95%"]),
        hi95=float(s["exp(coef) upper 95%"]),
        p=float(s["p"]),
    )


def expected_exits(d: pl.DataFrame, control) -> np.ndarray:
    """Each prospect's cumulative hazard at their own duration from a draft-only Cox stratified by group (the
    expected number of exits they contribute under the draft model)."""
    fit, _ = cox(d, control)
    pdf = d.select("duration", "event", "group", *control).to_pandas()
    ch = fit.predict_cumulative_hazard(pdf, times=[1, 2, 3])
    t = d["duration"].to_numpy()
    return np.array([ch.iloc[min(ti, 3) - 1, i] for i, ti in enumerate(t)])


def oe_test(d: pl.DataFrame, measure: str, expected: str = "expected", n_perm=N_PERM, seed=SEED) -> dict:
    """Observed vs expected exits in the measure's tail; permutation p (two-sided, on observed − expected) shuffles
    tail membership within cells, keeping each cell's tail size among the at-risk prospects."""
    x = d.drop_nulls(measure)
    tail = x["tail"].to_numpy()
    obs, exp = x["event"].to_numpy(), x[expected].to_numpy()
    oe = obs[tail].sum() / exp[tail].sum()
    cells = x["cell"].cast(pl.Categorical).to_physical().to_numpy()
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        t = np.zeros(len(x), bool)
        for c in np.unique(cells):
            idx = np.flatnonzero(cells == c)
            t[rng.choice(idx, tail[idx].sum(), replace=False)] = True
        null[b] = obs[t].sum() - exp[t].sum()
    diff = obs[tail].sum() - exp[tail].sum()
    p = (np.sum(np.abs(null - null.mean()) >= abs(diff - null.mean())) + 1) / (n_perm + 1)
    return dict(
        measure=measure,
        n_tail=int(tail.sum()),
        observed=int(obs[tail].sum()),
        expected=float(exp[tail].sum()),
        oe=float(oe),
        oe_rest=float(obs[~tail].sum() / exp[~tail].sum()),
        p_perm=float(p),
    )


def plot_tenure(d: pl.DataFrame, path) -> None:
    """Kaplan–Meier: by round band (left), raw-joules tail vs rest (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, by, title in [
        (axes[0], "round_band", "By draft round"),
        (axes[1], "raw_tail", "Raw joules: top 20% within group × class"),
    ]:
        for (level,), x in sorted(d.drop_nulls(by).group_by([by]), key=lambda kv: str(kv[0][0])):
            KaplanMeierFitter().fit(x["duration"], x["event"], label=f"{level} (n {x.height})").plot_survival_function(
                ax=ax, ci_show=by == "raw_tail"
            )
        ax.set_xlabel("seasons since debut")
        ax.set_title(title)
        ax.set_xlim(0, 3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("share still playing (Kaplan–Meier)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT
    from eeg.outcomes.mismatch import prospect_frame

    cfg = config.load()
    control = cfg["outcomes"]["draft_control"]
    last = cfg["outcomes"]["last_season"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    out = pl.read_parquet(processed / "outcomes.parquet").select(
        "player_id", "seasons_active", "first_active", "last_active", "exited", pl.col("round_band").cast(pl.String)
    )
    base = prospect_frame(processed, interim)
    # tails and z-scores over the whole class (as in step 6), then restricted to the at-risk set
    for m in MEASURES:
        flags = tail_flag(base, m).select("player_id", pl.col("tail").alias(f"tail_{m}"))
        base = base.join(flags, on="player_id", how="left").with_columns(within_z(base, m))
    d = at_risk(base.join(out, on="player_id"), last).with_columns(
        pl.when(pl.col("tail_TEE_z_abs").is_null())
        .then(None)
        .otherwise(pl.when(pl.col("tail_TEE_z_abs")).then(pl.lit("top 20%")).otherwise(pl.lit("rest")))
        .alias("raw_tail"),
        *(pl.col(f"tail_{m}").cast(pl.Int32).alias(f"tail_{m}_i") for m in MEASURES),
    )

    by_group = d.group_by("group").agg(pl.len().alias("at_risk"), pl.col("event").sum().alias("exits")).sort("group")
    km_band = km_table(d, "round_band")
    km_raw = km_table(d.drop_nulls("raw_tail"), "raw_tail")
    draft_fit, n_draft = cox(d, control)
    draft_hr = pl.DataFrame(
        [
            dict(covariate=c, hr=float(s["exp(coef)"]), lo95=float(s["exp(coef) lower 95%"]), hi95=float(s["exp(coef) upper 95%"]), p=float(s["p"]))
            for c, s in draft_fit.summary.iterrows()
        ]
    )  # fmt: skip
    rows = []
    for m in TEST_MEASURES:
        rows.append(cox_row(d, control, f"tail_{m}_i", f"{m}: top 20%"))
        rows.append(cox_row(d, control, f"z_{m}", f"{m}: per SD within cell"))
    hrs = pl.DataFrame(rows)
    d = d.with_columns(pl.Series("expected", expected_exits(d, control)))
    oe = pl.DataFrame([oe_test(d.with_columns(pl.col(f"tail_{m}").alias("tail")), m) for m in TEST_MEASURES])
    plot_tenure(d, reports / "figures" / "tenure.png")

    report = f"""# Tenure: does the explosion tail stay longer?

Generated by `uv run python -m eeg.survival.tenure`. Aggregates only. Secondary outcome (open decision 3).

- **At risk:** prospects who played and debuted before {last}: {d.height} ({int(d["event"].sum())} exits). Time = seasons from debut to the
  last season with a play; exit = no play in any later season through {last}; still playing in {last} = censored.
- **Provisional exits:** a 2024 debut exits by missing 2025 alone; 6 prospects who missed a season and came back
  are not exits. Three classes give at most three seasons, so this is early tenure.
- Tails as in step 6 (top {TAIL:.0%} of a measure within group × draft class, over the whole class). Cox models
  are stratified by position group (risk sets within a group) with the draft control ({" + ".join(control)}).

{_md(by_group)}

## Kaplan–Meier

Share still playing after 1 and 2 seasons. Figure: `reports/figures/tenure.png`.

{_md(km_band, 2)}

Raw joules (`TEE_z_abs`), top {TAIL:.0%} within group × class against the rest:

{_md(km_raw, 2)}

## Cox, stratified by group

Draft only ({n_draft} prospects): hazard ratio per pick and for undrafted.

{_md(draft_hr, 3)}

Each measure beyond the draft, one at a time: its top-{TAIL:.0%} flag, then its within-cell z-score. HR < 1 = stays
longer. Eight models: read p near 0.05 as weak.

{_md(hrs, 3)}

## Observed vs expected exits in the tail

Expected = each tail prospect's cumulative hazard at their own time under the draft-only model. `oe` < 1: fewer
exits than the draft slot predicts. `p_perm`: two-sided on observed − expected, tail membership
shuffled within group × class ({N_PERM}). Four measures: read p near 0.05 as weak.

{_md(oe, 2)}
"""
    (reports / "tenure.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
