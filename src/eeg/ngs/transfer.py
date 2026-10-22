"""What predicts transfer: the Combine engine against η, game output and DL get-off, within position group.

    uv run python -m eeg.ngs.transfer

Targets, per prospect, log scale:
- `eta_career` (η_KE, plays-weighted over qualified seasons) and `eta_first` (first qualified season). Every
  predictor here is measured at the Combine, before any season, so the career value is not look-ahead in this
  step: it is simply the more reliable read of the same quantity.
- `field_speed`, `field_burst`: realized game output (`s_peak` and `burst_a` season values, `games.season_statistic`, geometric mean
  over qualified seasons). η_KE = (field speed / lab speed)², so contrasting the two separates "does the engine show up in games"
  from "what fraction of it shows up".
- `get_off` (DL only): mean get-off time over qualified seasons (lower is faster).

Predictors (Combine only): kinetic profile axes and the Kinetic Anomaly, TEE_z_rel, the start (a10, `lab_burst`),
the 40's shape (split / forty), top speed (tracked v_max), mass, agility times (shuttle, 3-cone; sparse), and the
NGS scores as context.

Analyses (position groups scope and read results only; every statistic is within group):
1. univariate r with 95% Fisher-z intervals, values centred on group means;
2. a compact model per target (profile axes, a10, mass, college production): standardized betas, in-sample R²
   and cross-validated R² beyond the group mean (repeated K-fold; group means and slopes fit on training folds);
3. get-off role check: the same within DL sub-position (NFL position), which absorbs edge vs interior alignment.

Writes reports/transfer.md (aggregates only) and reports/figures/transfer.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.ngs.eta_readout import within_r

PREDICTORS = [
    "ax_explosive", "ax_speed_vs_bounce", "ax_up_vs_out", "log_anomaly", "TEE_z_rel", "log_a10", "split_share",
    "log_v_max", "log_mass", "log_shuttle", "log_cone", "ngs_college_production_score", "ngs_athleticism_score",
]  # fmt: skip
TARGETS = ["eta_career", "eta_first", "field_speed", "field_burst", "get_off"]
COMPACT = ["ax_explosive", "ax_speed_vs_bounce", "ax_up_vs_out", "log_a10", "log_mass", "ngs_college_production_score"]
GET_OFF_MODEL = ["log_a10", "log_mass", "ax_explosive", "ngs_college_production_score"]
FOLDS, REPEATS, SEED = 5, 20, 0


def univariate(d: pl.DataFrame, targets=TARGETS, predictors=PREDICTORS, group: str = "group") -> pl.DataFrame:
    """Long table: target, variable, n, r, lo95, hi95 (within group)."""
    rows = []
    for y in targets:
        for x in predictors:
            sub = d.select(group, x, y).drop_nulls()
            if sub.height > sub[group].n_unique() + 3:
                rows.append(dict(target=y, **within_r(sub, x, y, group)))
    return pl.DataFrame(rows)


def _centre(x: np.ndarray, y: np.ndarray, g: np.ndarray, ref=None):
    """Centre x and y on group means; means come from ref = (x, y, g) if given (training fold)."""
    rx, ry, rg = ref if ref is not None else (x, y, g)
    xc, yc = x.copy(), y.copy()
    for k in np.unique(g):
        m = rg == k
        xc[g == k] -= rx[m].mean(0) if m.any() else rx.mean(0)
        yc[g == k] -= ry[m].mean() if m.any() else ry.mean()
    return xc, yc


def fit_within(d: pl.DataFrame, y: str, xs, group: str = "group") -> dict:
    """OLS on group-centred values: standardized betas with 95% intervals, R² within group."""
    sub = d.select(group, y, *xs).drop_nulls()
    g, yv = sub[group].to_numpy(), sub[y].to_numpy().astype(float)
    x = sub.select(xs).to_numpy().astype(float)
    xc, yc = _centre(x, yv, g)
    beta, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    resid = yc - xc @ beta
    dof = len(yc) - len(xs) - len(np.unique(g))
    se = np.sqrt(np.diag(np.linalg.inv(xc.T @ xc)) * (resid @ resid) / dof)
    scale = xc.std(0) / yc.std()
    out = dict(target=y, n=len(yc), r2=1 - (resid @ resid) / (yc @ yc))
    for i, c in enumerate(xs):
        out |= {
            f"b_{c}": beta[i] * scale[i],
            f"lo_{c}": (beta[i] - 1.96 * se[i]) * scale[i],
            f"hi_{c}": (beta[i] + 1.96 * se[i]) * scale[i],
        }
    return out


def cv_r2(d: pl.DataFrame, y: str, xs, group: str = "group", folds=FOLDS, repeats=REPEATS, seed=SEED) -> float:
    """Out-of-sample R² beyond the group mean: 1 − SSE(model) / SSE(training group mean), pooled over
    repeated K-fold splits. Group means and slopes come from the training folds only."""
    sub = d.select(group, y, *xs).drop_nulls()
    g, yv = sub[group].to_numpy(), sub[y].to_numpy().astype(float)
    x = sub.select(xs).to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    sse_m = sse_0 = 0.0
    for _ in range(repeats):
        fold = rng.permutation(len(yv)) % folds
        for k in range(folds):
            tr, te = fold != k, fold == k
            ref = (x[tr], yv[tr], g[tr])
            xtr, ytr = _centre(x[tr], yv[tr], g[tr])
            beta, *_ = np.linalg.lstsq(xtr, ytr, rcond=None)
            xte, yte = _centre(x[te], yv[te], g[te], ref)
            sse_m += ((yte - xte @ beta) ** 2).sum()
            sse_0 += (yte**2).sum()
    return 1 - sse_m / sse_0


def prospect_table(processed, interim, stat: str) -> pl.DataFrame:
    """One row per prospect: group, sub-position, predictors and targets (logs where noted)."""
    cs = pl.read_parquet(interim / "combine_si.parquet")
    base = cs.select(
        "player_id",
        pl.col("pos_group").alias("group"),
        "pos",
        pl.col("mass_kg").log().alias("log_mass"),
        (pl.col("ten_split_s") / pl.col("forty_s")).alias("split_share"),
        pl.col("shuttle_s").log().alias("log_shuttle"),
        pl.col("cone_s").log().alias("log_cone"),
        "ngs_college_production_score",
        "ngs_athleticism_score",
    )
    lab = pl.read_parquet(processed / "lab_burst.parquet").select(
        "player_id", pl.col("a10").log().alias("log_a10"), pl.col("v_max_mean").log().alias("log_v_max")
    )
    prof = pl.read_parquet(processed / "profiles.parquet").select(
        "player_id",
        "ax_explosive",
        "ax_speed_vs_bounce",
        "ax_up_vs_out",
        pl.col("anomaly_d2").log().alias("log_anomaly"),
    )
    tee = pl.read_parquet(processed / "tee.parquet").select("player_id", "TEE_z_rel")
    eta = pl.read_parquet(processed / "eta_career.parquet").select(
        "player_id", pl.col("eta_ke_career").log().alias("eta_career"), pl.col("eta_ke_first").log().alias("eta_first")
    )
    field = (
        pl.read_parquet(processed / "field_seasons.parquet")
        .filter("qualified")
        .group_by("player_id")
        .agg(
            pl.col(f"s_peak_{stat}").log().mean().alias("field_speed"),
            pl.col(f"burst_a_{stat}").log().mean().alias("field_burst"),
        )
    )
    get_off = (
        pl.read_parquet(processed / "production_seasons.parquet")
        .filter(pl.col("qualified") & (pl.col("group") == "DL"))
        .drop_nulls("get_off")
        .group_by("player_id")
        .agg(pl.col("get_off").log().mean())
    )
    out = base
    for t in (lab, prof, tee, eta, field, get_off):
        out = out.join(t, on="player_id", how="left")
    return out.sort("player_id")


def plot_transfer(uni: pl.DataFrame, d: pl.DataFrame, path) -> None:
    """Left: within-group r of each predictor with η (career) and with field speed. Right: DL get-off against a10."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    names = [p for p in PREDICTORS if p not in ("log_shuttle", "log_cone")]
    ypos = np.arange(len(names))
    for off, (tgt, colour, label) in zip(
        (-0.2, 0.2), [("eta_career", "C3", "η_KE (transfer)"), ("field_speed", "C0", "game top speed (output)")]
    ):
        u = uni.filter(pl.col("target") == tgt).select("variable", "r", "lo95", "hi95")
        u = pl.DataFrame({"variable": names}).join(u, on="variable", how="left")
        ax0.errorbar(
            u["r"],
            ypos + off,
            xerr=[u["r"] - u["lo95"], u["hi95"] - u["r"]],
            fmt="o",
            color=colour,
            ecolor=colour,
            alpha=0.8,
            ms=4,
            label=label,
        )
    ax0.axvline(0, color="grey", lw=0.6)
    ax0.set_yticks(ypos, names, fontsize=8)
    ax0.invert_yaxis()
    ax0.set_xlabel("r within position group (95% CI)")
    ax0.set_title("The Combine predicts game speed, not transfer")
    ax0.legend(fontsize=8, loc="lower right")
    dl = d.filter(pl.col("group") == "DL").drop_nulls(["get_off", "log_a10"])
    for i, (pos, x) in enumerate(dl.group_by("pos", maintain_order=True)):
        ax1.scatter(np.exp(x["log_a10"]), np.exp(x["get_off"]), s=18, alpha=0.8, color=f"C{i}", label=pos[0])
    ax1.set_xlabel("a10 from the official 10-yard split (m/s²)")
    ax1.set_ylabel("game get-off time (s, lower is faster)")
    ax1.set_title("DL: the Combine start predicts the game start")
    ax1.legend(fontsize=8, title="NFL position", title_fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT

    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    d = prospect_table(processed, interim, config.load()["games"]["season_statistic"])
    uni = univariate(d)
    wide = uni.pivot(on="target", index="variable", values="r").select("variable", *TARGETS)
    sig = uni.filter((pl.col("lo95") > 0) | (pl.col("hi95") < 0)).sort("target", "r")

    models = []
    for y in ["eta_career", "eta_first", "field_speed", "field_burst"]:
        models.append(fit_within(d, y, COMPACT) | {"cv_r2": cv_r2(d, y, COMPACT)})
    compact = pl.DataFrame(models).select("target", "n", "r2", "cv_r2", *(f"b_{c}" for c in COMPACT))

    dl = d.filter(pl.col("group") == "DL")
    dl_pos = dl.with_columns(pl.col("pos").alias("group"))
    go = pl.DataFrame(
        [
            fit_within(dl, "get_off", GET_OFF_MODEL)
            | {"cv_r2": cv_r2(dl, "get_off", GET_OFF_MODEL), "centred_on": "DL"},
            fit_within(dl_pos, "get_off", GET_OFF_MODEL)
            | {"cv_r2": cv_r2(dl_pos, "get_off", GET_OFF_MODEL), "centred_on": "NFL position"},
        ]
    )
    go_long = pl.DataFrame(
        [
            dict(centred_on=r["centred_on"], variable=c, beta=r[f"b_{c}"], lo95=r[f"lo_{c}"], hi95=r[f"hi_{c}"])
            for r in go.iter_rows(named=True)
            for c in GET_OFF_MODEL
        ]
    )
    go_r = pl.DataFrame(
        [
            within_r(dl_pos, x, "get_off")
            for x in ["log_a10", "log_mass", "log_v_max", "ax_explosive", "ngs_college_production_score"]
        ]
    )
    plot_transfer(uni, d, reports / "figures" / "transfer.png")

    report = f"""# What predicts transfer

Generated by `uv run python -m eeg.ngs.transfer`. Per prospect, Combine predictors against η_KE (transfer), game
output and DL get-off. Every statistic is within position group (values centred on group means); groups only read
results. Aggregates only. Predictors are measured at the Combine, so the career η is not look-ahead here.

- Prospects with η: {d["eta_career"].is_not_null().sum()} (profiled {d.drop_nulls(["eta_career", "ax_explosive"]).height});
  with game output: {d["field_speed"].is_not_null().sum()}; DL with get-off: {dl["get_off"].is_not_null().sum()}.
- Agility times are sparse (shuttle / 3-cone on ~40% of prospects), so their rows rest on ~100 prospects (DL ~28).

## Univariate: r within group

{_md(wide, 2)}

Intervals that exclude 0 (95%, Fisher z):

{_md(sig.select("target", "variable", "n", "r", "lo95", "hi95"), 2)}

## Compact model per target

Predictors {", ".join(f"`{c}`" for c in COMPACT)}, on prospects with all of them. Standardized betas; `r2` in
sample, `cv_r2` out of sample beyond the group mean ({FOLDS}-fold × {REPEATS}; ≤ 0 means no predictive value).

{_md(compact, 2)}

## DL get-off

Model {", ".join(f"`{c}`" for c in GET_OFF_MODEL)}, centred on the DL group and, as a role check, on the NFL
position (DT, NT, DE, OLB: edge and interior rushers line up differently).

{_md(go.select("centred_on", "n", "r2", "cv_r2"), 2)}

{_md(go_long, 2)}

Univariate within NFL position:

{_md(go_r, 2)}

Figure: `reports/figures/transfer.png`.
"""
    (reports / "transfer.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
