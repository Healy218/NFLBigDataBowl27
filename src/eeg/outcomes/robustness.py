"""Robustness of the headline results before the write-up ("True ball is an explosion").

    uv run python -m eeg.outcomes.robustness

The headline (step 6, eeg.outcomes.mismatch): the top 20% by raw explosive energy (TEE_z_abs) within position
group × draft class beat their draft-slot playing time more often than the rest. Checks, each as the share of the
tail beating its slot against the rest, with the permutation p of eeg.outcomes.mismatch.tail_test (measure
shuffled within group × class):

1. Tail cut-off: top 10%, 20%, 30%.
2. Expectation model: pick + undrafted (in use), log pick + undrafted, round bands, and a nonparametric one (mean
   outcome of the `K_NEAREST` nearest picks in the same group × class, the prospect left out).
3. Each draft class alone, and the pooled test with each position group left out.
4. Which channels carry it: each absolute energy channel of TEE alone (z within Combine year), the 10 Hz tracked
   40 (E_40, from the fitted v_max of the trace) against the result-based jumps and bench, and the tracked
   change-of-direction work (sparse).
5. The DL chain (Combine start → get-off → pressure rate, beyond the draft and plays, centred on NFL position) in
   each draft class.

Many checks on one dataset: read them for consistency of direction and size, not as new discoveries. Writes
reports/robustness.md (aggregates only) and reports/figures/robustness.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.outcomes.mismatch import slot_expectation, tail_test

K_NEAREST = 6
N_PERM = 2000
GROUPS = ["WR", "TE", "DB", "OL", "DL"]
CHANNELS = {
    "z_E_40": "40 energy (10 Hz tracked v_max)",
    "z_E_vert": "vertical jump energy (result)",
    "z_E_broad": "broad jump energy (result)",
    "z_W_bench": "bench work (result)",
    "z_W_cod_shuttle": "shuttle COD work (tracked, sparse)",
    "z_W_cod_cone": "3-cone COD work (tracked, sparse)",
}
RESULTS = ["z_E_vert", "z_E_broad", "z_W_bench"]
BANDS = ["band_2-3", "band_4-5", "band_6-7"]


def knn_expectation(d: pl.DataFrame, y: str, k: int = K_NEAREST, cell: str = "cell") -> pl.DataFrame:
    """Leave-one-out mean of y over the k nearest overall picks in the same cell (undrafted at their shared pick):
    adds `{y}_exp` and `{y}_resid`."""
    parts = []
    for _, x in d.group_by(cell, maintain_order=True):
        pick, yv = x["pick"].to_numpy().astype(float), x[y].to_numpy().astype(float)
        dist = np.abs(pick[:, None] - pick[None, :])
        np.fill_diagonal(dist, np.inf)
        near = np.argsort(dist, axis=1, kind="stable")[:, : min(k, len(yv) - 1)]
        exp = yv[near].mean(axis=1)
        parts.append(x.with_columns(pl.Series(f"{y}_exp", exp), pl.Series(f"{y}_resid", yv - exp)))
    return pl.concat(parts)


def with_expectations(d: pl.DataFrame, y: str, control) -> pl.DataFrame:
    """Residual columns `{y}_resid_<model>` for every expectation model; `{y}_resid` is the model in use."""
    models = {
        "pick": control,
        "log_pick": ["log_pick", "udfa"],
        "bands": [*BANDS, "udfa"],
    }
    out = d
    for name, xs in models.items():
        r = slot_expectation(d, y, xs).select("player_id", pl.col(f"{y}_resid").alias(f"{y}_resid_{name}"))
        out = out.join(r, on="player_id")
    r = knn_expectation(d, y).select("player_id", pl.col(f"{y}_resid").alias(f"{y}_resid_knn"))
    return out.join(r, on="player_id").with_columns(pl.col(f"{y}_resid_pick").alias(f"{y}_resid"))


def row(d: pl.DataFrame, check: str, variant: str, measure: str, resid: str, tail: float = 0.2) -> dict:
    """One check: tail_test on the given residual, labelled."""
    t = tail_test(d, measure, resid, tail=tail, n_perm=N_PERM)
    keep = ("n_tail", "beat_tail", "lo95", "hi95", "beat_rest", "p_share")
    return dict(check=check, variant=variant, outcome=resid.split("_resid")[0]) | {k: t[k] for k in keep}


def tee_channels(processed) -> pl.DataFrame:
    """Absolute TEE channel z-scores per prospect, plus the result-based composite (mean of available jumps and
    bench z, at least two)."""
    tee = pl.read_parquet(processed / "tee.parquet").select("player_id", *CHANNELS)
    n = pl.sum_horizontal(pl.col(c).is_not_null() for c in RESULTS)
    res = pl.when(n >= 2).then(pl.mean_horizontal(*RESULTS)).otherwise(None)
    return tee.with_columns(res.alias("z_results"))


def dl_chain_by_class(chain: pl.DataFrame, control) -> pl.DataFrame:
    """Per draft class: betas of get-off and of a10 on pressure rate beyond the draft and plays, centred on NFL
    position."""
    from eeg.outcomes.output import betas_long

    rows = []
    for (yr,), x in chain.group_by(["draft_year"], maintain_order=True):
        for xs, label in [(["log_get_off"], "pressure ~ get-off"), (["log_a10"], "pressure ~ a10")]:
            for r in betas_long(x, "pressure_rate", [*control, "log_plays", *xs], "pos", xs, label):
                rows.append(r | {"draft_year": yr})
        for r in betas_long(x, "log_get_off", ["log_a10"], "pos", ["log_a10"], "get-off ~ a10"):
            rows.append(r | {"draft_year": yr})
    return (
        pl.DataFrame(rows)
        .sort("outcome", "draft_year")
        .select("outcome", "draft_year", "variable", "n", "beta", "lo95", "hi95")
    )


def plot_robustness(t: pl.DataFrame, path) -> None:
    """Forest plot: share of the raw-joules tail beating its draft slot (95% CI) in each check, rest marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.32 * t.height + 1.2))
    y = np.arange(t.height)
    colour = ["C3" if c == "channel" else "C0" for c in t["check"]]
    ax.errorbar(
        t["beat_tail"], y, xerr=[t["beat_tail"] - t["lo95"], t["hi95"] - t["beat_tail"]], fmt="none", ecolor=colour
    )
    ax.scatter(t["beat_tail"], y, c=colour, s=18, zorder=3, label="explosive tail")
    ax.scatter(t["beat_rest"], y, marker="|", color="grey", s=60, zorder=3, label="rest of the class")
    ax.axvline(0.5, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y, [f"{c}: {v} ({o})" for c, v, o in zip(t["check"], t["variant"], t["outcome"])], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("share playing more than the draft slot predicts (95% CI)")
    ax.set_title("The explosive tail beats its draft slot across checks")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT
    from eeg.outcomes.mismatch import prospect_frame
    from eeg.outcomes.output import dl_chain

    cfg = config.load()
    control = cfg["outcomes"]["draft_control"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    d = prospect_frame(processed, interim).join(tee_channels(processed), on="player_id", how="left")
    for y in ("snaps", "starts"):
        d = with_expectations(d, y, control)
    raw = "TEE_z_abs"

    rows = []
    for tail in (0.1, 0.2, 0.3):
        for y in ("snaps", "starts"):
            rows.append(row(d, "cut-off", f"top {tail:.0%}", raw, f"{y}_resid", tail))
    for model in ("pick", "log_pick", "bands", "knn"):
        for y in ("snaps", "starts"):
            rows.append(row(d, "expectation", model, raw, f"{y}_resid_{model}"))
    for yr in (2023, 2024, 2025):
        for y in ("snaps", "starts"):
            rows.append(row(d.filter(pl.col("draft_year") == yr), "class", str(yr), raw, f"{y}_resid"))
    for g in GROUPS:
        rows.append(row(d.filter(pl.col("group") != g), "without group", g, raw, "snaps_resid"))
    for c in [*CHANNELS, "z_results"]:
        for y in ("snaps", "starts"):
            rows.append(row(d, "channel", c.removeprefix("z_"), c, f"{y}_resid"))
    checks = pl.DataFrame(rows)

    lab = pl.read_parquet(processed / "lab_burst.parquet").select("player_id", pl.col("a10").log().alias("log_a10"))
    chain = dl_chain(
        pl.read_parquet(processed / "production_seasons.parquet"),
        d.select("player_id", "group", "pos", "draft_year", *control).join(lab, on="player_id", how="left"),
    )
    dl = dl_chain_by_class(chain, control)
    corr = d.select(*CHANNELS, "z_results", "mass_kg").drop_nulls(["z_E_40", "z_results"])
    r_40 = float(np.corrcoef(corr["z_E_40"], corr["z_results"])[0, 1])

    plot_robustness(
        checks.filter(pl.col("outcome") == "snaps", ~pl.col("variant").is_in(["W_cod_shuttle", "W_cod_cone"])),
        reports / "figures" / "robustness.png",
    )

    report = f"""# Robustness: the explosive tail and the DL chain

Generated by `uv run python -m eeg.outcomes.robustness`. Aggregates only. The headline (`reports/mismatch.md`): the
top 20% by raw explosive energy (`TEE_z_abs`) within position group × draft class beat their draft-slot playing
time more often than the rest. Each row: share of the tail beating its slot (`beat_tail`, Wilson 95% interval)
against the rest (`beat_rest`), and the permutation p of that share (measure shuffled within group × class,
{N_PERM}). Many checks on one dataset: read them for consistency, not as new discoveries. Figure (snaps):
`reports/figures/robustness.png`.

## 1–3. Cut-off, expectation model, class, groups left out (raw joules)

Expectation models: `pick` = pick + undrafted per group with class intercepts (in use); `log_pick` = log pick +
undrafted; `bands` = round bands (2–3, 4–5, 6–7, undrafted; round 1 the reference); `knn` = mean of the
{K_NEAREST} nearest picks in the same group × class, the prospect left out.

{_md(checks.filter(pl.col("check") != "channel"), 3)}

## 4. Which energy channels carry it

Each absolute channel's z-score (within Combine year) as the measure, tail top 20% within group × class.
`E_40` is the 10 Hz tracked channel (½ m v_max² plus drag work, v_max fitted to the 40 trace); `results` is the
mean of the jump and bench z-scores (at least two). Within the prospects with both, r(E_40, results) =
{r_40:.2f}. The change-of-direction channels are tracked but measured on few prospects.

{_md(checks.filter(pl.col("check") == "channel"), 3)}

## 5. The DL chain by draft class

Pressure rate beyond the draft and log plays, and get-off on a10, centred on NFL position (DE, OLB, DT, NT), one
class at a time (pooled: get-off β −0.47, a10 0.36, get-off ~ a10 −0.44; `reports/output.md`).

{_md(dl, 2)}
"""
    (reports / "robustness.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
