"""Under- and over-drafted: do the most explosive prospects outplay their draft slot?

    uv run python -m eeg.outcomes.mismatch

Step 5 found that, on average across all prospects, the Combine adds nothing to playing time beyond the draft
(a linear slope). Mismatches live in the tail, so this asks a different question: among the most explosive
prospects of a group and class, how many play more than their draft slot predicts?

- Outcome: scrimmage snaps per available season (all offensive / defensive snaps, regular season, from career
  totals; WR and TE are not limited to routes here), log1p. Games started per season as a second outcome.
- Draft-slot expectation: per position group, OLS of the outcome on the draft control (`outcomes.draft_control`:
  pick + undrafted) with one intercept per draft class, leave-one-out (the prospect's own snaps don't shape their
  expectation) and clipped to the group's observed range. Residual > 0: plays more than the slot predicts. The
  group absorbs "linemen play every snap" and "skill players debut later"; the class absorbs seasons available.
- Explosion measures (Combine only): the explosive profile axis (size-adjusted output), raw joules (TEE_z_abs),
  joules per kg (TEE_z_rel), the start (a10) and, as the check on raw joules, body mass.
- Tail: top `TAIL` of the measure within group × class. Per measure: share of the tail beating its slot (Wilson
  95% interval) against the rest, mean residual in the tail against the rest (as a ratio of snaps), and a
  permutation p-value (the measure shuffled within group × class, so the null keeps every cell's draft
  structure). Five measures are tested, so a single p near 0.05 is weak.
- Dose-response: mean residual by within-cell quintile of each measure (is it a tail effect or a slope?).

Position groups scope the study and read results; every comparison is within group × class. Writes
data/processed/mismatch.parquet (named, not committed), reports/mismatch.md (aggregates only) and
reports/figures/mismatch.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl

TAIL = 0.2
N_PERM, SEED = 5000, 0
MEASURES = {
    "ax_explosive": "explosive axis (size-adjusted)",
    "TEE_z_abs": "raw joules (TEE_z_abs)",
    "TEE_z_rel": "joules per kg (TEE_z_rel)",
    "a10": "start (a10)",
    "mass_kg": "body mass (check)",
}
OUTCOMES = {"snaps": "scrimmage snaps per season", "starts": "games started per season"}


def slot_expectation(d: pl.DataFrame, y: str, control, group: str = "group", block: str = "draft_year") -> pl.DataFrame:
    """Leave-one-out expectation of y from the draft control, fitted per position group with one intercept per
    block (draft class) and shared slopes, clipped to the group's observed range: adds `{y}_exp`, `{y}_resid`.
    Separate slopes per group × class (~14 TE a class) let leave-one-out predictions run far outside the data."""
    parts = []
    for _, x in d.group_by(group, maintain_order=True):
        blocks = x[block].unique().sort().to_list()
        X = np.column_stack(
            [
                *((x[block] == b).cast(pl.Float64).to_numpy() for b in blocks),
                *(x[c].to_numpy().astype(float) for c in control),
            ]
        )
        yv = x[y].to_numpy().astype(float)
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        h = np.einsum("ij,ji->i", X, np.linalg.pinv(X.T @ X) @ X.T)
        loo = yv - (yv - X @ beta) / np.clip(1 - h, 1e-9, None)
        exp = np.clip(loo, yv.min(), yv.max())
        parts.append(x.with_columns(pl.Series(f"{y}_exp", exp), pl.Series(f"{y}_resid", yv - exp)))
    return pl.concat(parts)


def tail_flag(d: pl.DataFrame, measure: str, cell: str = "cell", tail: float = TAIL) -> pl.DataFrame:
    """Rows with the measure, with `q` (within-cell rank from the top / count) and `tail` (q ≤ tail)."""
    x = d.drop_nulls(measure)
    q = pl.col(measure).rank("ordinal", descending=True).over(cell) / pl.col(measure).count().over(cell)
    return x.with_columns(q.alias("q")).with_columns((pl.col("q") <= tail).alias("tail"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval of a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


def _perm_tail_means(resid: np.ndarray, cells: np.ndarray, k: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """Tail mean residual and tail share with residual > 0 under random tails of the same size per cell (k per
    row's cell): array (n_perm, 2)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, 2))
    for b in range(n_perm):
        order = np.lexsort((rng.random(len(resid)), cells))
        c = cells[order]
        start = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1]
        pos = np.arange(len(c)) - np.repeat(start, np.diff(np.r_[start, len(c)]))
        pick = resid[order][pos < k[order]]
        out[b] = pick.mean(), (pick > 0).mean()
    return out


def tail_test(
    d: pl.DataFrame, measure: str, resid: str, cell: str = "cell", tail: float = TAIL, n_perm=N_PERM, seed=SEED
) -> dict:
    """Tail vs rest on the slot residual: shares beating the slot (Wilson CI), mean residuals, their difference
    as a ratio of (1 + outcome), and a two-sided permutation p for the tail mean (tails resampled per cell)."""
    x = tail_flag(d, measure, cell, tail).drop_nulls(resid)
    t, r = x.filter("tail"), x.filter(~pl.col("tail"))
    beat_t, beat_r = int((t[resid] > 0).sum()), int((r[resid] > 0).sum())
    lo, hi = wilson(beat_t, t.height)
    k = x.with_columns(pl.col("tail").sum().over(cell).alias("k"))["k"].to_numpy()
    cells = x[cell].cast(pl.Categorical).to_physical().to_numpy()
    null = _perm_tail_means(x[resid].to_numpy(), cells, k, n_perm, seed)
    obs, centre = t[resid].mean(), x[resid].mean()
    share, share0 = beat_t / t.height, float((x[resid] > 0).mean())
    p = (np.sum(np.abs(null[:, 0] - centre) >= abs(obs - centre)) + 1) / (n_perm + 1)
    p_share = (np.sum(np.abs(null[:, 1] - share0) >= abs(share - share0)) + 1) / (n_perm + 1)
    return dict(
        measure=measure,
        n_tail=t.height,
        n_rest=r.height,
        beat_tail=beat_t / t.height,
        lo95=lo,
        hi95=hi,
        beat_rest=beat_r / r.height,
        resid_tail=obs,
        resid_rest=r[resid].mean(),
        ratio=float(np.exp(obs - r[resid].mean())),
        p_mean=float(p),
        p_share=float(p_share),
    )


def quintiles(d: pl.DataFrame, measure: str, resid: str, cell: str = "cell") -> pl.DataFrame:
    """Share beating the slot (residual > 0, Wilson 95%) by within-cell quintile of the measure (1 = top)."""
    x = tail_flag(d, measure, cell).drop_nulls(resid)
    x = x.with_columns(((pl.col("q") * 5).ceil().clip(1, 5)).cast(pl.Int32).alias("quintile"))
    g = x.group_by("quintile").agg(pl.len().alias("n"), (pl.col(resid) > 0).sum().alias("k")).sort("quintile")
    ci = [wilson(k, n) for k, n in zip(g["k"], g["n"])]
    return g.select(
        pl.lit(measure).alias("measure"),
        pl.lit(resid.removesuffix("_resid")).alias("outcome"),
        "quintile",
        "n",
        (pl.col("k") / pl.col("n")).alias("share"),
        pl.Series("lo95", [c[0] for c in ci]),
        pl.Series("hi95", [c[1] for c in ci]),
    )


def plot_mismatch(q: pl.DataFrame, path, show=("ax_explosive", "TEE_z_abs", "mass_kg")) -> None:
    """Share beating the draft slot by quintile, one panel per outcome, one line per measure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (out, label) in zip(axes, OUTCOMES.items()):
        for i, m in enumerate(show):
            g = q.filter((pl.col("measure") == m) & (pl.col("outcome") == out))
            x = g["quintile"].to_numpy() + (i - 1) * 0.08
            err = [g["share"] - g["lo95"], g["hi95"] - g["share"]]
            ax.errorbar(x, g["share"], yerr=err, marker="o", ms=4, capsize=2, label=MEASURES[m])
        base = (q.filter(pl.col("outcome") == out).group_by("measure").agg((pl.col("share") * pl.col("n")).sum()
                / pl.col("n").sum())["share"].mean())  # fmt: skip
        ax.axhline(base, color="grey", lw=0.6, ls="--")
        ax.set_xticks([1, 2, 3, 4, 5], ["top 20%", "2", "3", "4", "bottom 20%"])
        ax.set_xlabel("quintile within position group × draft class")
        ax.set_title(f"{label}: share beating the draft slot")
    axes[0].set_ylabel("share above draft-slot expectation (95% CI)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def prospect_frame(processed, interim) -> pl.DataFrame:
    """One row per prospect: group, class, draft control, outcomes (log1p) and the explosion measures."""
    from eeg.outcomes.output import draft_features

    cs = pl.read_parquet(interim / "combine_si.parquet").select(
        "player_id", "name", "pos", pl.col("pos_group").alias("group"), "mass_kg"
    )
    out = pl.read_parquet(processed / "outcomes.parquet").select(
        "player_id",
        "scrim_per_season",
        "starts_per_season",
        "draft_overall_pick",
        pl.col("scrim_per_season").log1p().alias("snaps"),
        pl.col("starts_per_season").log1p().alias("starts"),
    )
    draft = draft_features(pl.read_parquet(processed / "outcomes.parquet"))
    prof = pl.read_parquet(processed / "profiles.parquet").select("player_id", "ax_explosive")
    tee = pl.read_parquet(processed / "tee.parquet").select("player_id", "TEE_z_abs", "TEE_z_rel")
    lab = pl.read_parquet(processed / "lab_burst.parquet").select("player_id", "a10")
    d = cs.join(out, on="player_id").join(draft, on="player_id")
    for t in (prof, tee, lab):
        d = d.join(t, on="player_id", how="left")
    return d.with_columns(pl.concat_str("group", pl.col("draft_year").cast(pl.String), separator="|").alias("cell"))


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT

    cfg = config.load()
    control = cfg["outcomes"]["draft_control"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    d = prospect_frame(processed, interim)
    for y in OUTCOMES:
        d = slot_expectation(d, y, control)

    tests = pl.DataFrame([tail_test(d, m, f"{y}_resid") | {"outcome": y} for y in OUTCOMES for m in MEASURES])
    tests = tests.select("outcome", *(c for c in tests.columns if c != "outcome"))
    by_group = pl.DataFrame(
        [
            tail_test(d.filter(pl.col("group") == g), m, "snaps_resid", n_perm=2000) | {"group": g}
            for g in ["WR", "TE", "DB", "OL", "DL"]
            for m in ["ax_explosive", "TEE_z_abs", "mass_kg"]
        ]
    ).select("group", "measure", "n_tail", "beat_tail", "lo95", "hi95", "beat_rest", "ratio", "p_mean", "p_share")
    q = pl.concat([quintiles(d, m, f"{y}_resid") for y in OUTCOMES for m in MEASURES])
    plot_mismatch(q, reports / "figures" / "mismatch.png")
    q_wide = q.pivot(on="quintile", index=["outcome", "measure"], values="share")

    # raw joules beyond mass: is the tail just the heaviest prospects?
    both = tail_flag(tail_flag(d, "TEE_z_abs").rename({"tail": "tail_raw", "q": "q_raw"}), "mass_kg").drop_nulls(
        "snaps_resid"
    )
    overlap = both.group_by("tail_raw", "tail").agg(
        pl.len(), (pl.col("snaps_resid") > 0).mean().alias("beat_slot"), pl.col("snaps_resid").mean().alias("resid")
    )
    overlap = overlap.rename({"tail_raw": "raw_joules_tail", "tail": "mass_tail"}).sort(
        "raw_joules_tail", "mass_tail", descending=True
    )

    named = d.with_columns(
        *(
            (pl.col(m).rank("ordinal", descending=True).over("cell") / pl.col(m).count().over("cell")).alias(f"q_{m}")
            for m in MEASURES
        ),
        (pl.col("snaps_exp").exp() - 1).alias("expected_snaps_per_season"),
    ).select(
        "player_id", "name", "pos", "group", "draft_year", "draft_overall_pick", "scrim_per_season",
        "expected_snaps_per_season", "snaps_resid", "starts_per_season", "starts_resid", *MEASURES,
        *(f"q_{m}" for m in MEASURES),
    )  # fmt: skip
    named.sort("snaps_resid", descending=True).write_parquet(processed / "mismatch.parquet")

    report = f"""# Under- and over-drafted: the explosion tail against the draft slot

Generated by `uv run python -m eeg.outcomes.mismatch`. Aggregates only; the named table is
`data/processed/mismatch.parquet` (not committed). {d.height} prospects. Every comparison is within position group ×
draft class.

- **Expectation:** snaps (and starts) per available season, log1p, on the draft control ({" + ".join(control)}) per
  group with a class intercept; leave-one-out, clipped to the group's range. Residual > 0 means more playing time than the slot predicts.
- **Tail:** top {TAIL:.0%} of a measure within group × class. `beat_*`: share with residual > 0 (`lo95`–`hi95`: Wilson
  interval for the tail). `ratio`: tail vs rest in (1 + snaps), from the mean log residuals. `p_mean` / `p_share`:
  two-sided, {N_PERM} permutations of the measure within group × class, for the tail mean residual and the tail
  share beating the slot. The share is the robust one: log residuals are long-tailed (0 vs 1,000 snaps). Five
  measures, two outcomes: read p near 0.05 as weak.

## Tail vs rest

{_md(tests, 3)}

## By group (snaps; tails of 2–9 prospects per class, so wide intervals)

{_md(by_group, 3)}

## Dose-response: share beating the slot by quintile (1 = top 20%)

A tail effect shows as a high first quintile on a flat line; a slope as a steady decline. Figure (explosive axis,
raw joules, mass, with Wilson 95% intervals): `reports/figures/mismatch.png`.

{_md(q_wide, 2)}

## Raw joules beyond mass

Raw joules within a group is driven by mass (½ m v², m g h), yet the mass tail alone does not beat the slot.
Tails crossed (snaps; post hoc, so descriptive only):

{_md(overlap, 2)}
"""
    (reports / "mismatch.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
