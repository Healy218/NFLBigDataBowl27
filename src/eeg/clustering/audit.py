"""Feature audit before clustering: reliability, validity and collinearity.

    uv run python -m eeg.clustering.audit

- Test-retest: players with two well-fit 40 reps; r between rep 1 and rep 2 for
  v_max and tau (all prospects and per combine_year).
- Validity of tau: partial r of tau with the official 10-yd split, controlling for
  v_max. The split is an independent stopwatch measure of acceleration, so a strong
  partial r means tau carries real acceleration signal, not just fit noise.
- Collinearity: feature correlations, VIF and PCA (eigenvalues, loadings) of
  data/processed/features.parquet.

Writes reports/feature_audit.md (aggregate statistics only).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def test_retest(forty_reps: pl.DataFrame, cols=("v_max", "tau"), by: str = "draft_year") -> pl.DataFrame:
    """r between each player's first two reps (by rep_id), pooled ("all") and per `by`."""
    two = (
        forty_reps.sort("player_id", "rep_id")
        .group_by("player_id", maintain_order=True)
        .head(2)
        .filter(pl.len().over("player_id") == 2)
        .with_columns(pl.int_range(pl.len()).over("player_id").cast(pl.String).alias("rep"))
    )
    rows = []
    for c in cols:
        w = two.pivot(on="rep", index=["player_id", by], values=c)
        groups = [("all", w), *[(str(k), g) for (k,), g in w.group_by([by], maintain_order=True)]]
        for name, g in sorted(groups, key=lambda x: x[0]):
            rows.append(dict(column=c, cohort=name, n=g.height, r=g.select(pl.corr("0", "1")).item()))
    return pl.DataFrame(rows)


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """r of x and y after regressing each on z (with intercept)."""
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def tau_validity(context: pl.DataFrame, split: pl.DataFrame, by: str = "combine_year") -> pl.DataFrame:
    """Partial r(tau, official 10-yd split | v_max), and split R² from v_max alone vs v_max + tau."""
    d = context.join(split, on="player_id").drop_nulls(["ten_split_s", "v_max", "tau"])
    rows = []
    groups = [("all", d), *[(str(k), g) for (k,), g in d.group_by([by])]]
    for name, g in sorted(groups, key=lambda x: x[0]):
        s, vm, tau = (g[c].to_numpy() for c in ("ten_split_s", "v_max", "tau"))
        rows.append(
            dict(
                cohort=name,
                n=g.height,
                partial_r=partial_corr(tau, s, vm),
                r2_vmax=_r2(s, vm[:, None]),
                r2_vmax_tau=_r2(s, np.column_stack([vm, tau])),
            )
        )
    return pl.DataFrame(rows)


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    A = np.column_stack([np.ones(len(y)), X])
    res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - res @ res / np.sum((y - y.mean()) ** 2))


def vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each column: 1 / (1 - R² on the others)."""
    return np.array([1 / (1 - _r2(X[:, i], np.delete(X, i, axis=1))) for i in range(X.shape[1])])


def pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and loadings of the correlation matrix.

    Each loading vector's sign is set so its largest-magnitude entry is positive.
    """
    ev, vec = np.linalg.eigh(np.corrcoef(X.T))
    order = ev.argsort()[::-1]
    ev, vec = ev[order], vec[:, order]
    vec = vec * np.sign(vec[np.abs(vec).argmax(axis=0), np.arange(vec.shape[1])])
    return ev, vec


def _md(df: pl.DataFrame, digits: int = 2) -> str:
    def fmt(v):
        return f"{v:.{digits}f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join(lines)


def main() -> None:
    from eeg import config
    from eeg.config import PROJECT_ROOT

    cfg = config.load()
    interim, processed = PROJECT_ROOT / "data" / "interim", PROJECT_ROOT / "data" / "processed"
    reps = pl.read_parquet(interim / "forty_fits.parquet").filter(pl.col("rmse") <= cfg["sprint"]["max_fit_rmse_mps"])
    features = pl.read_parquet(processed / "features.parquet")
    context = pl.read_parquet(processed / "features_context.parquet")
    split = pl.read_parquet(interim / "combine_si.parquet").select("player_id", "ten_split_s")

    tr = test_retest(reps)
    tv = tau_validity(context, split)
    cols = features.columns[1:]
    X = features.drop("player_id").to_numpy()
    ev, vec = pca(X)
    corr = pl.DataFrame(np.corrcoef(X.T), schema=cols, orient="row").insert_column(0, pl.Series("feature", cols))
    loadings = pl.DataFrame(vec, schema=[f"PC{i + 1}" for i in range(len(cols))], orient="row").insert_column(
        0, pl.Series("feature", cols)
    )
    var = pl.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(ev))],
            "eigenvalue": ev,
            "var_share": ev / ev.sum(),
            "cumulative": np.cumsum(ev) / ev.sum(),
        }
    )
    vifs = pl.DataFrame({"feature": cols, "VIF": vif(X)})

    report = f"""# Phase 3 feature audit

Generated by `uv run python -m eeg.clustering.audit` on {features.height} prospects (`data/processed/features.parquet`).

## Test-retest of the 40 fit (players with two reps)

{_md(tr.pivot(on="cohort", index="column", values="r"))}

n per cohort: {", ".join(f"{k} {v}" for k, v in tr.filter(pl.col("column") == "v_max").select("cohort", "n").iter_rows())}

## tau against the official 10-yd split

partial_r = r(tau, split | v_max). R² of the split on v_max alone and on v_max + tau.

{_md(tv)}

## Feature correlations

{_md(corr)}

## VIF

{_md(vifs)}

## PCA (correlation matrix)

{_md(var)}

{_md(loadings)}
"""
    out = PROJECT_ROOT / "reports" / "feature_audit.md"
    out.write_text(report)
    print(report)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
