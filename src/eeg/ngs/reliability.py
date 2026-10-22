"""Reliability of field channels, lab channels and η; exposure needed for a stable season value.

    uv run python -m eeg.ngs.reliability

All on log values (η is a ratio, so log η = log field − log lab).

- Field, split-half: within each player-season, games in date order alternate
  between two halves; each candidate statistic (eeg.ngs.channels.season_stats)
  is computed on both halves; r between halves is stepped up to a full season
  with Spearman–Brown, 2r / (1 + r).
- Lab, rep-to-rep: the same estimators on a prospect's first two well-fit 40
  reps; r between them is the reliability of one rep.
- η: reliability of log field − log lab, with independent errors:
  1 − [(1 − rel_F) var_F + (1 − rel_L) var_L] / var(log η).
- Exposure: reliability of a statistic computed on k games (two disjoint random
  sets of k games per player-season with ≥ 2k games; no step-up).
- Exposure bias: within-group r of each log season statistic with log season
  plays. A statistic that grows with exposure credits playing time to the
  player's speed; the play-based top3_of{K} (eeg.ngs.channels) is built not to.

Every figure is given pooled and within position group (values centred on the
group mean). Within is the one that matters if η is read within position, and
pooling credits role differences as reliability. Position groups are used only
to read results.

Games count only with ≥ `games.min_game_plays` plays. Split-half and η use the
qualified player-seasons (`qualified` in field_seasons.parquet); the exposure
curve uses every player-season with ≥ 2k counting games. Writes
reports/field_reliability.md, reports/field_reliability.csv (aggregates) and
reports/figures/reliability_by_games.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.ngs.channels import CHANNELS, LAB, TOP_K, play_season_stats, season_stats

KEY = ["player_id", "season"]
GAME_COUNTS = (1, 2, 3, 4, 6, 8)
SEED = 0


def spearman_brown(r: float, factor: float = 2.0) -> float:
    return factor * r / (1 + (factor - 1) * r)


def _log(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr > 0).then(expr.log()).otherwise(None)


def _centre(df: pl.DataFrame, cols, by: str | None) -> pl.DataFrame:
    """Drop rows with a null in cols; if by is given, subtract the group mean of each column."""
    df = df.drop_nulls(cols)
    if by is None:
        return df
    return df.with_columns((pl.col(c) - pl.col(c).mean().over(by)).alias(c) for c in cols)


def _r(df: pl.DataFrame, a: str, b: str) -> float:
    return float(np.corrcoef(df[a].to_numpy(), df[b].to_numpy())[0, 1]) if df.height > 2 else float("nan")


def half_stats(games: pl.DataFrame, half: pl.Expr, channels=CHANNELS) -> pl.DataFrame:
    """Log of every channel × statistic per player-season and half (half is an expression giving 0 or 1)."""
    exprs = [_log(e).alias(f"{c}_{k}") for c in channels for k, e in season_stats(c).items()]
    return games.with_columns(half.alias("half")).group_by(*KEY, "half").agg(*exprs)


def _paired(halves: pl.DataFrame) -> pl.DataFrame:
    a = halves.filter(pl.col("half") == 0).drop("half")
    b = halves.filter(pl.col("half") == 1).drop("half")
    return a.join(b, on=KEY, suffix="_b")


def split_half(games: pl.DataFrame, groups: pl.DataFrame | None = None, channels=CHANNELS) -> pl.DataFrame:
    """Per channel × statistic: half-season r and Spearman–Brown reliability, pooled and within group."""
    order = pl.col("game_id").rank("ordinal").over(KEY) - 1
    pairs = _paired(half_stats(games, (order % 2).cast(pl.Int32), channels))
    if groups is not None:
        pairs = pairs.join(groups, on="player_id", how="left")
    rows = []
    for c in channels:
        for k in season_stats(c):
            col = f"{c}_{k}"
            row = dict(channel=c, stat=k)
            for label, by in [("pooled", None), ("within", "group")] if groups is not None else [("pooled", None)]:
                d = _centre(pairs, [col, f"{col}_b"], by)
                r = _r(d, col, f"{col}_b")
                row |= {"n": d.height, f"r_half_{label}": r, f"rel_{label}": spearman_brown(r)}
            rows.append(row)
    return pl.DataFrame(rows)


def _game_half(games: pl.DataFrame) -> pl.DataFrame:
    """Games in date order alternate between halves 0 and 1 within each player-season."""
    order = pl.col("game_id").rank("ordinal").over(KEY) - 1
    return games.with_columns((order % 2).cast(pl.Int32).alias("half"))


def _play_halves(plays: pl.DataFrame, games: pl.DataFrame, sample: int, channels) -> pl.DataFrame:
    """Log play-based statistic per player-season and half, halves given by `half` on the game rows."""
    p = plays.join(games.select("player_id", "game_id", "half"), on=["player_id", "game_id"])
    st = play_season_stats(p, games, sample, channels, by=("season", "half"))
    return st.with_columns(_log(pl.col(c)).alias(c) for c in st.columns if c not in (*KEY, "half"))


def play_split_half(
    plays: pl.DataFrame, games: pl.DataFrame, sample: int, groups: pl.DataFrame | None = None, channels=CHANNELS
) -> pl.DataFrame:
    """As split_half for the play-based statistic top{TOP_K}_of{sample}: halves by game parity, each half needs
    `sample` plays (shorter halves drop out)."""
    pairs = _paired(_play_halves(plays, _game_half(games), sample, channels))
    if groups is not None:
        pairs = pairs.join(groups, on="player_id", how="left")
    stat = f"top{TOP_K}_of{sample}"
    rows = []
    for c in channels:
        col = f"{c}_{stat}"
        row = dict(channel=c, stat=stat)
        for label, by in [("pooled", None), ("within", "group")] if groups is not None else [("pooled", None)]:
            d = _centre(pairs, [col, f"{col}_b"], by)
            r = _r(d, col, f"{col}_b")
            row |= {"n": d.height, f"r_half_{label}": r, f"rel_{label}": spearman_brown(r)}
        rows.append(row)
    return pl.DataFrame(rows)


def exposure_bias(seasons: pl.DataFrame, groups: pl.DataFrame, stats) -> pl.DataFrame:
    """Per channel × statistic, within group: r of the log season value with log season plays, and its retest
    across consecutive qualified seasons of a player."""
    d0 = seasons.join(groups, on="player_id", how="left").with_columns(pl.col("plays").log().alias("log_plays"))
    rows = []
    for c in CHANNELS:
        for k in stats:
            col = f"{c}_{k}"
            d = d0.select(*KEY, "group", "log_plays", _log(pl.col(col)).alias(col))
            nxt = d.select("player_id", (pl.col("season") - 1).alias("season"), pl.col(col).alias(f"{col}_next"))
            plays_r = _r(_centre(d, ["log_plays", col], "group"), "log_plays", col)
            rt = _centre(d.join(nxt, on=KEY), [col, f"{col}_next"], "group")
            rows.append(
                dict(
                    channel=c,
                    stat=k,
                    r_plays_within=plays_r,
                    retest_within=_r(rt, col, f"{col}_next"),
                    n_pairs=rt.height,
                )
            )
    return pl.DataFrame(rows)


def reliability_by_games(
    games: pl.DataFrame, picks: dict[str, str], ks=GAME_COUNTS, groups=None, seed: int = SEED
) -> pl.DataFrame:
    """Reliability (r between two disjoint random sets of k games) of each picked channel statistic."""
    rng = np.random.default_rng(seed)
    g = games.with_columns(pl.Series("u", rng.random(games.height)))
    g = g.with_columns((pl.col("u").rank("ordinal").over(KEY) - 1).alias("slot"), pl.len().over(KEY).alias("n_games"))
    rows = []
    for k in ks:
        sub = g.filter((pl.col("n_games") >= 2 * k) & (pl.col("slot") < 2 * k))
        pairs = _paired(half_stats(sub, (pl.col("slot") >= k).cast(pl.Int32), list(picks)))
        if groups is not None:
            pairs = pairs.join(groups, on="player_id", how="left")
        for c, stat in picks.items():
            col = f"{c}_{stat}"
            d = _centre(pairs, [col, f"{col}_b"], "group" if groups is not None else None)
            rows.append(dict(games=k, channel=c, stat=stat, n=d.height, reliability=_r(d, col, f"{col}_b")))
    return pl.DataFrame(rows)


def play_reliability_by_games(
    plays: pl.DataFrame, games: pl.DataFrame, sample: int, channels, ks=GAME_COUNTS, groups=None, seed: int = SEED
) -> pl.DataFrame:
    """As reliability_by_games for the play-based statistic (a set of k games needs `sample` plays)."""
    rng = np.random.default_rng(seed)
    g = games.with_columns(pl.Series("u", rng.random(games.height)))
    g = g.with_columns((pl.col("u").rank("ordinal").over(KEY) - 1).alias("slot"), pl.len().over(KEY).alias("n_games"))
    stat = f"top{TOP_K}_of{sample}"
    rows = []
    for k in ks:
        sub = g.filter((pl.col("n_games") >= 2 * k) & (pl.col("slot") < 2 * k))
        sub = sub.with_columns((pl.col("slot") >= k).cast(pl.Int32).alias("half"))
        pairs = _paired(_play_halves(plays, sub, sample, channels))
        if groups is not None:
            pairs = pairs.join(groups, on="player_id", how="left")
        for c in channels:
            col = f"{c}_{stat}"
            d = _centre(pairs, [col, f"{col}_b"], "group" if groups is not None else None)
            rows.append(dict(games=k, channel=c, stat=stat, n=d.height, reliability=_r(d, col, f"{col}_b")))
    return pl.DataFrame(rows)


def lab_rep_reliability(rep_channels: pl.DataFrame, cols, groups: pl.DataFrame | None = None) -> pl.DataFrame:
    """r of log value between each prospect's first two reps (reliability of one rep), pooled and within group."""
    two = rep_channels.sort("player_id", "rep_id").group_by("player_id", maintain_order=True).head(2)
    two = two.filter(pl.len().over("player_id") == 2).with_columns(pl.int_range(2).over("player_id").alias("half"))
    logs = two.select("player_id", "half", *(_log(pl.col(c)).alias(c) for c in cols)).with_columns(season=0)
    pairs = _paired(logs)
    if groups is not None:
        pairs = pairs.join(groups, on="player_id", how="left")
    rows = []
    for c in cols:
        row = dict(lab=c)
        for label, by in [("pooled", None), ("within", "group")] if groups is not None else [("pooled", None)]:
            d = _centre(pairs, [c, f"{c}_b"], by)
            row |= {"n": d.height, f"rel_{label}": _r(d, c, f"{c}_b")}
        rows.append(row)
    return pl.DataFrame(rows)


def eta_reliability(rel_f: float, rel_l: float, log_f: np.ndarray, log_l: np.ndarray) -> dict:
    """Reliability of log η = log F − log L with independent errors, plus the F–L correlation."""
    var_f, var_l = np.var(log_f, ddof=1), np.var(log_l, ddof=1)
    var_eta = np.var(log_f - log_l, ddof=1)
    err = (1 - rel_f) * var_f + (1 - rel_l) * var_l
    return dict(
        rel_eta=1 - err / var_eta, r_field_lab=float(np.corrcoef(log_f, log_l)[0, 1]), sd_log_eta=np.sqrt(var_eta)
    )


def eta_table(
    seasons: pl.DataFrame, lab: pl.DataFrame, field_rel: pl.DataFrame, lab_rel: pl.DataFrame, groups: pl.DataFrame
) -> pl.DataFrame:
    """Per channel × statistic: reliability of log η, pooled and within group."""
    base = seasons.join(lab, on="player_id").join(groups, on="player_id", how="left")
    rows = []
    for fr in field_rel.iter_rows(named=True):
        c, k = fr["channel"], fr["stat"]
        f_col, l_col = f"{c}_{k}", LAB[c]
        lr = lab_rel.filter(pl.col("lab") == l_col).row(0, named=True)
        d0 = base.select("group", _log(pl.col(f_col)).alias("F"), _log(pl.col(l_col)).alias("L"))
        row = dict(channel=c, stat=k, lab=l_col)
        for label, by in [("pooled", None), ("within", "group")]:
            d = _centre(d0, ["F", "L"], by)
            e = eta_reliability(fr[f"rel_{label}"], lr[f"rel_{label}"], d["F"].to_numpy(), d["L"].to_numpy())
            row |= {"n": d.height} | {f"{m}_{label}": v for m, v in e.items()}
        rows.append(row)
    return pl.DataFrame(rows)


def plot_by_games(curve: pl.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (c, stat), g in curve.group_by("channel", "stat", maintain_order=True):
        ax.plot(g["games"], g["reliability"], marker="o", label=f"{c} ({stat})")
    ax.axhline(0.7, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("games in the statistic")
    ax.set_ylabel("reliability within position group")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT
    from eeg.ngs.channels import counting_games, drag_coef, lab_channels

    cfg = config.load()
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    groups = pl.read_parquet(interim / "combine_si.parquet").select("player_id", pl.col("pos_group").alias("group"))
    g = cfg["games"]
    seasons = pl.read_parquet(processed / "field_seasons.parquet")
    qual = seasons.filter("qualified").select(KEY)
    all_games = counting_games(pl.read_parquet(processed / "field_games.parquet"), g["min_game_plays"])
    games = all_games.join(qual, on=KEY)
    seasons = seasons.join(qual, on=KEY)
    lab = pl.read_parquet(processed / "lab_channels.parquet")

    sample = g["season_sample_plays"]
    plays = pl.read_parquet(processed / "field_plays.parquet")
    field_rel = pl.concat([split_half(games, groups), play_split_half(plays, games, sample, groups)])
    fits = pl.read_parquet(interim / "forty_fits.parquet").filter(
        pl.col("t40_model").is_not_nan() & (pl.col("rmse") <= cfg["sprint"]["max_fit_rmse_mps"])
    )
    prospects = pl.read_parquet(interim / "combine_si.parquet").select("player_id", "ht_m", "mass_kg")
    reps = lab_channels(
        pl.read_parquet(interim / "timed_drills.parquet"), fits, drag_coef(prospects, cfg["sprint"]), cfg["games"], True
    )
    lab_cols = sorted(set(LAB.values()))
    lab_rel = lab_rep_reliability(reps, lab_cols, groups)
    eta = eta_table(seasons, lab, field_rel, lab_rel, groups)
    bias = exposure_bias(seasons, groups, field_rel["stat"].unique(maintain_order=True).to_list())

    table = field_rel.join(eta.drop("n"), on=["channel", "stat"]).join(bias, on=["channel", "stat"])
    table.write_csv(reports / "field_reliability.csv", float_precision=4)
    stat = g["season_statistic"]
    curve = pl.concat(
        [
            reliability_by_games(all_games, {c: "top3" for c in CHANNELS}, groups=groups),
            play_reliability_by_games(plays, all_games, sample, CHANNELS, groups=groups),
        ]
    )
    exposure = pl.DataFrame(
        [
            dict(min_games=k, player_seasons=s.height, prospects=s["player_id"].n_unique())
            for k in (4, 6, 8, 10)
            for s in [pl.read_parquet(processed / "field_seasons.parquet").filter(
                (pl.col("plays") >= g["min_season_plays"]) & (pl.col("games") >= k)
            )]
        ]
    )  # fmt: skip
    plot_by_games(curve.filter(pl.col("stat") == stat), reports / "figures" / "reliability_by_games.png")
    curve_wide = curve.pivot(on="channel", index=["stat", "games"], values="reliability")

    show = ["channel", "stat", "n", "rel_pooled", "rel_within", "rel_eta_pooled", "rel_eta_within",
            "r_field_lab_within", "sd_log_eta_within"]  # fmt: skip
    report = f"""# Field reliability and exposure

Generated by `uv run python -m eeg.ngs.reliability`. Counting games: ≥ {g["min_game_plays"]} plays. Qualified
player-seasons: ≥ {g["min_season_plays"]} plays and ≥ {g["min_season_games"]} counting games ({qual.height}). Log values throughout. "within": centred on the position-group mean (groups are used only to
read results); "pooled": not centred. Full table: `reports/field_reliability.csv`.

## Field statistics: split-half by game (odd vs even games, Spearman–Brown) and η

`rel_*`: reliability of the full-season field statistic. `rel_eta_*`: reliability of log(field / lab) with the
lab rep-to-rep error below. `r_field_lab_within`: correlation of field and lab within position group.
`sd_log_eta_within`: spread of log η within position group (0.10 ≈ ±10%). `top3_of{sample}`: expected mean of the
3 best plays in {sample} random plays; its halves need {sample} plays each, so its `n` is smaller (burst most).

{_md(table.select(show), 3)}

## Exposure bias and season retest

`r_plays_within`: within-group r of the log season value with log season plays. Maxima over games grow with
exposure (more games and snaps, more chances at a fast one), so a statistic built on them credits playing time to
the player. `retest_within`: r across consecutive qualified seasons of a player (role and development change
included).

{_md(table.select("channel", "stat", "r_plays_within", "retest_within", "n_pairs"), 3)}

## Lab channels: rep-to-rep (first two well-fit 40 reps; reliability of one rep)

{_md(lab_rel, 3)}

## Season statistic in use: `{stat}` (`games.season_statistic`)

The game-based statistics (`max`, `game_med`, `top3`) all rise with season plays; `top3` was the season value
until 2026-10-22, when the exposure bias showed up against playing time (Iteration 4, step 5). The play-based
statistic removes it and, for speed and kinetic energy, is also more reliable (split-half, η) and retests better
across seasons. For burst its η split-half is a little lower (its halves are the shortest), its retest higher.

## Exposure: reliability within position group over k games

Two disjoint random sets of k games per player-season with ≥ 2k games; r between them (no step-up). For
`top3_of{sample}` a set of k games needs {sample} plays, so its rows for few games rest on high-snap seasons.
Figure (statistic in use): `reports/figures/reliability_by_games.png`.

{_md(curve_wide, 3)}

Player-seasons (≥ {g["min_season_plays"]} plays) kept by the minimum number of counting games:

{_md(exposure)}
"""
    (reports / "field_reliability.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
