"""Validate field kinematics against themselves and against the Combine 40.

    uv run python -m eeg.ngs.validate

Checks, on data/interim/game_tracking_{season}.parquet (REG + POST):
1. Reported NGS speed vs speed from central x/y differences, by speed band.
2. What NGS `a` measures: correlation with total |a| and with |dv/dt|.
3. Glitch rule: what dropping glitch tracks removes, and what keeping them would do
   to each player-season's top speed.
4. Field vs lab, same estimators on both sides (reported speed; a_s = smoothed
   d(speed)/dt): per player-season peak speed and peak acceleration against the
   best Combine 40 rep (data/interim/forty_fits.parquet, timed_drills.parquet).
   "Sustained" peaks need the value held for SUSTAIN_FRAMES frames, so one-frame
   spikes can't set them. Season-to-season retest for players with two qualifying
   seasons. Position groups are used only to read the result.
5. Sensitivity of peak acceleration to the Savitzky–Golay window.

Writes reports/field_kinematics.md (aggregates only), data/processed/field_validation.parquet
(per player-season, not committed) and reports/figures/field_vs_lab.png.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

from eeg.ngs.kinematics import savgol_track

TRACK = ["game_id", "play_id", "player_id"]
SUSTAIN_FRAMES = 3  # = games.sustain_frames (config/physics.yaml)
MIN_PLAYS = 100
SPEED_BANDS = [0.0, 2.0, 4.0, 6.0, 8.0, 9.0]
WINDOWS = (3, 5, 7, 9)
POS_ORDER = ("WR", "DB", "TE", "DL", "OL")


def _band(col: str, edges, labels) -> pl.Expr:
    """Label col by half-open intervals [edges[i], edges[i + 1]); labels has len(edges) + 1 entries."""
    expr = pl.when(pl.col(col) < edges[0]).then(pl.lit(labels[0]))
    for e, lab in zip(edges[1:], labels[1:]):
        expr = expr.when(pl.col(col) < e).then(pl.lit(lab))
    return expr.otherwise(pl.lit(labels[-1]))


def speed_agreement(frames: pl.DataFrame) -> pl.DataFrame:
    """By reported-speed band (clean tracks): ratio of x/y speed to reported speed, and mismatch share."""
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(SPEED_BANDS, SPEED_BANDS[1:])] + [f">{SPEED_BANDS[-1]:g}"]
    ratio = pl.col("v_xy_mps") / pl.col("s_mps")
    return (
        frames.filter(~pl.col("glitch_track") & (pl.col("s_mps") > 0.5))
        .with_columns(_band("s_mps", SPEED_BANDS[1:], labels).alias("band"))
        .group_by("band")
        .agg(
            pl.len().alias("frames"),
            ratio.quantile(0.05).alias("xy_over_reported_p5"),
            ratio.median().alias("median"),
            ratio.quantile(0.95).alias("p95"),
            pl.col("flag_mismatch").mean().alias("mismatch_share"),
        )
        .sort("band")
    )


def accel_identity(frames: pl.DataFrame, window: int = 7) -> pl.DataFrame:
    """Correlation of NGS `a` with acceleration estimators (clean tracks, moving frames)."""
    f = frames.filter(~pl.col("glitch_track")).with_columns(pl.len().over(TRACK).alias("n_frames"))
    fid, n = f["frame_id"].to_numpy(), f["n_frames"].to_numpy()

    def d(col, k):
        return savgol_track(f[col].to_numpy(), fid, n, window, deriv=k)

    vx, vy, ax, ay = d("x_m", 1), d("y_m", 1), d("x_m", 2), d("y_m", 2)
    speed = np.hypot(vx, vy)
    a_tot = np.hypot(ax, ay)
    a_tan = (vx * ax + vy * ay) / np.maximum(speed, 1e-9)
    a_norm = np.sqrt(np.maximum(a_tot**2 - a_tan**2, 0))
    a_ngs, s = f["a_mps2"].to_numpy(), f["s_mps"].to_numpy()
    ok = ~np.isnan(a_tot) & (s > 1)
    big = ok & (a_ngs > 2)
    rows = []
    for name, v in [
        ("total |a| from x/y", a_tot),
        ("tangential |dv/dt| from x/y", np.abs(a_tan)),
        ("normal (turning) from x/y", a_norm),
        ("|d(reported speed)/dt|", np.abs(d("s_mps", 1))),
    ]:
        rows.append(
            dict(
                estimator=name, r_with_ngs_a=pearsonr(v[ok], a_ngs[ok])[0], median_ratio=np.median(v[big] / a_ngs[big])
            )
        )
    return pl.DataFrame(rows)


def field_peaks(frames: pl.DataFrame, accel_col: str = "a_s_mps2", keep_glitch: bool = False) -> pl.DataFrame:
    """Per player-season: plays, peak and sustained-peak speed, peak and sustained-peak acceleration."""
    f = frames if keep_glitch else frames.filter(~pl.col("glitch_track"))
    sust = SUSTAIN_FRAMES
    return (
        f.with_columns(
            pl.col("s_mps").rolling_min(sust).over(TRACK).alias("s_sust"),
            pl.col(accel_col).rolling_min(sust).over(TRACK).alias("a_sust"),
        )
        .group_by("player_id", "season")
        .agg(
            pl.struct("game_id", "play_id").n_unique().alias("plays"),
            pl.col("s_mps").max().alias("s_peak"),
            pl.col("s_sust").max().alias("s_peak_sust"),
            pl.col(accel_col).max().alias("a_peak"),
            pl.col("a_sust").max().alias("a_peak_sust"),
        )
        .sort("player_id", "season")
    )


def lab_peaks(drills: pl.DataFrame, best: pl.DataFrame, window: int) -> pl.DataFrame:
    """Per player, on the best 40 rep up to the finish line: fitted v_max, a0 = v_max/τ, and the field estimators."""
    reps = (
        drills.filter(pl.col("drill") == "forty")
        .join(best.select("rep_id", "v_max", "tau", "t_finish"), on="rep_id")
        .sort("rep_id", "t_s")
        .with_columns(
            pl.int_range(pl.len()).over("rep_id").alias("frame_id"),
            pl.len().over("rep_id").alias("n_frames"),
        )
    )
    a_s = savgol_track(reps["s_mps"].to_numpy(), reps["frame_id"].to_numpy(), reps["n_frames"].to_numpy(), window)
    sust = SUSTAIN_FRAMES
    return (
        reps.with_columns(pl.Series("a_s", a_s).fill_nan(None))
        .with_columns(
            pl.col("s_mps").rolling_min(sust).over("rep_id").alias("s_sust"),
            pl.col("a_s").rolling_min(sust).over("rep_id").alias("a_sust"),
        )
        .filter(pl.col("t_s") <= pl.col("t_finish"))
        .group_by("player_id")
        .agg(
            pl.col("v_max").first().alias("lab_v_max"),
            (pl.col("v_max") / pl.col("tau")).first().alias("lab_a0"),
            pl.col("s_mps").max().alias("lab_s_peak"),
            pl.col("s_sust").max().alias("lab_s_peak_sust"),
            pl.col("a_s").max().alias("lab_a_peak"),
            pl.col("a_sust").max().alias("lab_a_peak_sust"),
        )
        .sort("player_id")
    )


CHANNELS = {  # field column -> lab column (same estimator)
    "s_peak": "lab_s_peak",
    "s_peak_sust": "lab_s_peak_sust",
    "a_peak": "lab_a_peak",
    "a_peak_sust": "lab_a_peak_sust",
}


def compare(df: pl.DataFrame, pairs: dict[str, str] = CHANNELS) -> pl.DataFrame:
    """Per channel: field / lab ratio distribution, share above the lab value, Pearson and Spearman r."""
    rows = []
    for field, lab in pairs.items():
        d = df.select(field, lab).drop_nulls()
        x, y = d[lab].to_numpy(), d[field].to_numpy()
        q = np.quantile(y / x, [0.1, 0.5, 0.9])
        rows.append(
            dict(
                field=field, lab=lab, n=len(x), field_above_lab=float((y > x).mean()),
                ratio_p10=q[0], ratio_median=q[1], ratio_p90=q[2],
                pearson=pearsonr(x, y)[0], spearman=spearmanr(x, y)[0],
            )
        )  # fmt: skip
    return pl.DataFrame(rows)


def season_retest(field: pl.DataFrame, cols, min_plays: int = MIN_PLAYS) -> pl.DataFrame:
    """Pearson r of each column between a player's consecutive qualifying seasons."""
    q = field.filter(pl.col("plays") >= min_plays).sort("player_id", "season")
    nxt = (
        q.with_columns(pl.col("season") - 1)
        .rename({c: f"{c}_next" for c in cols})
        .select("player_id", "season", *(f"{c}_next" for c in cols))
    )
    pairs = q.join(nxt, on=["player_id", "season"])
    return pl.DataFrame([dict(column=c, pairs=pairs.height, r=pearsonr(pairs[c], pairs[f"{c}_next"])[0]) for c in cols])


def accel_peak_context(frames: pl.DataFrame) -> dict:
    """Where each player-season's sustained peak acceleration happens within its play."""
    f = frames.filter(~pl.col("glitch_track")).with_columns(
        pl.col("a_s_mps2").rolling_min(SUSTAIN_FRAMES).over(TRACK).alias("a_sust"),
        pl.col("a_s_mps2").rolling_min(10).shift(1).over(TRACK).alias("a_min_prev_1s"),
    )
    pk = f.sort("a_sust", descending=True, nulls_last=True).group_by("player_id", "season").first()
    return dict(
        t_snap_median=pk["t_snap_s"].median(),
        share_first_1_5s=pk["t_snap_s"].is_between(0, 1.5).mean(),
        share_after_hard_decel=(pk["a_min_prev_1s"] < -4).mean(),
        speed_at_peak_median=pk["s_mps"].median(),
    )


def glitch_effect(frames: pl.DataFrame) -> dict:
    """Share of tracks dropped, and top-speed changes if glitch tracks were kept."""
    tracks = frames.select(*TRACK, "glitch_track").unique()
    kept, clean = field_peaks(frames, keep_glitch=True), field_peaks(frames)
    j = kept.join(clean, on=["player_id", "season"], suffix="_clean", how="left")
    diff = (j["s_peak"] - j["s_peak_clean"].fill_null(0.0)).to_numpy()
    return dict(
        tracks=tracks.height,
        glitch_tracks=int(tracks["glitch_track"].sum()),
        player_seasons=j.height,
        changed=int((diff > 1e-9).sum()),
        changed_by_over_0_5=int((diff > 0.5).sum()),
        peak_over_11_kept=int((j["s_peak"] > 11).sum()),
        peak_over_11_clean=int((j["s_peak_clean"] > 11).sum()),
    )


def window_sensitivity(frames: pl.DataFrame, drills: pl.DataFrame, best: pl.DataFrame, min_plays: int) -> pl.DataFrame:
    """Peak acceleration (field and lab) under each Savitzky–Golay window."""
    f = frames.filter(~pl.col("glitch_track")).with_columns(pl.len().over(TRACK).alias("n_frames"))
    fid, n, s = f["frame_id"].to_numpy(), f["n_frames"].to_numpy(), f["s_mps"].to_numpy()
    rows, ref = [], None
    for w in WINDOWS:
        fw = f.with_columns(pl.Series("a_w", savgol_track(s, fid, n, w)).fill_nan(None))
        peaks = field_peaks(fw, accel_col="a_w").filter(pl.col("plays") >= min_plays)
        both = peaks.join(lab_peaks(drills, best, w), on="player_id")
        ref = both if ref is None else ref
        same = both.join(
            ref.select("player_id", "season", pl.col("a_peak_sust").alias("ref")), on=["player_id", "season"]
        )
        rows.append(
            dict(
                window=w, field_a_peak_sust_median=both["a_peak_sust"].median(),
                lab_a_peak_sust_median=both["lab_a_peak_sust"].median(),
                ratio_median=(both["a_peak_sust"] / both["lab_a_peak_sust"]).median(),
                field_lab_spearman=spearmanr(both["a_peak_sust"], both["lab_a_peak_sust"])[0],
                rank_vs_window_3=spearmanr(same["a_peak_sust"], same["ref"])[0],
            )
        )  # fmt: skip
    return pl.DataFrame(rows)


def plot_field_vs_lab(df: pl.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (field, lab, unit) in zip(
        axes, [("s_peak_sust", "lab_s_peak_sust", "m/s"), ("a_peak_sust", "lab_a_peak_sust", "m/s²")]
    ):
        for pos in POS_ORDER:
            g = df.filter(pl.col("pos_group") == pos)
            ax.scatter(g[lab], g[field], s=10, alpha=0.6, label=pos)
        lo = min(df[lab].min(), df[field].min())
        hi = max(df[lab].max(), df[field].max())
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, label="field = lab")
        ax.set_xlabel(f"lab: {lab} ({unit})")
        ax.set_ylabel(f"field season: {field} ({unit})")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.clustering.audit import _md
    from eeg.config import PROJECT_ROOT
    from eeg.ingest.load_games import SEASONS
    from eeg.physics.sprint import best_reps

    cfg = config.load()
    window = cfg["games"]["accel_window_frames"]
    interim, processed, reports = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports"))
    cols = ["player_id", "season", "game_id", "play_id", "frame_id", "x_m", "y_m", "s_mps", "a_mps2",
            "v_xy_mps", "a_s_mps2", "t_snap_s", "flag_mismatch", "glitch_track"]  # fmt: skip
    frames = pl.concat([pl.read_parquet(interim / f"game_tracking_{s}.parquet", columns=cols) for s in SEASONS])
    drills = pl.read_parquet(interim / "timed_drills.parquet")
    best = best_reps(pl.read_parquet(interim / "forty_fits.parquet"), cfg["sprint"]["max_fit_rmse_mps"])
    pos = pl.read_parquet(interim / "combine_si.parquet").select("player_id", "pos_group")

    agree = speed_agreement(frames)
    ident = accel_identity(frames)
    glitch = glitch_effect(frames)

    field = field_peaks(frames)
    lab = lab_peaks(drills, best, window)
    joined = field.join(lab, on="player_id", how="left").join(pos, on="player_id", how="left")
    joined.write_parquet(processed / "field_validation.parquet")
    q = joined.filter((pl.col("plays") >= MIN_PLAYS) & pl.col("lab_s_peak").is_not_null())
    plot_field_vs_lab(q, reports / "figures" / "field_vs_lab.png")

    pooled = compare(q)
    vs_model = compare(q, {"s_peak_sust": "lab_v_max", "a_peak_sust": "lab_a0"})
    lab_internal = compare(lab, {"lab_s_peak_sust": "lab_v_max", "lab_a_peak_sust": "lab_a0"})
    by_pos = pl.concat(
        [compare(q.filter(pl.col("pos_group") == p), {"s_peak_sust": "lab_s_peak_sust", "a_peak_sust": "lab_a_peak_sust"})
         .insert_column(0, pl.lit(p).alias("pos_group")) for p in POS_ORDER]
    ).select("pos_group", "field", "n", "field_above_lab", "ratio_median", "pearson", "spearman")  # fmt: skip
    exposure = (
        q.with_columns(_band("plays", [250, 500], ["100-249", "250-499", "500+"]).alias("plays_band"))
        .group_by("plays_band")
        .agg(
            pl.len().alias("player_seasons"),
            (pl.col("s_peak_sust") / pl.col("lab_s_peak_sust")).median().alias("speed_ratio_median"),
            (pl.col("a_peak_sust") / pl.col("lab_a_peak_sust")).median().alias("accel_ratio_median"),
        )
        .sort("plays_band")
    )
    retest = season_retest(joined, ["s_peak", "s_peak_sust", "a_peak", "a_peak_sust"])
    sens = window_sensitivity(frames, drills, best, MIN_PLAYS)
    ctx = accel_peak_context(frames)

    n_players = q["player_id"].n_unique()
    report = f"""# Field kinematics validation

Generated by `uv run python -m eeg.ngs.validate` on {frames.height:,} REG + POST frames
({", ".join(map(str, SEASONS))}). Aggregates only; per player-season values are in
`data/processed/field_validation.parquet` (not committed). Figure: `reports/figures/field_vs_lab.png`.

## 1. Reported speed vs x/y speed (clean tracks, reported speed > 0.5 m/s)

Ratio of central-difference x/y speed to NGS reported speed, by reported speed (m/s).
`mismatch_share`: frames with |difference| > `games.max_speed_mismatch_mps`.

{_md(agree, 3)}

## 2. What NGS `a` measures (clean tracks, speed > 1 m/s; Savitzky–Golay window 7 on x/y)

{_md(ident, 3)}

NGS `a` follows total acceleration, including turning, and is never negative, so it can't separate
propulsion from braking or cutting. Field and lab acceleration both use `a_s` = Savitzky–Golay
d(reported speed)/dt (window `games.accel_window_frames` = {window}), the quantity the lab τ was fit to.

## 3. Glitch rule: drop whole tracks with a glitch frame

- Tracks: {glitch["tracks"]:,}; with a glitch frame: {glitch["glitch_tracks"]:,} ({glitch["glitch_tracks"] / glitch["tracks"]:.2%}).
- Player-seasons: {glitch["player_seasons"]}. Keeping glitch tracks would raise the peak speed of {glitch["changed"]}
  ({glitch["changed_by_over_0_5"]} by more than 0.5 m/s), and {glitch["peak_over_11_kept"]} would have a peak over 11 m/s
  (clean: {glitch["peak_over_11_clean"]}).

## 4. Field vs lab (player-seasons with ≥ {MIN_PLAYS} plays and a usable 40: {q.height}, {n_players} prospects)

Peaks are per season; "sust" peaks must hold for {SUSTAIN_FRAMES} frames ({SUSTAIN_FRAMES / 10:.1f} s). Lab values come from
the best 40 rep up to the finish line, using the same estimators.

{_md(pooled, 3)}

Against the 40 model (fitted v_max and a0 = v_max/τ):

{_md(vs_model, 3)}

Inside the lab (all {lab.height} prospects with a usable 40), the measured peaks against the model:

{_md(lab_internal, 3)}

By position group (post hoc reading only):

{_md(by_pos, 3)}

Exposure: ratio of field to lab (sustained peaks) by plays in the season.

{_md(exposure, 3)}

Season-to-season retest (Pearson r between consecutive qualifying seasons of a player):

{_md(retest, 3)}

Where the season's sustained peak acceleration happens: median {ctx["t_snap_median"]:.1f} s after the snap;
{ctx["share_first_1_5s"]:.0%} within the first 1.5 s; {ctx["share_after_hard_decel"]:.0%} within 1 s of braking harder than
4 m/s² (re-acceleration out of a cut); median speed at the peak {ctx["speed_at_peak_median"]:.1f} m/s.

## 5. Acceleration window sensitivity (sustained peak, player-seasons with ≥ {MIN_PLAYS} plays)

{_md(sens, 3)}
"""
    (reports / "field_kinematics.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
