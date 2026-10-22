"""Final figures for the write-up, "True ball is an explosion".

    uv run python -m eeg.viz.figures

Five figures, each built from the same functions as the analysis reports (no numbers typed in):
1. From drill to joules: one prospect's tracked 40 with the fitted v(t) = v_max (1 − e^(−(t − t0)/τ)), and their
   explosive energy per drill against their Combine class. The prospect is described, not named.
2. Explosion is real on the field, and the draft has priced it on average: cross-validated R² of the Combine for
   game top speed and burst; of draft slot, the Combine, and both for playing time.
3. The explosive tail beats its draft slot: share beating the slot by raw-joules quintile (mass for contrast), and
   the robustness checks.
4. Defensive line: the Combine start → the game start → pressure (partial plots, centred on NFL position).
5. How much Combine top speed reaches games, by position group (η_speed), and the compression with engine size.

Writes reports/figures/final_*.png (not committed; reports/figures is gitignored).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from eeg.clustering.readout import POS_ORDER

GROUP_COLOURS = {"WR": "#1f77b4", "DB": "#2ca02c", "TE": "#9467bd", "DL": "#d62728", "OL": "#8c564b"}
EXPLOSION, GREY, DRAFT = "#e4572e", "#8a8a8a", "#4a6fa5"


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
        }
    )


def _save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def _partial(d: pl.DataFrame, y: str, x: str, controls, group: str) -> tuple[np.ndarray, np.ndarray]:
    """Frisch–Waugh partial residuals of y and x on the controls, everything centred on group means."""
    sub = d.select(group, y, x, *controls).drop_nulls()
    cols = [y, x, *controls]
    c = sub.with_columns((pl.col(k) - pl.col(k).mean().over(group)) for k in cols)
    z = c.select(controls).to_numpy().astype(float) if controls else np.zeros((c.height, 0))

    def resid(v):
        if z.shape[1] == 0:
            return v
        b, *_ = np.linalg.lstsq(z, v, rcond=None)
        return v - z @ b

    return resid(c[x].to_numpy().astype(float)), resid(c[y].to_numpy().astype(float))


# 1. From drill to joules --------------------------------------------------------------------------------------


def fig_physics(interim, processed, path, player_id: int, label: str, cfg: dict) -> None:
    """Left: tracked 40 speed and the fitted model. Right: z-score of each energy channel within the Combine
    class, with the joules."""
    import matplotlib.pyplot as plt

    from eeg.physics.sprint import best_reps, speed_model

    fits = pl.read_parquet(interim / "forty_fits.parquet")
    rep = best_reps(fits, cfg["sprint"]["max_fit_rmse_mps"]).filter(pl.col("player_id") == player_id).row(0, named=True)
    trace = pl.read_parquet(interim / "timed_drills.parquet").filter(pl.col("rep_id") == rep["rep_id"]).sort("t_s")
    trace = trace.filter(pl.col("t_s") <= rep["t_finish"] + 0.3)
    tee = pl.read_parquet(processed / "tee.parquet").filter(pl.col("player_id") == player_id).row(0, named=True)
    mass = tee["mass_kg"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    t = np.linspace(rep["t0"], rep["t_finish"], 200)
    ax0.scatter(trace["t_s"] - rep["t0"], trace["s_mps"], s=10, color=GREY, label="10 Hz Combine tracking")
    ax0.plot(t - rep["t0"], speed_model(t, rep["v_max"], rep["tau"], rep["t0"]), color=EXPLOSION, lw=2,
             label=rf"fit: $v_{{max}}$ = {rep['v_max']:.2f} m/s, $\tau$ = {rep['tau']:.2f} s")  # fmt: skip
    ke = 0.5 * mass * rep["v_max"] ** 2
    ax0.axhline(rep["v_max"], color=EXPLOSION, lw=0.6, ls="--")
    ax0.text(
        0.98, 0.08, f"kinetic energy at top speed\n½ m v² = {ke:,.0f} J ({mass:.0f} kg)",
        transform=ax0.transAxes, ha="right", fontsize=8,
    )  # fmt: skip
    ax0.set_xlim(left=0)
    ax0.set_xlabel("seconds from the start")
    ax0.set_ylabel("speed (m/s)")
    ax0.set_title("The 40, as the sensors saw it")
    ax0.legend(loc="center right")

    chans = [
        ("E_40", "40-yard dash (tracked: ½mv² + drag)", "E_40"),
        ("E_vert", "vertical jump", "E_vert"),
        ("E_broad", "broad jump", "E_broad"),
        ("W_bench", "bench press", "W_bench"),
    ]
    rows = [(lab, tee[f"z_{c}"], tee[j]) for c, lab, j in chans if tee[f"z_{c}"] is not None]
    rows.append(("total explosive energy", tee["TEE_z_abs"], None))
    y = np.arange(len(rows))[::-1]
    z = [r[1] for r in rows]
    ax1.barh(y, z, color=[EXPLOSION if i == len(rows) - 1 else "#f4a582" for i in range(len(rows))])
    for yi, (lab, zi, j) in zip(y, rows):
        ax1.text(zi + 0.05, yi, f"{j:,.0f} J" if j is not None else f"{zi:+.2f} SD", va="center", fontsize=8)
    ax1.set_yticks(y, [r[0] for r in rows])
    ax1.axvline(0, color=GREY, lw=0.6)
    ax1.set_xlabel("SD above the Combine class (absolute joules)")
    ax1.set_title("Drills become joules")
    fig.suptitle(label, fontsize=10, x=0.01, ha="left", color="#333333")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# 2. Real on the field, priced by the draft --------------------------------------------------------------------


def fig_priced(interim, processed, path, cfg: dict) -> None:
    """Left: CV R² of the Combine for game top speed and burst (within group). Right: CV R² for playing time of
    the draft, the Combine and both (within group × class)."""
    import matplotlib.pyplot as plt

    from eeg.ngs.transfer import COMPACT, cv_r2, prospect_table
    from eeg.outcomes.output import draft_features, model_row, playing_time

    control = cfg["outcomes"]["draft_control"]
    d = prospect_table(processed, interim, cfg["games"]["season_statistic"])
    field = [
        (lab, cv_r2(d, y, COMPACT)) for y, lab in [("field_speed", "game top speed"), ("field_burst", "game burst")]
    ]
    seasons = pl.read_parquet(processed / "outcomes_seasons.parquet")
    pt = (
        d.join(draft_features(pl.read_parquet(processed / "outcomes.parquet")), on="player_id")
        .join(playing_time(seasons), on="player_id")
        .with_columns(pl.concat_str("group", pl.col("draft_year").cast(pl.String), separator="|").alias("cell"))
        .drop_nulls([*COMPACT, *control])
    )
    play = [
        ("Combine", model_row(pt, "pt", COMPACT, "cell", "")["cv_r2"], EXPLOSION),
        ("draft slot", model_row(pt, "pt", control, "cell", "")["cv_r2"], DRAFT),
        ("draft + Combine", model_row(pt, "pt", [*control, *COMPACT], "cell", "")["cv_r2"], GREY),
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    ax0.barh([f[0] for f in field][::-1], [f[1] for f in field][::-1], color=EXPLOSION)
    for i, (_, v) in enumerate(field[::-1]):
        ax0.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax0.set_title("The Combine engine shows up on the field")
    ax0.set_xlabel("cross-validated R² within position group")
    ax1.barh([p[0] for p in play][::-1], [p[1] for p in play][::-1], color=[p[2] for p in play][::-1])
    for i, p in enumerate(play[::-1]):
        ax1.text(p[1] + 0.01, i, f"{p[1]:.2f}", va="center", fontsize=8)
    ax1.set_title("…but on average the draft has priced it in")
    ax1.set_xlabel("cross-validated R², playing time (within group × class)")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# 3. The explosive tail beats its slot -------------------------------------------------------------------------


def fig_tail(interim, processed, path, cfg: dict) -> None:
    """Left: share beating the draft slot by quintile, raw joules vs mass. Right: the tail across checks."""
    import matplotlib.pyplot as plt

    from eeg.outcomes.mismatch import prospect_frame, quintiles
    from eeg.outcomes.robustness import row, tee_channels, with_expectations

    control = cfg["outcomes"]["draft_control"]
    d = prospect_frame(processed, interim).join(tee_channels(processed), on="player_id", how="left")
    d = with_expectations(d, "snaps", control)
    q = pl.concat([quintiles(d, m, "snaps_resid") for m in ("TEE_z_abs", "mass_kg")])
    checks = [
        ("top 20% (headline)", row(d, "", "", "TEE_z_abs", "snaps_resid", 0.2)),
        ("top 10%", row(d, "", "", "TEE_z_abs", "snaps_resid", 0.1)),
        ("top 30%", row(d, "", "", "TEE_z_abs", "snaps_resid", 0.3)),
        ("expected from log pick", row(d, "", "", "TEE_z_abs", "snaps_resid_log_pick")),
        ("expected from round", row(d, "", "", "TEE_z_abs", "snaps_resid_bands")),
        ("expected from nearest picks", row(d, "", "", "TEE_z_abs", "snaps_resid_knn")),
        *((f"{yr} class only", row(d.filter(pl.col("draft_year") == yr), "", "", "TEE_z_abs", "snaps_resid"))
          for yr in (2023, 2024, 2025)),
        ("without DL", row(d.filter(pl.col("group") != "DL"), "", "", "TEE_z_abs", "snaps_resid")),
        ("tracked 40 energy alone", row(d, "", "", "z_E_40", "snaps_resid")),
    ]  # fmt: skip

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    for m, colour, lab in [("TEE_z_abs", EXPLOSION, "raw explosive energy"), ("mass_kg", GREY, "body mass")]:
        g = q.filter(pl.col("measure") == m)
        off = -0.06 if m == "TEE_z_abs" else 0.06
        ax0.errorbar(
            g["quintile"] + off, g["share"], yerr=[g["share"] - g["lo95"], g["hi95"] - g["share"]],
            marker="o", ms=5, capsize=2, color=colour, label=lab,
        )  # fmt: skip
    ax0.axhline(0.5, color=GREY, lw=0.6, ls="--")
    ax0.set_xticks([1, 2, 3, 4, 5], ["top 20%", "2nd", "3rd", "4th", "bottom 20%"])
    ax0.set_xlabel("quintile within position group × draft class")
    ax0.set_ylabel("share playing more snaps than the draft slot predicts")
    ax0.set_title("Only the most explosive fifth outplays its slot")
    ax0.legend(loc="upper right")

    y = np.arange(len(checks))[::-1]
    tail = np.array([c[1]["beat_tail"] for c in checks])
    lo, hi = np.array([c[1]["lo95"] for c in checks]), np.array([c[1]["hi95"] for c in checks])
    ax1.errorbar(tail, y, xerr=[tail - lo, hi - tail], fmt="o", ms=5, color=EXPLOSION, ecolor=EXPLOSION, capsize=2)
    ax1.scatter([c[1]["beat_rest"] for c in checks], y, marker="|", s=90, color=GREY, label="rest of the class")
    ax1.axvline(0.5, color=GREY, lw=0.6, ls="--")
    ax1.set_yticks(y, [f"{c[0]}  (n {c[1]['n_tail']})" for c in checks])
    ax1.set_xlabel("share of the explosive tail beating its slot (95% CI)")
    ax1.set_title("It holds across every check")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# 4. The DL chain ----------------------------------------------------------------------------------------------


def fig_dl_chain(interim, processed, path, cfg: dict) -> None:
    """Partial plots within NFL position: a10 → get-off; get-off → pressure rate beyond the draft and plays."""
    import matplotlib.pyplot as plt

    from eeg.outcomes.mismatch import prospect_frame
    from eeg.outcomes.output import betas_long, dl_chain

    control = cfg["outcomes"]["draft_control"]
    d = prospect_frame(processed, interim)
    lab = pl.read_parquet(processed / "lab_burst.parquet").select("player_id", pl.col("a10").log().alias("log_a10"))
    chain = dl_chain(
        pl.read_parquet(processed / "production_seasons.parquet"),
        d.select("player_id", "group", "pos", "draft_year", *control).join(lab, on="player_id", how="left"),
    ).with_columns((pl.col("pressure_rate") * 100).alias("pressure_pct"))
    ctrl = [*control, "log_plays"]
    b1 = betas_long(chain, "log_get_off", ["log_a10"], "pos", ["log_a10"], "")[0]
    b2 = betas_long(chain, "pressure_rate", [*ctrl, "log_get_off"], "pos", ["log_get_off"], "")[0]
    b3 = betas_long(chain, "pressure_rate", [*ctrl, "log_a10"], "pos", ["log_a10"], "")[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    panels = [
        (axes[0], "log_get_off", "log_a10", [], b1, "Combine start (10-yard split as a10), % vs position mean",
         "game get-off time, % vs position mean (lower = faster)", "A fast Combine start means a fast get-off", 100, 100),
        (axes[1], "pressure_pct", "log_get_off", ctrl, b2, "game get-off time, % vs position mean (lower = faster)",
         "pressure rate, points vs expected (draft, snaps)", "A fast get-off means more pressure", 100, 1),
    ]  # fmt: skip
    for ax, y, x, c, b, xl, yl, title, sx, sy in panels:
        xv, yv = _partial(chain, y, x, c, "pos")
        xv, yv = xv * sx, yv * sy
        ax.scatter(xv, yv, s=16, color=GROUP_COLOURS["DL"], alpha=0.7)
        slope = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 50)
        ax.plot(xs, np.polyval(slope, xs), color="#333333", lw=1.2)
        ax.text(0.03, 0.95, f"β {b['beta']:+.2f} (95% CI {b['lo95']:+.2f} to {b['hi95']:+.2f}), n {b['n']}",
                transform=ax.transAxes, fontsize=8, va="top")  # fmt: skip
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
    fig.suptitle(
        f"Defensive line, within NFL position (DE, OLB, DT, NT): the Combine start reaches pressure through the get-off"
        f" (start → pressure β {b3['beta']:+.2f} beyond the draft)",
        fontsize=9, x=0.01, ha="left", color="#333333",
    )  # fmt: skip
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# 5. How much top speed reaches games --------------------------------------------------------------------------


def fig_eta(processed, interim, path) -> None:
    """Left: share of Combine top speed reached in games (η_speed, career) by group. Right: within group, against
    Combine top speed (bigger engines reach a smaller share)."""
    import matplotlib.pyplot as plt

    groups = pl.read_parquet(interim / "combine_si.parquet").select("player_id", pl.col("pos_group").alias("group"))
    eta = pl.read_parquet(processed / "eta.parquet")
    lab = eta.group_by("player_id").agg(pl.col("lab_speed").first())
    car = pl.read_parquet(processed / "eta_career.parquet").select("player_id", "eta_speed_career")
    d = car.join(lab, on="player_id").join(groups, on="player_id")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    rng = np.random.default_rng(0)
    for i, g in enumerate(POS_ORDER):
        v = d.filter(pl.col("group") == g)["eta_speed_career"].to_numpy() * 100
        ax0.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops={"color": "#333333"})
        ax0.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, alpha=0.6, color=GROUP_COLOURS[g])
    meds = [np.median(d.filter(pl.col("group") == g)["eta_speed_career"].to_numpy()) * 100 for g in POS_ORDER]
    ax0.set_xticks(range(len(POS_ORDER)), [f"{g}\nmedian {m:.0f}%" for g, m in zip(POS_ORDER, meds)])
    ax0.set_ylabel("game top speed as % of Combine top speed")
    ax0.set_title("How much of the 40 reaches the game")

    c = d.with_columns(
        ((pl.col("lab_speed").log() - pl.col("lab_speed").log().mean().over("group")) * 100).alias("x"),
        ((pl.col("eta_speed_career").log() - pl.col("eta_speed_career").log().mean().over("group")) * 100).alias("y"),
    )
    for g in POS_ORDER:
        x = c.filter(pl.col("group") == g)
        ax1.scatter(x["x"], x["y"], s=10, alpha=0.7, color=GROUP_COLOURS[g], label=g)
    r = float(np.corrcoef(c["x"], c["y"])[0, 1])
    slope = np.polyfit(c["x"], c["y"], 1)
    xs = np.linspace(c["x"].min(), c["x"].max(), 50)
    ax1.plot(xs, np.polyval(slope, xs), color="#333333", lw=1.2)
    ax1.text(0.97, 0.95, f"within-group r {r:+.2f}", transform=ax1.transAxes, ha="right", va="top", fontsize=8)
    ax1.axhline(0, color=GREY, lw=0.5)
    ax1.axvline(0, color=GREY, lw=0.5)
    ax1.set_xlabel("Combine top speed, % vs position-group mean")
    ax1.set_ylabel("share reached in games, % vs group mean")
    ax1.set_title("Bigger engines reach a smaller share")
    ax1.legend(ncol=5, loc="lower left")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def main() -> None:
    from eeg import config
    from eeg.config import PROJECT_ROOT

    _style()
    cfg = config.load()
    interim, processed, figs = (PROJECT_ROOT / d for d in ("data/interim", "data/processed", "reports/figures"))
    ex = cfg["figures"]["example_prospect"]
    m = (
        pl.read_parquet(processed / "mismatch.parquet")
        .filter(pl.col("player_id") == ex["player_id"])
        .row(0, named=True)
    )
    label = (
        f"{ex['label']}: pick {m['draft_overall_pick']}, expected {m['expected_snaps_per_season']:.0f} snaps a season"
        f" from that slot, played {m['scrim_per_season']:.0f}"
    )
    fig_physics(interim, processed, figs / "final_1_physics.png", ex["player_id"], label, cfg)
    fig_priced(interim, processed, figs / "final_2_priced.png", cfg)
    fig_tail(interim, processed, figs / "final_3_tail.png", cfg)
    fig_dl_chain(interim, processed, figs / "final_4_dl_chain.png", cfg)
    fig_eta(processed, interim, figs / "final_5_eta.png")
    print("wrote", ", ".join(sorted(p.name for p in figs.glob("final_*.png"))))


if __name__ == "__main__":
    main()
