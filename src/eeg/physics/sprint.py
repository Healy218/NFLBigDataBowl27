"""40-yard dash: mono-exponential sprint fit, kinetic energy and drag work.

Model (Samozino-style), with start time t0 free because the tracked rep window
starts 0.17-0.29 s before the official clock:

    v(t) = v_max (1 - exp(-(t - t0)/tau))                       for t > t0, else 0
    x(t) = v_max [(t - t0) - tau (1 - exp(-(t - t0)/tau))]

E_40 = ½ m v_max² + W_drag, with W_drag = ∫ ½ ρ C_d A v³ dt from t0 to the finish.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import curve_fit, least_squares

from eeg.ingest import units as u

START_LINE_YD = 10.0
TEN_YD_LINE_YD = 20.0
FINISH_LINE_YD = 50.0
D10_M = 10 * u.yd_to_m
D40_M = 40 * u.yd_to_m

# curve_fit bounds: (v_max m/s, tau s, t0 s relative to the first tracked frame)
_LO = (3.0, 0.2, -2.0)
_HI = (13.0, 3.0, 1.5)


def speed_model(t, v_max, tau, t0):
    dt = np.maximum(np.asarray(t, dtype=float) - t0, 0.0)
    return v_max * (1.0 - np.exp(-dt / tau))


def distance_model(t, v_max, tau, t0):
    dt = np.maximum(np.asarray(t, dtype=float) - t0, 0.0)
    return v_max * (dt - tau * (1.0 - np.exp(-dt / tau)))


def time_to_distance(d_m, v_max, tau):
    """Time from t0 to cover d_m metres under the model (Newton on x(t) = d)."""
    t = d_m / v_max + tau  # good start: x(t) ≈ v_max (t - tau) once at speed
    for _ in range(50):
        f = distance_model(t, v_max, tau, 0.0) - d_m
        t_new = t - f / speed_model(t, v_max, tau, 0.0)
        if abs(t_new - t) < 1e-10:
            return float(t_new)
        t = t_new
    return float(t)


def drag_work(v_max, tau, duration_s, rho, cd, area_m2):
    """∫_0^T ½ ρ C_d A v(t)³ dt for v = v_max (1 - e^{-t/τ}), in closed form [J]."""
    e1, e2, e3 = (np.exp(-k * duration_s / tau) for k in (1, 2, 3))
    integral = duration_s + 3 * tau * (e1 - 1) - 1.5 * tau * (e2 - 1) + tau / 3 * (e3 - 1)
    return 0.5 * rho * cd * area_m2 * v_max**3 * integral


def body_surface_area(height_m, mass_kg):
    """Du Bois BSA [m²] = 0.007184 · W[kg]^0.425 · H[cm]^0.725."""
    return 0.007184 * mass_kg**0.425 * (height_m * u.m_to_cm) ** 0.725


def frontal_area(height_m, mass_kg, factor: float):
    """Projected frontal area A = factor · BSA [m²]."""
    return factor * body_surface_area(height_m, mass_kg)


def kinetic_energy(mass_kg, v_max):
    return 0.5 * mass_kg * v_max**2


@dataclass(frozen=True)
class SprintFit:
    v_max: float
    tau: float
    t0: float
    rmse: float
    n_frames: int


def fit_trace(t_s: np.ndarray, s_mps: np.ndarray) -> SprintFit:
    """Least-squares fit of speed_model to a measured speed trace."""
    p0 = (min(max(float(np.max(s_mps)), _LO[0] + 0.1), _HI[0] - 0.1), 1.0, -0.2)
    popt, _ = curve_fit(speed_model, t_s, s_mps, p0=p0, bounds=(_LO, _HI), maxfev=10_000)
    rmse = float(np.sqrt(np.mean((speed_model(t_s, *popt) - s_mps) ** 2)))
    return SprintFit(*map(float, popt), rmse=rmse, n_frames=len(t_s))


def fit_splits(t10_s: float, t40_s: float) -> tuple[float, float]:
    """(v_max, tau) such that x(t10) = 10 yd and x(t40) = 40 yd, with t0 = 0.

    Fallback for prospects with an official 10-yd split and 40 but no trace.
    """

    def resid(p):
        v_max, tau = p
        return [distance_model(t10_s, v_max, tau, 0.0) - D10_M, distance_model(t40_s, v_max, tau, 0.0) - D40_M]

    sol = least_squares(resid, x0=(D40_M / t40_s * 1.2, 1.0), bounds=((3.0, 0.05), (15.0, 3.0)))
    return float(sol.x[0]), float(sol.x[1])


def crossing_time(t: np.ndarray, x: np.ndarray, line: float) -> float:
    """Interpolated time of the first crossing of x = line (NaN if never crossed)."""
    idx = np.flatnonzero((x[:-1] < line) & (x[1:] >= line))
    if idx.size == 0:
        return math.nan
    i = idx[0]
    return float(t[i] + (line - x[i]) / (x[i + 1] - x[i]) * (t[i + 1] - t[i]))


def fit_forty_reps(frames: pl.DataFrame) -> pl.DataFrame:
    """One row per 40 rep: fitted (v_max, tau, t0), fit quality and timing checks.

    ``frames`` is data/interim/timed_drills.parquet (any drills; the 40 is selected).
    Fits use frames up to the finish-line crossing (x = 50 yd).
    t40_model = t_finish - t0 is the model's equivalent of the official time.
    """
    rows = []
    forty = frames.filter(pl.col("drill") == "forty").sort("rep_id", "t_s")
    for (rep_id,), r in forty.group_by(["rep_id"], maintain_order=True):
        t = r["t_s"].to_numpy()
        x = r["x_yd"].to_numpy()
        s = r["s_mps"].to_numpy()
        t50 = crossing_time(t, x, FINISH_LINE_YD)
        t20 = crossing_time(t, x, TEN_YD_LINE_YD)
        keep = t <= (t50 if not math.isnan(t50) else t[-1]) + 1e-9
        fit = fit_trace(t[keep], s[keep])
        rows.append(
            dict(
                rep_id=rep_id,
                player_id=r["player_id"][0],
                draft_year=r["draft_year"][0],
                v_max=fit.v_max,
                tau=fit.tau,
                t0=fit.t0,
                rmse=fit.rmse,
                n_frames=fit.n_frames,
                t_finish=t50,
                t40_model=t50 - fit.t0,
                fly30_trk=t50 - t20,
            )
        )
    return pl.DataFrame(rows)


def best_reps(reps: pl.DataFrame, max_rmse: float) -> pl.DataFrame:
    """Per player, the well-fit rep with the fastest modelled 40.

    The official time is the best attempt, so the fastest rep is the matching one.
    Reps with fit RMSE > max_rmse are excluded; players with no usable rep drop out.
    """
    return (
        reps.filter(pl.col("t40_model").is_not_nan() & (pl.col("rmse") <= max_rmse))
        .sort("t40_model")
        .group_by("player_id", maintain_order=True)
        .first()
        .join(reps.group_by("player_id").agg(pl.len().alias("n_reps")), on="player_id")
    )


def forty_energy(best: pl.DataFrame, prospects: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Per player E_40 = ½ m v_max² + W_drag over the modelled run (t0 -> finish).

    ``best`` from best_reps, ``prospects`` from combine_si.parquet (player_id, mass_kg, ht_m),
    ``cfg`` the ``sprint`` section of config.load().
    """
    df = best.join(prospects.select("player_id", "mass_kg", "ht_m"), on="player_id", how="left")
    area = frontal_area(df["ht_m"].to_numpy(), df["mass_kg"].to_numpy(), cfg["frontal_area_bsa_factor"])
    w_drag = drag_work(
        df["v_max"].to_numpy(), df["tau"].to_numpy(), df["t40_model"].to_numpy(),
        cfg["air_density"], cfg["drag_coefficient"], area,
    )  # fmt: skip
    return df.with_columns(
        pl.Series("frontal_area_m2", area),
        kinetic_energy(pl.col("mass_kg"), pl.col("v_max")).alias("KE_40"),
        pl.Series("W_drag_40", w_drag),
    ).with_columns((pl.col("KE_40") + pl.col("W_drag_40")).alias("E_40"))


def main() -> None:
    from eeg import config
    from eeg.config import PROJECT_ROOT

    interim = PROJECT_ROOT / "data" / "interim"
    reps = fit_forty_reps(pl.read_parquet(interim / "timed_drills.parquet"))
    reps.write_parquet(interim / "forty_fits.parquet")
    cfg = config.load()["sprint"]
    best = best_reps(reps, cfg["max_fit_rmse_mps"])
    out = forty_energy(best, pl.read_parquet(interim / "combine_si.parquet"), cfg)
    print(f"{reps.height} reps fit -> forty_fits.parquet; {out.height} players with E_40")
    print(out.select("v_max", "tau", "rmse", "t40_model", "KE_40", "W_drag_40", "E_40").describe())


if __name__ == "__main__":
    main()
