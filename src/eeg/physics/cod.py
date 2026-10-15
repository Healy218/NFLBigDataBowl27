"""Change of direction (20-yd shuttle, 3-cone): rep segmentation and KE work.

    uv run python -m eeg.physics.cod

Work is the total absolute change in kinetic energy over the rep,

    W_cod = ½ m Σ |Δ(v²)|        (acceleration and braking both count as |W|)

so each straight leg that starts and ends at a plant contributes ≈ m v_peak², the
final leg through the line ½ m v_end², and the curved 3-cone turns count only the
speed they actually shed. Measured speed is used directly, so the nominal 5/10/5
segment lengths are never needed (the sensor turns ~1 yd short of each line).

Segmentation, with u = displacement along the first leg's direction [yd]:
  shuttle  out (u ≥ +min_leg), far leg (u ≤ -min_leg), finish = first u: - -> + crossing
           after the far point (the run already passes u = 0 mid-way through the 10-yd leg)
  cone     out (u ≥ +min_leg), loop round cone 3 (|lateral| ≥ min_leg),
           finish = first u: + -> - crossing after the loop's extreme
If the window ends before the crossing but within finish_tol_yd of the start line,
the window end is the finish (NGS cuts many windows at the finish event).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

COD_DRILLS = ("shuttle", "cone")
PLANT_SPEED_MPS = 1.5  # interior speed minima below this count as plants
# Both drills have two hard plants before the finish (the 3-cone adds a slower turn
# round cone 3). Fewer means the window opened mid-drill.
MIN_PLANTS = {"shuttle": 2, "cone": 2}


def ke_variation(s_mps: np.ndarray) -> float:
    """Σ |Δ(v²)| over a speed trace [m²/s²]; W_cod = ½ m · this."""
    return float(np.sum(np.abs(np.diff(np.asarray(s_mps, dtype=float) ** 2))))


def path_length(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def first_leg_sign(dx: np.ndarray, min_yd: float) -> float:
    """Direction (+1/-1) of the first leg: sign of dx when |dx| first reaches min_yd."""
    idx = np.flatnonzero(np.abs(dx) >= min_yd)
    return float(np.sign(dx[idx[0]])) if idx.size else math.nan


def crossing_after(t: np.ndarray, v: np.ndarray, i0: int, rising: bool) -> tuple[float, int | None]:
    """Interpolated time of the first zero crossing of v at or after index i0.

    Returns (time, index of the frame before the crossing), or (nan, None).
    """
    a, b = v[i0:-1], v[i0 + 1 :]
    hit = (a <= 0) & (b > 0) if rising else (a > 0) & (b <= 0)
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return math.nan, None
    i = i0 + int(idx[0])
    return float(t[i] + (0 - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])), i


def onset_time(t: np.ndarray, s: np.ndarray, speed: float) -> float:
    """Interpolated time speed first reaches `speed` (t[0] if the rep already starts above it)."""
    idx = np.flatnonzero(s >= speed)
    if idx.size == 0:
        return math.nan
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (speed - s[i - 1]) / (s[i] - s[i - 1]) * (t[i] - t[i - 1]))


def count_plants(s: np.ndarray, speed: float = PLANT_SPEED_MPS) -> int:
    interior = s[1:-1]
    return int(np.sum((interior < s[:-2]) & (interior <= s[2:]) & (interior < speed)))


def segment_rep(drill: str, t, x_yd, y_yd, s_mps, cfg: dict) -> dict:
    """Validate one rep and find its finish. ``cfg`` is the ``cod`` config section.

    Returns valid, reason, t_fin, i_fin (last frame index before the finish), finish_rule.
    """
    t, x, y, s = (np.asarray(a, dtype=float) for a in (t, x_yd, y_yd, s_mps))
    min_leg, tol = cfg["min_leg_yd"], cfg["finish_tol_yd"]
    out = dict(valid=False, reason="", t_fin=math.nan, i_fin=None, finish_rule=None)
    sgn = first_leg_sign(x - x[0], min_leg)
    if math.isnan(sgn):
        return out | dict(reason="no_first_leg")
    uu = (x - x[0]) * sgn

    if drill == "shuttle":
        i_far = int(np.argmin(uu))
        if uu[i_far] > -min_leg:
            return out | dict(reason="no_far_leg")
        rising = True
    elif drill == "cone":
        dy = y - y[0]
        i_far = int(np.argmax(np.abs(dy)))
        if abs(dy[i_far]) < min_leg or uu[: i_far + 1].max() < min_leg:
            return out | dict(reason="no_cone_loop")
        rising = False
    else:
        raise ValueError(f"unknown COD drill {drill!r}")

    t_fin, i = crossing_after(t, uu, i_far, rising)
    if i is not None:
        return out | dict(valid=True, t_fin=t_fin, i_fin=i, finish_rule="crossing")
    if abs(uu[-1]) <= tol:
        return out | dict(valid=True, t_fin=float(t[-1]), i_fin=len(t) - 1, finish_rule="window_end")
    return out | dict(reason="no_finish")


def trim_to_finish(t: np.ndarray, s: np.ndarray, t_fin: float, i_fin: int) -> np.ndarray:
    """Speed trace from the first frame to t_fin, with an interpolated end sample."""
    if i_fin >= len(t) - 1:
        return s.copy()
    s_end = s[i_fin] + (t_fin - t[i_fin]) / (t[i_fin + 1] - t[i_fin]) * (s[i_fin + 1] - s[i_fin])
    return np.append(s[: i_fin + 1], s_end)


def segment_cod_reps(frames: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """One row per shuttle / 3-cone rep: validity, finish, timing and mass-free work.

    ``frames`` is data/interim/timed_drills.parquet. sum_dv2 = Σ|Δ(v²)| from the last
    frame below onset speed to the finish, so W_cod = ½ m sum_dv2 (idle jitter before
    the run is excluded). Reps that finish below cfg["min_finish_speed_ratio"] · v_peak
    were jogged out (reason "no_finish_sprint"); reps with fewer than MIN_PLANTS plants
    opened mid-drill (reason "truncated_start"). t_trk = t_fin - t_on (onset at
    cfg["onset_speed_mps"]) is the tracked counterpart of the official time.
    """
    rows = []
    cod = frames.filter(pl.col("drill").is_in(COD_DRILLS)).sort("rep_id", "t_s")
    for (rep_id,), r in cod.group_by(["rep_id"], maintain_order=True):
        drill = r["drill"][0]
        t, s = r["t_s"].to_numpy(), r["s_mps"].to_numpy()
        seg = segment_rep(drill, t, r["x_yd"].to_numpy(), r["y_yd"].to_numpy(), s, cfg)
        row = dict(
            rep_id=rep_id, player_id=r["player_id"][0], draft_year=r["draft_year"][0], drill=drill,
            valid=seg["valid"], reason=seg["reason"], finish_rule=seg["finish_rule"], t_fin=seg["t_fin"],
            t_on=math.nan, t_trk=math.nan, path_m=math.nan, v_peak=math.nan, v_fin=math.nan, n_plants=None,
            sum_dv2=math.nan,
        )  # fmt: skip
        if seg["valid"]:
            i = seg["i_fin"]
            t_on = onset_time(t, s, cfg["onset_speed_mps"])
            i0 = max(int(np.searchsorted(t, t_on)) - 1, 0)  # last frame below onset speed
            st = trim_to_finish(t, s, seg["t_fin"], i)[i0:]
            row |= dict(
                t_on=t_on,
                t_trk=seg["t_fin"] - t_on,
                path_m=path_length(r["x_m"].to_numpy()[i0 : i + 1], r["y_m"].to_numpy()[i0 : i + 1]),
                v_peak=float(st.max()),
                v_fin=float(st[-1]),
                n_plants=count_plants(st),
                sum_dv2=ke_variation(st),
            )
            if row["n_plants"] < MIN_PLANTS[drill]:
                row |= dict(valid=False, reason="truncated_start")
            elif row["v_fin"] < cfg["min_finish_speed_ratio"] * row["v_peak"]:
                row |= dict(valid=False, reason="no_finish_sprint")
        rows.append(row)
    return pl.DataFrame(rows, schema_overrides={"n_plants": pl.Int64, "finish_rule": pl.String})


def best_reps(reps: pl.DataFrame) -> pl.DataFrame:
    """Per player and drill, the valid rep with the fastest tracked time (as for the 40)."""
    return (
        reps.filter(pl.col("valid") & pl.col("t_trk").is_not_nan())
        .sort("t_trk")
        .group_by("player_id", "drill", maintain_order=True)
        .first()
        .join(reps.group_by("player_id", "drill").agg(pl.len().alias("n_reps")), on=["player_id", "drill"])
        .sort("drill", "player_id")
    )


def cod_energy(best: pl.DataFrame, prospects: pl.DataFrame) -> pl.DataFrame:
    """W_cod = ½ m Σ|Δ(v²)| per player and drill, with Combine weigh-in mass [J]."""
    return best.join(prospects.select("player_id", "mass_kg"), on="player_id", how="left").with_columns(
        (0.5 * pl.col("mass_kg") * pl.col("sum_dv2")).alias("W_cod")
    )


def main() -> None:
    from eeg import config
    from eeg.config import PROJECT_ROOT

    interim = PROJECT_ROOT / "data" / "interim"
    cfg = config.load()["cod"]
    reps = segment_cod_reps(pl.read_parquet(interim / "timed_drills.parquet"), cfg)
    reps.write_parquet(interim / "cod_reps.parquet")
    out = cod_energy(best_reps(reps), pl.read_parquet(interim / "combine_si.parquet"))
    print(reps.group_by("drill", "valid", "reason", "finish_rule").len().sort("drill", "valid", "reason"))
    for drill in COD_DRILLS:
        d = out.filter(pl.col("drill") == drill)
        print(f"{drill}: {d.height} players with W_cod")
        print(d.select("t_trk", "path_m", "v_peak", "n_plants", "W_cod").describe())


if __name__ == "__main__":
    main()
