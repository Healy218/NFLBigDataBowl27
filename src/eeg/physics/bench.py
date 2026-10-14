"""Bench press: concentric work against the 225 lb bar.

Inputs may be floats, numpy arrays, polars Series or polars expressions; α is a scalar.
"""

from eeg.ingest.units import BAR_KG, G


def bench_rom(arm_m, alpha: float):
    """Bar travel per rep [m] = α × Combine arm length (shoulder to fingertip)."""
    return alpha * arm_m


def work_bench(reps, arm_m, alpha: float, bar_kg: float = BAR_KG, g: float = G):
    """W = n_reps · m_bar · g · ROM [J], concentric phase only."""
    return reps * bar_kg * g * bench_rom(arm_m, alpha)
