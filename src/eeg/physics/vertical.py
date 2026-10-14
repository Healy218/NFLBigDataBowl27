"""Vertical jump: apex potential energy and Sayers peak power.

Inputs may be floats, numpy arrays, polars Series or polars expressions.
"""

from eeg.ingest.units import G, m_to_cm


def energy_vertical(mass_kg, height_m, g: float = G):
    """E = m g h [J]: energy at the apex of the CoM rise (excludes takeoff-phase work)."""
    return mass_kg * g * height_m


def sayers_peak_power(mass_kg, height_m, a: float, b: float, c: float):
    """P_peak = a * h[cm] + b * m[kg] + c [W] (Sayers et al., 1999).

    Peak power, not energy: keep it as a separate channel, never add it to Joules.
    Out of sample for heavy linemen; see config vertical.sayers_out_of_sample_groups.
    """
    return a * (height_m * m_to_cm) + b * mass_kg + c
