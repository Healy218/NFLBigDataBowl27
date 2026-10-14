"""Broad jump: takeoff kinetic energy from projectile range.

Treats the jump as a projectile launched and landing at the same height, so
R = v0^2 sin(2θ) / g. At θ = 45° this is the minimum v0 for the range, i.e. a
lower bound on takeoff KE. Inputs may be floats, numpy arrays, polars Series or
polars expressions; θ is a scalar.
"""

import math

from eeg.ingest.units import G


def _sin_2theta(theta_deg: float) -> float:
    if not 0.0 < theta_deg < 90.0:
        raise ValueError(f"takeoff angle must be in (0, 90) deg, got {theta_deg}")
    return math.sin(math.radians(2.0 * theta_deg))


def takeoff_velocity(range_m, theta_deg: float, g: float = G):
    """v0 = sqrt(R g / sin 2θ) [m/s]."""
    return (range_m * g / _sin_2theta(theta_deg)) ** 0.5


def energy_broad(mass_kg, range_m, theta_deg: float, g: float = G):
    """E = ½ m v0² = ½ m R g / sin 2θ [J]."""
    return 0.5 * mass_kg * range_m * g / _sin_2theta(theta_deg)
