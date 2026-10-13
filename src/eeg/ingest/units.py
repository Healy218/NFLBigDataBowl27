"""Exact unit conversions and definitional physical constants.

Modelling assumptions (drag, takeoff angle, ROM fraction, ...) live in
config/physics.yaml; only exact, non-tunable values belong here.
"""

# Length (international yard and pound agreement, 1959)
in_to_m = 0.0254
ft_to_m = 12 * in_to_m  # 0.3048
yd_to_m = 3 * ft_to_m  # 0.9144
m_to_cm = 100.0

# Mass
lb_to_kg = 0.45359237

# Standard gravity (CGPM 1901), m/s^2
G = 9.80665

# Combine bench press bar: 225 lb, in kg (~102.058)
BAR_LB = 225.0
BAR_KG = BAR_LB * lb_to_kg
