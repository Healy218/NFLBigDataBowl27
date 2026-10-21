"""Smoothed derivatives along tracks (game plays or Combine reps), the same estimator for field and lab.

Tangential acceleration is the Savitzky–Golay derivative of the reported NGS
speed, not of x/y. NGS `a` is the magnitude of total acceleration (it tracks
|a| from x/y with r 0.88, but |dv/dt| with r 0.61), so it includes turning
and has no braking sign. The lab τ was fit to the same reported speed, so the
derivative of speed is the like-for-like field estimator.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_coeffs


def savgol_track(
    values: np.ndarray,
    frame_id: np.ndarray,
    n_frames: np.ndarray,
    window: int,
    polyorder: int = 2,
    deriv: int = 1,
    dt: float = 0.1,
) -> np.ndarray:
    """Savitzky–Golay smoothing or derivative of tracks concatenated in order.

    ``values`` holds whole tracks back to back, each sorted by time; ``frame_id``
    is the 0-based index within its track and ``n_frames`` the track's length
    (both per row). Frames within window // 2 of a track's ends are NaN, so no
    window crosses two tracks.
    """
    values = np.asarray(values, float)
    if len(values) < window:
        return np.full(len(values), np.nan)
    coeffs = savgol_coeffs(window, polyorder, deriv=deriv, delta=dt, use="conv")
    out = np.convolve(values, coeffs, mode="same")
    h = window // 2
    out[(frame_id < h) | (frame_id >= n_frames - h)] = np.nan
    return out
