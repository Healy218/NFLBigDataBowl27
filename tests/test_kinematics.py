import numpy as np
import pytest

from eeg.ngs.kinematics import savgol_track


def _tracks(*lengths):
    frame_id = np.concatenate([np.arange(n) for n in lengths])
    n_frames = np.concatenate([np.full(n, n) for n in lengths])
    return frame_id, n_frames


def test_derivative_is_exact_for_quadratics_and_masks_track_ends():
    fid, n = _tracks(10, 8)
    t = 0.1 * fid
    v = np.where(n == 10, 3.0 * t**2, 5.0 - 2.0 * t)  # two tracks with different laws
    d = savgol_track(v, fid, n, window=5)
    first, second = d[:10], d[10:]
    assert np.isnan(first[[0, 1, 8, 9]]).all() and np.isnan(second[[0, 1, 6, 7]]).all()
    assert first[2:8] == pytest.approx(6.0 * t[2:8])
    assert second[2:6] == pytest.approx(-2.0)  # no window mixes the two tracks


def test_smoothing_and_short_tracks():
    fid, n = _tracks(9)
    v = np.full(9, 4.0)
    assert savgol_track(v, fid, n, window=7, deriv=0)[3:6] == pytest.approx(4.0)
    fid, n = _tracks(3)
    assert np.isnan(savgol_track(np.ones(3), fid, n, window=5)).all()
