import pytest

from eeg.ingest import units as u


def test_exact_factors():
    assert u.in_to_m == 0.0254
    assert u.ft_to_m == pytest.approx(0.3048, abs=1e-15)
    assert u.yd_to_m == pytest.approx(0.9144, abs=1e-15)
    assert u.lb_to_kg == 0.45359237
    assert u.G == 9.80665


def test_bar_mass():
    assert u.BAR_KG == pytest.approx(102.058, abs=5e-4)


@pytest.mark.parametrize("x", [0.0, 1.0, 74.0, 1e3])
def test_round_trips(x):
    assert x * u.in_to_m / u.in_to_m == pytest.approx(x)
    assert x * u.yd_to_m / u.yd_to_m == pytest.approx(x)
    assert x * u.lb_to_kg / u.lb_to_kg == pytest.approx(x)


def test_combine_scale_values():
    # 6-2 prospect, 40 yd, 225 lb
    assert 74 * u.in_to_m == pytest.approx(1.8796)
    assert 40 * u.yd_to_m == pytest.approx(36.576)
    assert 225 * u.lb_to_kg == pytest.approx(102.0583, abs=1e-4)
    # NGS speed 10 yd/s -> m/s
    assert 10 * u.yd_to_m == pytest.approx(9.144)
