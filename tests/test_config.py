import pytest

from eeg import config


@pytest.fixture(scope="module")
def raw():
    return config.load_raw()


def test_defaults_match_handoff(raw):
    v = config.values(raw)
    assert v["broad"]["theta_deg"] == 45.0
    assert v["bench"]["rom_arm_fraction"] == 0.50
    assert v["sprint"]["air_density"] == 1.18
    assert v["sprint"]["drag_coefficient"] == 0.9
    assert v["sprint"]["frontal_area_bsa_factor"] == 0.266
    assert (v["vertical"]["sayers_a"], v["vertical"]["sayers_b"], v["vertical"]["sayers_c"]) == (
        60.7,
        45.3,
        -2055.0,
    )


def test_swept_parameters(raw):
    keys = {p.key for p in config.sweep_params(raw)}
    assert keys == {
        "broad.theta_deg",
        "sprint.air_density",
        "sprint.drag_coefficient",
        "sprint.frontal_area_bsa_factor",
        "bench.rom_arm_fraction",
    }
    for p in config.sweep_params(raw):
        assert p.low < p.high
        assert p.low <= p.value <= p.high


def test_with_override_does_not_mutate(raw):
    out = config.with_override(raw, "broad.theta_deg", 30.0)
    assert out["broad"]["theta_deg"]["value"] == 30.0
    assert raw["broad"]["theta_deg"]["value"] == 45.0


@pytest.mark.parametrize(
    "block, msg",
    [
        ({"low": 1.0, "high": 2.0}, "must have a 'value'"),
        ({"value": 1.5, "low": 1.0}, "both 'low' and 'high'"),
        ({"value": 3.0, "low": 1.0, "high": 2.0}, "outside"),
    ],
)
def test_validate_rejects_bad_blocks(block, msg):
    with pytest.raises(ValueError, match=msg):
        config.validate({"sec": {"p": block}})
