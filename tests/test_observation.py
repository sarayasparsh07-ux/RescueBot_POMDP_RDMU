import numpy as np
import pytest

from rescuebot_core import (HEAT_OBS, OBS_COLD, OBS_HOT, OBS_NONE, OBS_WARM, RescueConfig,
                            sensor_with_noise)


def test_heat_likelihoods_sum_to_one(model):
    tot = sum(model.obs_lik[o] for o in HEAT_OBS)
    np.testing.assert_allclose(tot, 1.0)
    assert np.all(model.obs_lik >= 0)


def test_none_observation_has_unit_likelihood(model):
    assert np.all(model.obs_lik[OBS_NONE] == 1.0)


@pytest.mark.parametrize("d,expected", [(0, (0.05, 0.20, 0.75)), (1, (0.05, 0.20, 0.75)),
                                        (2, (0.20, 0.65, 0.15)), (3, (0.20, 0.65, 0.15)),
                                        (4, (0.75, 0.20, 0.05)), (6, (0.75, 0.20, 0.05))])
def test_distance_bands(model, d, expected):
    # find a pair of cells at distance d
    for i in range(model.n):
        for j in range(model.n):
            if model.dist[i, j] == d:
                got = tuple(model.obs_lik[o, i, j] for o in (OBS_COLD, OBS_WARM, OBS_HOT))
                assert got == pytest.approx(expected)
                return
    pytest.fail("no pair found")


def test_invalid_sensor_table_rejected():
    with pytest.raises(ValueError):
        RescueConfig(sensor_table=((0.5, 0.5, 0.5), (0.2, 0.65, 0.15), (0.75, 0.2, 0.05)))


@pytest.mark.parametrize("level", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_noise_family_is_valid(level):
    tab = np.array(sensor_with_noise(level))
    np.testing.assert_allclose(tab.sum(axis=1), 1.0)
    assert np.all(tab >= 0)


def test_noise_family_endpoints():
    np.testing.assert_allclose(sensor_with_noise(1.0), RescueConfig().sensor_table)
    np.testing.assert_allclose(sensor_with_noise(2.0), np.full((3, 3), 1 / 3))
    assert np.allclose(np.array(sensor_with_noise(0.0)).max(axis=1), 1.0)
