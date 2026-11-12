import pytest

from rescuebot_core import DOWN, LEFT, RIGHT, SCAN, UP, RescueConfig, value_iteration


def test_move_cost_even_when_failed(model):
    r, q = model.idx((0, 0)), model.idx((3, 3))
    rew, comp = model.reward(r, RIGHT, r, q)  # failed move
    assert rew == -10 and comp["move_cost"] == -10


def test_boundary_move_still_costs(model):
    r, q = model.idx((0, 0)), model.idx((3, 3))
    assert model.reward(r, UP, r, q)[0] == -10


def test_hazard_entry_penalty(model):
    r, q = model.idx((1, 0)), model.idx((3, 3))
    rew, comp = model.reward(r, RIGHT, model.idx((1, 1)), q)
    assert rew == -60 and comp["hazard"] == -50


def test_staying_in_hazard_not_repenalised(model):
    h, q = model.idx((1, 1)), model.idx((3, 3))
    assert model.reward(h, RIGHT, h, q)[0] == -10


def test_rescue_reward(model):
    r, q = model.idx((3, 2)), model.idx((3, 3))
    rew, comp = model.reward(r, RIGHT, q, q)
    assert rew == 90 and comp["rescue"] == 100


def test_scan_free_by_default_and_configurable(model):
    r, q = model.idx((0, 0)), model.idx((3, 3))
    assert model.reward(r, SCAN, r, q)[0] == 0
    m2 = RescueConfig(scan_cost=2.0).public_model()
    assert m2.reward(r, SCAN, r, q)[0] == -2


def test_terminal_has_no_reward(model):
    q = model.idx((3, 3))
    for a in (UP, DOWN, LEFT, RIGHT, SCAN):
        assert model.reward(q, a, q, q)[0] == 0


def test_value_iteration_consistency(model):
    Q, V = value_iteration(model)
    # adjacent survivor: V = -10 + 0.8*100 + 0.2*0.95*V  ->  V = 70 / 0.81
    r, q = model.idx((3, 2)), model.idx((3, 3))
    assert V[r, q] == pytest.approx(70 / (1 - 0.2 * 0.95), rel=1e-6)
    assert (V >= -1e-9).all()
