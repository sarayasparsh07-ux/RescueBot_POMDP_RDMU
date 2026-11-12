import numpy as np
import pytest

from rescuebot_core import ACTIONS, MOVE_ACTIONS, SCAN, UP, DOWN, LEFT, RIGHT, RescueConfig


def test_full_transition_rows_sum_to_one(model):
    for a in ACTIONS:
        states, T = model.full_transition_matrix(a)
        assert np.all(T >= 0)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-12)
        # 16 robot cells x 14 safe survivor cells (terminal states included)
        assert len(states) == 16 * 14


def test_successful_and_failed_move(model):
    s = (model.idx((1, 0)), model.idx((3, 3)))
    dist = model.transition_distribution(s, RIGHT)
    assert dist[(model.idx((1, 1)), s[1])] == pytest.approx(0.8)
    assert dist[(model.idx((1, 0)), s[1])] == pytest.approx(0.2)
    assert len(dist) == 2


def test_boundary_move_aggregates_to_certain_stay(model):
    s = (model.idx((0, 0)), model.idx((3, 3)))
    for a in (UP, LEFT):
        dist = model.transition_distribution(s, a)
        assert dist == {s: pytest.approx(1.0)}


def test_scan_never_moves(model):
    for r in range(model.n):
        s = (r, model.idx((3, 3)))
        if r == s[1]:
            continue
        assert model.transition_distribution(s, SCAN) == {s: 1.0}


def test_survivor_static_in_every_transition(model):
    for a in ACTIONS:
        for s in model.state_space():
            for s2 in model.transition_distribution(s, a):
                assert s2[1] == s[1]


def test_terminal_states_absorbing(model):
    q = model.idx((2, 3))
    for a in ACTIONS:
        assert model.transition_distribution((q, q), a) == {(q, q): 1.0}


def test_no_sideways_movement(model):
    s = (model.idx((2, 1)), model.idx((0, 3)))
    for a in MOVE_ACTIONS:
        outs = {model.cell(r) for (r, _) in model.transition_distribution(s, a)}
        assert outs <= {(2, 1), model.cell(model.next_cell[a][s[0]])}


@pytest.mark.parametrize("p", [0.0, 0.5, 0.95, 1.0])
def test_configurable_success_probability(p):
    m = RescueConfig(p_success=p).public_model()
    for a in ACTIONS:
        np.testing.assert_allclose(m.move_matrix[a].sum(axis=1), 1.0)
    assert m.move_matrix[DOWN, 0, 4] == pytest.approx(p)
