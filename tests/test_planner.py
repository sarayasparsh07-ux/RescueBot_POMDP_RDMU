"""Information-discipline and horizon tests for the POMCP planner."""
import numpy as np
import pytest

from rescuebot_core import (ACTIONS, MOVE_ACTIONS, OBS_NONE, OBS_WARM, RIGHT, SCAN, Belief, MissionRunner,
                            POMCPPlanner, RescueConfig, _HNode)


def far_belief(model):
    """Robot spread over the top-left 2x2 block, survivor known to be (4,4)."""
    joint = np.zeros((model.n, model.n))
    q = model.idx((3, 3))
    for cell, p in [((0, 0), 0.4), ((0, 1), 0.3), ((1, 0), 0.3)]:
        joint[model.idx(cell), q] = p
    return Belief(model, joint)


def rollout_trace(model, belief, r, q, seed=0, limit=4):
    pl = POMCPPlanner(model, rollout="belief", seed=seed)
    pl._limit = limit
    trace = []
    pl._rollout(r, q, 0, _HNode(belief.joint.copy()), trace=trace)
    return trace


def test_belief_rollout_ignores_hidden_robot_position(model):
    """Same agent information, different hidden robot cell -> identical rollout actions."""
    b = far_belief(model)
    q = model.idx((3, 3))
    traces = [rollout_trace(model, b, model.idx(cell), q) for cell in [(0, 0), (0, 1), (1, 0)]]
    assert len(traces[0]) == 4  # survivor >= 5 moves away: no rescue within 4 steps
    assert traces[0] == traces[1] == traces[2]


def test_belief_rollout_ignores_hidden_survivor_position(model):
    joint = Belief.initial(model).update(SCAN, OBS_WARM).joint
    b = Belief(model, joint)
    r = model.start_idx
    # candidate survivor cells at Manhattan distance >= 5 from every cell the robot can reach in 4 moves? use limit 1
    traces = [rollout_trace(model, b, r, model.idx(c), limit=1) for c in [(3, 3), (2, 3), (3, 2)]]
    assert traces[0] == traces[1] == traces[2]


def test_oracle_rollout_does_use_hidden_state(model):
    """Contrast: the labelled oracle rollout changes with the hidden robot cell."""
    b = far_belief(model)
    q = model.idx((3, 3))
    outs = []
    for cell in [(0, 1), (1, 0)]:
        pl = POMCPPlanner(model, rollout="oracle", rollout_epsilon=0.0, seed=0)
        pl._limit = 1
        tr = []
        pl._rollout(model.idx(cell), q, 0, _HNode(), trace=tr)
        outs.append(tr)
    assert outs[0] != outs[1]


def test_rollout_belief_update_matches_bayes_filter(model):
    pl = POMCPPlanner(model, seed=0)
    b = Belief.initial(model).update(SCAN, OBS_WARM)
    joint = b.joint.copy()
    for a in [RIGHT, RIGHT, 1, 1]:
        joint = pl._advance_belief(joint, a)
        b = b.update(a, OBS_NONE)
        np.testing.assert_allclose(joint, b.joint, atol=1e-12)


def test_rollout_action_signature_has_no_state():
    import inspect
    params = list(inspect.signature(POMCPPlanner.rollout_action).parameters)
    assert params == ["self", "joint"]


def test_planner_decision_independent_of_hidden_robot(model):
    """The planner only sees the belief: two runs from the same belief and seed agree."""
    b = far_belief(model)
    d1 = POMCPPlanner(model, n_simulations=300, seed=4).select_action(b, steps_remaining=12)
    d2 = POMCPPlanner(model, n_simulations=300, seed=4).select_action(b, steps_remaining=12)
    assert d1.action == d2.action and d1.info["q_values"] == d2.info["q_values"]


@pytest.mark.parametrize("remaining", [1, 2, 5, 10])
def test_search_respects_remaining_horizon(model, remaining):
    for rollout in ("belief", "oracle", "random"):
        pl = POMCPPlanner(model, n_simulations=400, horizon=10, rollout=rollout, seed=1)
        d = pl.select_action(Belief.initial(model), steps_remaining=remaining)
        assert pl.max_depth_reached <= remaining - 1, (rollout, pl.max_depth_reached)
        assert d.info["horizon"] == min(10, remaining)
        assert d.info["steps_remaining"] == remaining


def test_no_reward_after_deadline(model):
    """With one decision left, Scan earns exactly its immediate reward (0) and a move
    from a certain state earns only its one-step expectation."""
    pl = POMCPPlanner(model, n_simulations=2000, seed=2)
    d = pl.select_action(Belief.initial(model), steps_remaining=1)
    assert d.info["q_values"]["Scan"] == 0.0
    joint = np.zeros((model.n, model.n))
    joint[model.idx((3, 2)), model.idx((3, 3))] = 1.0
    d = POMCPPlanner(model, n_simulations=2000, seed=3).select_action(Belief(model, joint), steps_remaining=1)
    assert d.info["q_values"]["Right"] == pytest.approx(0.8 * 90 + 0.2 * -10, abs=4.0)  # no discounted future term
    assert d.info["q_values"]["Left"] == pytest.approx(-10.0)  # (4,3)->(4,2): safe, no rescue possible


def test_runner_passes_remaining_steps(cfg):
    seen = []

    class Rec(POMCPPlanner):
        def select_action(self, belief, ground_truth=None, steps_remaining=None):
            seen.append(steps_remaining)
            return super().select_action(belief, ground_truth, steps_remaining)

    c = cfg.with_(max_steps=5)
    r = MissionRunner(c, Rec(c.public_model(), n_simulations=50, seed=0), env_seed=0)
    r.run()
    assert seen == list(range(5, 5 - len(seen), -1))


def test_rejects_zero_remaining(model):
    with pytest.raises(ValueError):
        POMCPPlanner(model, n_simulations=10).select_action(Belief.initial(model), steps_remaining=0)


# ---------------------------------------------------------------- tree-search mechanics
from rescuebot_core import _ANode  # noqa: E402
import math  # noqa: E402


def test_unvisited_actions_are_tried_first(model):
    """1 simulation expands the root (rollout only); the next 5 must try each action once."""
    pl = POMCPPlanner(model, n_simulations=6, seed=0)
    d = pl.select_action(Belief.initial(model), steps_remaining=30)
    assert sorted(d.info["visits"].values()) == [1, 1, 1, 1, 1]


def test_ucb_formula(model):
    pl = POMCPPlanner(model, exploration=150.0, seed=0)
    node = _HNode()
    node.actions = [_ANode() for _ in ACTIONS]
    stats = [(10, -20.0), (40, -5.0), (5, -60.0), (20, -15.0), (25, -8.0)]
    for an, (n, q) in zip(node.actions, stats):
        an.n, an.q = n, q
    node.n = sum(n for n, _ in stats)
    expected = max(range(5), key=lambda a: stats[a][1] + 150.0 * math.sqrt(math.log(node.n) / stats[a][0]))
    assert pl._ucb_action(node) == expected


def test_ucb_ties_broken_at_random(model):
    pl = POMCPPlanner(model, seed=0)
    node = _HNode()
    node.actions = [_ANode() for _ in ACTIONS]
    for an in node.actions:
        an.n, an.q = 10, 0.0
    node.n = 50
    picks = {pl._ucb_action(node) for _ in range(200)}
    assert picks == set(ACTIONS)


def test_recursive_discounting_and_terminal_stop(model):
    """Deterministic moves, survivor two cells away: return = -10 + 0.95 * (-10 + 100);
    nothing is collected after the rescue even though 10 steps remain."""
    m = RescueConfig(p_success=1.0).public_model()
    pl = POMCPPlanner(m, rollout="oracle", rollout_epsilon=0.0, seed=0)
    pl._limit = 10
    trace = []
    ret = pl._rollout(m.idx((3, 1)), m.idx((3, 3)), 0, _HNode(), trace=trace)
    assert ret == pytest.approx(-10 + 0.95 * 90)
    assert len(trace) == 2


def test_tree_backup_matches_discounted_return(model):
    """With p = 1 and a certain belief, the root value of the rescuing move equals the
    discounted one-step reward exactly (terminal: no further term)."""
    m = RescueConfig(p_success=1.0).public_model()
    joint = np.zeros((m.n, m.n))
    joint[m.idx((3, 2)), m.idx((3, 3))] = 1.0
    d = POMCPPlanner(m, n_simulations=300, seed=0).select_action(Belief(m, joint), steps_remaining=10)
    assert d.info["q_values"]["Right"] == pytest.approx(90.0)
    assert d.action == RIGHT
