import numpy as np
import pytest

from rescuebot_core import (ACTIONS, DEFAULT_POMCP, RIGHT, SCAN, Belief, ClairvoyantPolicy,
                            HiddenStateAccessError, MissionRunner, POMCPPlanner, PolicySpec,
                            RandomPolicy, RescueBotEnv, RescueConfig, RiskAwareHeuristicPolicy,
                            evaluate_policies, run_episode, wilson_interval, bootstrap_ci)


def test_reset_and_step_structure(cfg):
    env = RescueBotEnv(cfg, seed=1)
    assert env.get_observation() == "None"
    res = env.step(SCAN)
    assert res.action == "Scan" and res.observation in ("Cold", "Warm", "Hot")
    assert res.reward == 0 and res.cumulative_reward == 0
    assert env.get_state_for_evaluation()["robot"] == (0, 0)


def test_rescue_applied_once_and_terminal(cfg):
    c = cfg.with_(survivor=(0, 1), p_success=1.0)
    env = RescueBotEnv(c, seed=0)
    res = env.step(RIGHT)
    assert res.rescued and res.done and res.reward == 90
    with pytest.raises(RuntimeError):
        env.step(RIGHT)


def test_truncation_at_episode_limit(cfg):
    env = RescueBotEnv(cfg.with_(max_steps=3), seed=0)
    for _ in range(3):
        res = env.step(SCAN)
    assert res.truncated and res.done and not res.rescued


def test_hidden_state_locked_during_decisions(cfg):
    env = RescueBotEnv(cfg, seed=0)
    env.lock()
    with pytest.raises(HiddenStateAccessError):
        env.get_state_for_evaluation()
    env.unlock()
    assert env.get_state_for_evaluation()["survivor"] == (3, 3)


def test_model_has_no_survivor(cfg):
    m = cfg.public_model()
    assert not hasattr(m, "survivor")
    assert "survivor" not in vars(m)


class _Spy(RandomPolicy):
    """Tries to peek at the environment during its decision."""

    def __init__(self, model, env):
        super().__init__(model, seed=0)
        self.env = env

    def select_action(self, belief, ground_truth=None, steps_remaining=None):
        self.env.get_state_for_evaluation()
        return super().select_action(belief)


def test_runner_blocks_leaking_policy(cfg):
    runner = MissionRunner(cfg, RandomPolicy(cfg.public_model(), seed=0), env_seed=0)
    runner.policy = _Spy(cfg.public_model(), runner.env)
    with pytest.raises(HiddenStateAccessError):
        runner.step()


@pytest.mark.parametrize("policy_cls", [POMCPPlanner, RiskAwareHeuristicPolicy])
def test_policy_independent_of_true_survivor(cfg, policy_cls):
    """Same belief + same planner seed -> same decision, whatever the truth."""
    decisions = []
    for surv in [(3, 3), (0, 3), (2, 0)]:
        c = cfg.with_(survivor=surv)
        kw = {"n_simulations": 200} if policy_cls is POMCPPlanner else {}
        pol = policy_cls(c.public_model(), seed=42, **kw)
        b = Belief.initial(c.public_model()).update(SCAN, 2)
        d = pol.select_action(b)
        decisions.append((d.action, d.info.get("q_values")))
    assert decisions[0] == decisions[1] == decisions[2]


def test_ground_truth_overlay_does_not_change_policy(cfg):
    """Reading ground truth for display between steps must not alter decisions."""
    def run(peek):
        r = MissionRunner(cfg, POMCPPlanner(cfg.public_model(), n_simulations=150, seed=3), env_seed=9)
        acts = []
        while not r.done:
            if peek:
                r.env.get_state_for_evaluation()
            acts.append(r.step()["action"])
        return acts
    assert run(False) == run(True)


def test_pomcp_q_values_and_valid_action(model):
    d = POMCPPlanner(model, n_simulations=300, seed=0).select_action(Belief.initial(model))
    assert d.action in ACTIONS
    assert set(d.info["q_values"]) == {"Up", "Down", "Left", "Right", "Scan"}
    assert sum(d.info["visits"].values()) <= 300


def test_pomcp_rescues_adjacent_survivor_when_certain(cfg):
    c = cfg.with_(survivor=(0, 1))
    m = c.public_model()
    joint = np.zeros((16, 16))
    joint[0, 1] = 1.0
    d = POMCPPlanner(m, n_simulations=400, seed=0).select_action(Belief(m, joint))
    assert d.action == RIGHT


def test_reproducible_episodes(cfg):
    a = run_episode(cfg, PolicySpec.of("pomcp", n_simulations=100), 3)
    b = run_episode(cfg, PolicySpec.of("pomcp", n_simulations=100), 3)
    assert a["actions"] == b["actions"] and a["total_reward"] == b["total_reward"]


def test_common_random_numbers_share_survivors(cfg):
    df = evaluate_policies(cfg, [PolicySpec.of("random"), PolicySpec.of("heuristic")], 8)
    s = df.pivot(index="episode", columns="policy", values="survivor")
    assert (s.iloc[:, 0] == s.iloc[:, 1]).all()


def test_clairvoyant_is_optimal_reference(cfg):
    r = run_episode(cfg, PolicySpec.of("clairvoyant"), 0, survivor_mode="fixed")
    assert r["success"] and r["hazard_entries"] == 0 and r["scans"] == 0


def test_intervals():
    lo, hi = wilson_interval(50, 50)
    assert lo == pytest.approx(0.9287, abs=1e-3) and hi == 1.0
    for k in range(0, 101):
        lo, hi = wilson_interval(k, 100)
        assert lo <= k / 100 <= hi
    lo, hi = bootstrap_ci([1, 2, 3, 4, 5], seed=0)
    assert lo < 3 < hi


def test_belief_stays_valid_through_episode(cfg):
    r = MissionRunner(cfg, RiskAwareHeuristicPolicy(cfg.public_model()), env_seed=4)
    while not r.done:
        r.step()
        r.belief.check()
