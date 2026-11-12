import numpy as np
import pytest

from rescuebot_core import (ACTIONS, DOWN, HEAT_OBS, OBS_COLD, OBS_HOT, OBS_NONE, OBS_WARM, RIGHT,
                            SCAN, Belief, ImpossibleObservationError, entropy_bits)


def test_initial_belief(model):
    b = Belief.initial(model)
    assert b.joint.sum() == pytest.approx(1.0)
    rob = b.robot_marginal()
    assert rob[0, 0] == pytest.approx(1.0)
    surv = b.survivor_marginal()
    assert surv[0, 0] == 0.0 and surv[1, 1] == 0.0 and surv[2, 2] == 0.0
    assert np.count_nonzero(surv) == 13
    assert b.survivor_entropy() == pytest.approx(np.log2(13))


def brute_force_update(model, joint, a, o):
    """Independent reference using the full transition matrix."""
    states, T = model.full_transition_matrix(a)
    prior = np.array([joint[r, q] for r, q in states])
    pred = prior @ T
    lik = np.array([model.obs_lik[o, r, q] * (r != q) for r, q in states])
    post = pred * lik
    post /= post.sum()
    out = np.zeros_like(joint)
    for (r, q), p in zip(states, post):
        out[r, q] = p
    return out


@pytest.mark.parametrize("seq", [[(RIGHT, OBS_NONE), (SCAN, OBS_WARM)],
                                 [(DOWN, OBS_NONE), (DOWN, OBS_NONE), (SCAN, OBS_HOT)],
                                 [(SCAN, OBS_COLD), (RIGHT, OBS_NONE), (SCAN, OBS_COLD)]])
def test_update_matches_brute_force(model, seq):
    b = Belief.initial(model)
    ref = b.joint.copy()
    for a, o in seq:
        b = b.update(a, o)
        ref = brute_force_update(model, ref, a, o)
        np.testing.assert_allclose(b.joint, ref, atol=1e-12)
        assert b.joint.sum() == pytest.approx(1.0)
        assert np.all(b.joint >= 0)
        assert np.all(b.joint[:, model.hazard_mask] == 0)
        assert np.all(np.diag(b.joint) == 0)  # mission continues -> not co-located


def test_survivor_marginal_unchanged_by_movement(model):
    b = Belief.initial(model).update(SCAN, OBS_WARM)
    before = b.survivor_marginal()
    after = b.predict(RIGHT).sum(axis=0).reshape(4, 4)
    np.testing.assert_allclose(before, after, atol=1e-12)


def test_hot_concentrates_near_robot(model):
    b = Belief.initial(model)
    hot = b.update(SCAN, OBS_HOT).survivor_marginal()
    cold = b.update(SCAN, OBS_COLD).survivor_marginal()
    near = [(0, 1), (1, 0)]
    assert sum(hot[c] for c in near) > sum(cold[c] for c in near)
    assert hot[3, 3] < cold[3, 3]


def test_impossible_observation_raises_not_uniform(model):
    perfect = model.obs_lik.copy()
    b = Belief.initial(model)
    # make Hot impossible everywhere
    model.obs_lik[OBS_HOT] = 0.0
    try:
        with pytest.raises(ImpossibleObservationError):
            b.update(SCAN, OBS_HOT)
    finally:
        model.obs_lik[:] = perfect


def test_wrong_observation_type_rejected(model):
    b = Belief.initial(model)
    with pytest.raises(ValueError):
        b.update(SCAN, OBS_NONE)
    with pytest.raises(ValueError):
        b.update(RIGHT, OBS_HOT)


def test_rescue_conditioning(model):
    b = Belief.initial(model)
    b2 = b.update(RIGHT, OBS_NONE, rescued=True)
    np.testing.assert_allclose(b2.robot_marginal(), b2.survivor_marginal())
    assert b2.survivor_marginal()[0, 1] == pytest.approx(1.0)


def test_entropy():
    assert entropy_bits([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy_bits([1.0, 0.0]) == 0.0


def test_predicted_obs_distribution_sums_to_one(model):
    b = Belief.initial(model)
    assert sum(b.predicted_obs_distribution().values()) == pytest.approx(1.0)
