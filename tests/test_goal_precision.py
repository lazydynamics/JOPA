import jax.numpy as jnp
import numpy as np
import pytest

from jopa import Agent, Block, Frozen, Gaussian, JointModel, LearnedLinear, Wishart
from jopa.distributions import gaussian_prior


def _message(value, precision=1e4):
    value = jnp.asarray(value, dtype=jnp.float32)
    matrix = precision * jnp.eye(value.shape[0])
    return Gaussian(eta=matrix @ value, lam=matrix)


def _known_posterior_model():
    transition = LearnedLinear(dim=2, du=1)
    matrix = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    control = jnp.array([[0.0], [0.1]])
    transition.q_a = gaussian_prior(4, 1e-8, matrix.ravel())
    transition.q_b = gaussian_prior(2, 1e-8, control.ravel())
    transition.q_W = Wishart(
        df=1e4, inv_scale=1e4 * 1e-4 * jnp.eye(2))
    return JointModel([
        Block("z", transition, observe=Frozen(_message))])


def test_goal_precision_scalar_and_vector():
    scalar = Agent(_known_posterior_model(), goal_precision=3.0)
    vector = Agent(_known_posterior_model(), goal_precision=jnp.array([3.0, 5.0]))
    assert np.allclose(np.asarray(scalar._planner.goal_precision), [3.0, 3.0])
    assert np.allclose(np.asarray(vector._planner.goal_precision), [3.0, 5.0])


@pytest.mark.parametrize("value", [np.ones(3), np.ones((2, 2)), np.array([1.0, -0.1])])
def test_goal_precision_rejects_wrong_shape_or_negative(value):
    with pytest.raises(ValueError, match="precision"):
        Agent(_known_posterior_model(), goal_precision=value)


def test_goal_precision_weights_the_plan_per_dimension():
    """A goal on position only must not pull velocity: the planned action
    differs from one that weights both dimensions."""
    start, goal = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    actions = []
    for precision in (np.array([50.0, 0.0]), np.array([50.0, 100.0])):
        agent = Agent(_known_posterior_model(), horizon=6, u_clip=None,
                      action_precision=0.2, goal_precision=precision)
        agent.goal(goal)
        actions.append(agent.step(start))
    assert not np.allclose(actions[0], actions[1], atol=1e-3)
