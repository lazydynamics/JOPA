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


def test_goal_precision_scalar_vector_and_full_matrix_normalization():
    scalar = Agent(_known_posterior_model(), goal_precision=3.0)
    vector = Agent(
        _known_posterior_model(), goal_precision=jnp.array([3.0, 5.0]))
    matrix_value = jnp.array([[3.0, 0.5], [0.5, 5.0]])
    matrix = Agent(
        _known_posterior_model(), goal_precision=matrix_value)

    assert np.allclose(np.asarray(scalar._pg), 3.0 * np.eye(2))
    assert np.allclose(np.asarray(vector._pg), np.diag([3.0, 5.0]))
    assert np.allclose(np.asarray(matrix._pg), np.asarray(matrix_value))


@pytest.mark.parametrize("value", [
    np.ones(3),
    np.ones((2, 3)),
    np.array([[1.0, 1.0], [0.0, 1.0]]),
    np.array([[1.0, 0.0], [0.0, -0.1]]),
])
def test_goal_precision_rejects_incompatible_or_non_psd_values(value):
    with pytest.raises(ValueError, match="goal_precision"):
        Agent(_known_posterior_model(), goal_precision=value)


def test_diagonal_matrix_is_exact_planner_equivalent_to_vector():
    vector_agent = Agent(
        _known_posterior_model(), horizon=6, u_clip=None,
        action_precision=0.2, goal_precision=np.array([50.0, 100.0]),
        goal_schedule="stage", adapt_hold=False)
    matrix_agent = Agent(
        _known_posterior_model(), horizon=6, u_clip=None,
        action_precision=0.2,
        goal_precision=np.diag([50.0, 100.0]),
        goal_schedule="stage", adapt_hold=False)
    goal = np.array([1.0, 0.0])
    start = np.array([0.0, 0.0])
    vector_agent.goal(goal)
    matrix_agent.goal(goal)

    vector_action = vector_agent.step(start)
    matrix_action = matrix_agent.step(start)
    assert np.allclose(vector_action, matrix_action, atol=1e-7)

