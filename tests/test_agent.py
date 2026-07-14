"""Closed-loop Agent: sense → filter → plan → act with online relearning."""
import numpy as np
import jax.numpy as jnp
import pytest

from jopa import Agent, Block, JointModel, LearnedLinear, Frozen, Gaussian


def _msg(mean, prec=1e4):
    m = jnp.asarray(mean, dtype=jnp.float32)
    return Gaussian(eta=prec * m, lam=prec * jnp.eye(m.shape[0]))


def _warm_model(A, B, rng, n_traj=15, T=40):
    trajs = []
    for _ in range(n_traj):
        x = rng.randn(2) * 0.5
        seq, us = [x.copy()], []
        for _ in range(T - 1):
            u = rng.randn(1)
            us.append(u)
            x = A @ x + (B @ u).ravel() + 0.005 * rng.randn(2)
            seq.append(x.copy())
        trajs.append({"z": seq, "control": us})
    block = Block("z", LearnedLinear(dim=2, du=1, n_iterations=40),
                  observe=Frozen(_msg))
    model = JointModel([block])
    model.learn(trajs)
    return model


def test_agent_reaches_and_holds_double_integrator():
    rng = np.random.RandomState(0)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    model = _warm_model(A, B, rng)

    agent = Agent(model, horizon=8, forget=0.5, u_clip=None,
                  action_precision=0.05, goal_precision=200.0)
    agent.goal(np.array([1.0, 0.0]))
    x = np.array([0.0, 0.0])
    errs = []
    for _ in range(60):
        u = agent.step(x)
        x = A @ x + (B @ u).ravel()
        errs.append(abs(x[0] - 1.0))
    assert min(errs) < 0.05
    assert np.mean(errs[-10:]) < 0.1          # holds, not just crosses


def test_agent_goal_switch_clears_regime():
    rng = np.random.RandomState(1)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    model = _warm_model(A, B, rng)
    agent = Agent(model, horizon=8, u_clip=None)
    agent.goal(np.array([1.0, 0.0]))
    x = np.array([0.0, 0.0])
    for _ in range(40):
        u = agent.step(x)
        x = A @ x + (B @ u).ravel()
    agent.goal(np.array([-0.5, 0.0]))          # regime switch
    assert agent._means == [] and agent._acts == []
    errs = []
    for _ in range(60):
        u = agent.step(x)
        x = A @ x + (B @ u).ravel()
        errs.append(abs(x[0] + 0.5))
    assert np.mean(errs[-10:]) < 0.1


def test_agent_requires_goal_and_learned_control():
    rng = np.random.RandomState(2)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    model = _warm_model(A, B, rng)
    agent = Agent(model)
    with pytest.raises(ValueError):
        agent.step(np.zeros(2))
    fresh = JointModel([Block("z", LearnedLinear(dim=2, du=1),
                              observe=Frozen(_msg))])
    with pytest.raises(ValueError):
        Agent(fresh)
