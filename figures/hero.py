"""Render the JOPA hero animation: message passing, and what uncertainty buys.

Every number drawn is produced by the library — conjugate system
identification, forward/backward message passing, and exact action inference
on the same factor graph. Nothing is scripted.

    python examples/message_passing_animation.py --out docs/hero.gif
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jopa import (
    Agent,
    Block,
    Frozen,
    Gaussian,
    JointModel,
    LearnedLinear,
)

DT = 0.1
DAMPING = 0.88
POSITION_NOISE = 0.10
VELOCITY_NOISE = 0.30
TRUE_A = np.array([[1.0, DT], [0.0, DAMPING]])
TRUE_B = np.array([[0.0], [0.6]])


def rollout(rng, steps, controller=None, state=None):
    """Simulate the plant; `controller(x_noisy) -> u` closes the loop."""
    x = np.array([0.0, 0.0]) if state is None else np.array(state, float)
    states, observations, controls = [x.copy()], [], []
    for _ in range(steps):
        noise = rng.normal(0, [POSITION_NOISE, VELOCITY_NOISE])
        observations.append(x + noise)
        u = (rng.uniform(-1, 1, 1) if controller is None
             else np.asarray(controller(x + noise), float).reshape(1))
        controls.append(u)
        x = TRUE_A @ x + TRUE_B @ u
        states.append(x.copy())
    observations.append(x + rng.normal(0, [POSITION_NOISE, VELOCITY_NOISE]))
    return np.array(states), np.array(observations), np.array(controls)


def identify(rng, n_trajectories=32, steps=40):
    """Conjugate system identification from babble: q(A, B, W).

    The observations are noisy, so the sensor noise enters as message
    precision and the chain smooths the latent states while the parameter
    messages accumulate. Regressing the noisy observations directly would
    attenuate the transition toward zero.
    """
    precision = 1.0 / np.array([POSITION_NOISE, VELOCITY_NOISE]) ** 2
    message = observation_message(precision)
    sequences, controls = [], []
    for _ in range(n_trajectories):
        _, observed, control = rollout(rng, steps)
        sequences.append([message(y) for y in observed])
        controls.append([jnp.asarray(u) for u in control])
    transition = LearnedLinear(dim=2, du=1, offset=True, n_iterations=30,
                               prior_a_cov=0.5, prior_b_cov=100.0,
                               prior_W_df=4)
    transition.learn(sequences, controls)
    return transition


def observation_message(precision):
    matrix = jnp.diag(jnp.asarray(precision, dtype=jnp.float32))

    def message(value):
        mean = jnp.asarray(value, dtype=jnp.float32)
        return Gaussian(eta=matrix @ mean, lam=matrix)

    return message


@dataclass
class Estimation:
    """Smoothed past and predicted future from one message-passing sweep."""
    truth: np.ndarray
    observations: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_observed: int


def estimate(transition, rng, steps=44, observed=26):
    """Smooth the observed prefix and predict the unobserved tail.

    The same public call the digits examples use: `smooth` runs one
    forward-backward sweep, and `n_predict` extends the chain past the last
    observation, so prediction is the same operation as smoothing.
    """
    truth, observations, controls = rollout(rng, steps)
    precision = 1.0 / np.array([POSITION_NOISE, VELOCITY_NOISE]) ** 2
    model = JointModel([
        Block("z", transition, observe=Frozen(observation_message(precision)))
    ])
    predict = len(observations) - observed
    result = model.smooth(
        {"z": list(observations[:observed])}, n_predict=predict,
        controls=[u for u in controls[:observed - 1]],
        predict_controls=[u for u in controls[observed - 1:]])["z"]
    means = np.asarray(result["means"])
    std = np.sqrt(np.maximum(
        np.stack([np.diag(c) for c in np.asarray(result["covs"])]), 0.0))
    return Estimation(truth, observations, means, std, observed)


@dataclass
class ControlRun:
    """One closed-loop episode, with the belief and action posteriors kept."""
    label: str
    error: np.ndarray
    belief_std: np.ndarray
    action_std: np.ndarray
    control: np.ndarray


def control(transition, rng, trust_noise, label, steps=60, goal=1.0):
    """Close the loop; `trust_noise` is the noise the agent believes it has."""
    precision = 1.0 / np.asarray(trust_noise, float) ** 2
    model = JointModel([
        Block("z", transition, observe=Frozen(observation_message(precision)))
    ])
    agent = Agent(model, horizon=12, action_precision=0.05,
                  goal_precision=np.array([160.0, 12.0]),
                  goal_schedule="stage", u_clip=1.0,
                  relin_surprise=1e9, adapt_hold=False,
                  condition_on_estimate=True, plan_precision_scale=1e3,
                  track_uncertainty=True)
    agent.goal(np.array([goal, 0.0]))
    error, belief_std, action_std, applied = [], [], [], []

    def controller(observed):
        u = agent.step(observed)
        belief_std.append(float(np.sqrt(np.trace(
            np.asarray(agent.last_belief_cov)) / 2.0)))
        action_std.append(float(np.sqrt(
            np.asarray(agent.last_action_cov)[0, 0])))
        applied.append(float(u[0]))
        return u

    truth, _, _ = rollout(rng, steps, controller=controller)
    error = np.abs(truth[1:, 0] - goal)
    return ControlRun(label, error, np.array(belief_std),
                      np.array(action_std), np.array(applied))


def build(seed=0):
    """Produce every quantity the animation shows."""
    transition = identify(np.random.RandomState(seed))
    estimation = estimate(transition, np.random.RandomState(seed + 1))
    honest = control(transition, np.random.RandomState(seed + 2),
                     [POSITION_NOISE, VELOCITY_NOISE],
                     "calibrated sensor")
    overconfident = control(transition, np.random.RandomState(seed + 2),
                            [POSITION_NOISE / 12.0, VELOCITY_NOISE / 12.0],
                            "sensor claims certainty")
    return {
        "A": np.asarray(transition.A),
        "B": np.asarray(transition.B),
        "estimation": estimation,
        "runs": [honest, overconfident],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="assets/hero.gif")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--model-figure", default=None,
                       help="also write the static annotated model figure")
    args = parser.parse_args(argv)

    from style import render, render_model
    if args.model_figure:
        print(render_model(args.model_figure), flush=True)
    data = build(args.seed)
    render(data, args.out, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
