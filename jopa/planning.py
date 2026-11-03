"""Planning as inference.

The reaching graph over commands ``u[0..H−1]``:

    u[k] ~ N(0, λ⁻¹ I),   1[|u[k]| ≤ limit]                  action prior, limits
    x[k+1] = mean(x[k−order+1..k], actuator(u[k]))           deterministic rollout
    goal ~ N(x[k+1], γ⁻¹ I)                                   goal at every step

The plan is the posterior mode, found by `jopa.fg.Graph.infer_restarts`.
``actuator`` maps a command expression to what the plant receives — the
identity, or a `jopa.Actuator` belief. Precisions are scalars or
per-dimension vectors; ``limit=None`` drops the box, and the graph of a linear
model is then affine, so one iteration is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .fg import Box, Graph, Normal, Variable
from .nodes.dynamics import Dynamics


@dataclass(frozen=True)
class Planner:
    horizon: int = 3
    starts: int = 8                 # zero commands plus starts − 1 uniform draws
    iterations: int = 20
    action_precision: float = 1e-2
    goal_precision: float = 1.0
    limit: float | None = 1.0

    def graph(self, model: Dynamics, history, goal, du, actuator=None) -> Graph:
        """``history``: the ``model.order`` latest states, oldest first."""
        nodes, states = [], tuple(history)
        for k in range(self.horizon):
            u = Variable(f"u{k}", du)
            nodes.append(Normal(u, 0.0, self.action_precision))
            if self.limit is not None:
                nodes.append(Box(u, -self.limit, self.limit))
            states = (*states[1:], model.step(states, u if actuator is None else actuator(u)))
            nodes.append(Normal(states[-1], goal, self.goal_precision))
        return Graph(nodes)

    def __call__(self, model: Dynamics, history, goal, key, du, actuator=None):
        """Commands (horizon, du) within ±limit and the graph energy at the plan.
        One problem; ``jax.vmap`` it over a batch."""
        graph = self.graph(model, history, goal, du, actuator)
        u0 = jax.random.uniform(key, (self.starts, self.horizon, du), minval=-1.0, maxval=1.0)
        u0 = u0.at[0].set(0.0)
        post = graph.infer_restarts({f"u{k}": u0[:, k] for k in range(self.horizon)},
                                    self.iterations)
        return jnp.stack([post.mean[f"u{k}"] for k in range(self.horizon)]), post.energy
