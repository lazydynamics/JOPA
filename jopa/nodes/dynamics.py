"""The dynamics interface:  x[t+1] = mean(x[t−order+1], …, x[t], u[t]).

A subclass supplies ``mean`` (on arrays) and ``order``; `step` and
`rollout` follow, so any dynamics — learned, physical or hybrid — plugs into
planning (`jopa.Planner`) and actuator adaptation (`jopa.Actuator`).

Subclasses must be pytrees (``jax.tree_util.register_pytree_node_class``),
so a trained model is an argument of a compiled planner, not a constant
baked into it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..fg.expr import Fn


class Dynamics:
    order: int = 1

    def mean(self, states, u):
        """E[x[t+1]] given ``states`` (oldest first, ``order`` of them) and ``u``."""
        raise NotImplementedError

    def step(self, states, u) -> Fn:
        """The mean as a deterministic node, what a planner rolls forward."""
        if len(states) != self.order:
            raise ValueError(f"expected {self.order} states, got {len(states)}")
        return Fn(lambda *a: self.mean(a[:-1], a[-1]), *states, u)

    def rollout(self, states, actions):
        """Open-loop means: ``states`` the first ``order`` frames (each (B, d)),
        ``actions`` (B, H, du) → (B, H, d)."""
        states, out = tuple(states), []
        for k in range(actions.shape[1]):
            out.append(self.mean(states, actions[:, k]))
            states = (*states[1:], out[-1])
        return jnp.stack(out, axis=1)


@jax.tree_util.register_pytree_node_class
class Linear(Dynamics):
    """``x[t+1] = A x[t] + B u[t] + c``: a learned transition at its posterior
    mean, or a gray-box model linearized at the current belief."""

    def __init__(self, A, B, c):
        self.A, self.B, self.c = A, B, c

    def tree_flatten(self):
        return (self.A, self.B, self.c), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    def mean(self, states, u):
        return states[-1] @ self.A.T + u @ self.B.T + self.c
