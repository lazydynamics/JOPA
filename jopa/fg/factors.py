"""Gaussian factor nodes and their linearization.

One node type covers priors, observations, goals and noisy transitions:

    Normal(x, mean, precision):   p(x | mean) ∝ exp −½ (x − mean)ᵀ P (x − mean)

where ``x`` and ``mean`` are expressions (`jopa.fg.expr`): a variable, a
deterministic node of variables, or a constant. ``Normal(x_next,
transition.step(x, u), 1/σ²)`` is a transition node, ``Normal(x, goal, γ)`` a
goal, ``Normal(u, 0, λ)`` an action prior.

Messages out of a node whose residual is nonlinear in its variables use the
linearization approximation: at the current estimate z₀ the node is replaced
by the Gaussian potential of its first-order expansion r(z) ≈ r₀ + J (z − z₀),

    Λ = Jᵀ P J,    η = Jᵀ P (J z₀ − r₀),

which is exact when the residual is affine.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..distributions import Gaussian
from .expr import evaluate, variables


class Potential(NamedTuple):
    """Gaussian potential ``exp(ηᵀz − ½ zᵀΛz)`` over the concatenation of ``vars``."""
    vars: tuple[str, ...]
    eta: jnp.ndarray
    lam: jnp.ndarray


def _weigh(precision, J):
    """``P J`` for a scalar, diagonal (vector) or full precision."""
    p = jnp.asarray(precision)
    if p.ndim == 2:
        return p @ J
    return p.reshape(-1, *([1] * (J.ndim - 1))) * J if p.ndim == 1 else p * J


class Normal:
    """Gaussian node ``x ~ N(mean, P⁻¹)``; see the module docstring.

    ``precision`` is a scalar, a vector matching the residual's flattened
    size (diagonal), or a full matrix. Leading batch axes in ``x`` and
    ``mean`` act as a plate: one node holds many conditionally independent
    terms, e.g. every observed transition of a run.
    """

    def __init__(self, x, mean, precision):
        self.x = x
        self.mean = mean
        self.precision = precision
        self.vars = variables(x, mean)

    @classmethod
    def from_gaussian(cls, x, belief: Gaussian):
        """Prior node carrying an information-form belief, e.g. a posterior
        from an earlier inference that becomes the next prior."""
        return cls(x, belief.mean, belief.lam)

    def residual(self, values: dict, cache: dict | None = None):
        cache = {} if cache is None else cache
        r = evaluate(self.x, values, cache) - evaluate(self.mean, values, cache)
        return jnp.reshape(r, -1)

    def energy(self, values: dict):
        """Negative log of the node up to a constant: ½ rᵀ P r."""
        r = self.residual(values)
        return 0.5 * jnp.sum(r * _weigh(self.precision, r))

    def linearize(self, values: dict) -> Potential:
        names = tuple(v.name for v in self.vars)
        if not names:
            raise ValueError("Normal node has no variables; it is a constant")
        z0 = tuple(values[n] for n in names)

        def r(*z):
            return self.residual(dict(values, **dict(zip(names, z, strict=True))))

        r0 = r(*z0)
        J = jnp.concatenate(
            [j.reshape(r0.shape[0], -1)
             for j in jax.jacfwd(r, argnums=tuple(range(len(names))))(*z0)], axis=1)
        PJ = _weigh(self.precision, J)
        z0 = jnp.concatenate([jnp.reshape(z, -1) for z in z0])
        return Potential(names, PJ.T @ (J @ z0 - r0), J.T @ PJ)


class Box:
    """Indicator node ``1[lo ≤ x ≤ hi]`` on a variable — a uniform prior on a box.

    It sends no Gaussian message; inference keeps every estimate inside the
    box by projection, so the result is the mode of the constrained
    posterior (projected Gauss–Newton). Covariances at an active bound are
    those of the unconstrained linearization.
    """

    def __init__(self, x, lo, hi):
        self.x = x
        self.lo, self.hi = lo, hi
        self.vars = variables(x)
        if len(self.vars) != 1 or self.vars[0] is not x:
            raise ValueError("Box constrains a single variable directly")

    def project(self, values: dict) -> dict:
        name = self.x.name
        return dict(values, **{name: jnp.clip(values[name], self.lo, self.hi)})
