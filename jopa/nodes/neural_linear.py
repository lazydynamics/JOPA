"""Neural-linear transition node.

    x[t+1] = x[t] + Wᵀ φ(x[t−order+1], …, x[t], u[t]) + ε,    ε ~ N(0, σ² I)

φ is a learned feature map, fixed during inference (`jopa.learn` trains
it). The last layer W ~ MN(M, Λ⁻¹, σ²I) has exact conjugate messages: an
observed transition sends ``(φyᵀ, φφᵀ)`` with ``y = x[t+1] − x[t]``, and W's
posterior is its prior times their product (`condition`). Toward the states
and the input the node is nonlinear and is linearized by `jopa.fg`
inference like any other node.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..distributions import MatrixNormal
from .dynamics import Dynamics


@jax.tree_util.register_pytree_node_class
class NeuralLinear(Dynamics):
    """
    Parameters
    ----------
    features : ``features(params, inputs) -> (..., m)``; ``inputs`` is the
        concatenation of the ``order`` most recent states and the input.
    params : parameters of ``features``.
    weights : `MatrixNormal` belief over W (m × d), or ``None`` with an
        explicit ``weight_mean`` (a point head, as during feature training).
    noise_var : σ², the per-dimension transition noise.
    order : past states the features see (2 puts velocity in the state).
    """

    def __init__(self, features, params, weights: MatrixNormal | None, noise_var, order=2,
                 weight_mean=None):
        self.features = features
        self.params = params
        self.weights = weights
        self.noise_var = noise_var
        self.order = order
        self.weight_mean = weights.mean if weight_mean is None else weight_mean

    def tree_flatten(self):
        return ((self.params, self.weights, self.noise_var, self.weight_mean),
                (self.features, self.order))

    @classmethod
    def tree_unflatten(cls, aux, children):
        params, weights, noise_var, weight_mean = children
        return cls(aux[0], params, weights, noise_var, aux[1], weight_mean)

    def phi(self, states, u):
        return self.features(self.params, jnp.concatenate([*states, u], axis=-1))

    def mean(self, states, u):
        return states[-1] + self.phi(states, u) @ self.weight_mean

    def weight_message(self, states, u, x_next) -> MatrixNormal:
        """Exact message to W from observed transitions (leading axis = plate)."""
        m, d = self.weight_mean.shape
        F = self.phi(states, u).reshape(-1, m)
        Y = (x_next - states[-1]).reshape(-1, d)
        return MatrixNormal(F.T @ Y, F.T @ F)

    def condition(self, states, u, x_next) -> NeuralLinear:
        """This node with W's posterior after the observed transitions."""
        return NeuralLinear(self.features, self.params,
                            self.weights * self.weight_message(states, u, x_next),
                            self.noise_var, self.order)
