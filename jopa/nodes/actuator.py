"""The actuator, learned by a `ContinuousTransition` node.

The plant executes ``A·u_j`` for every command block ``u_j`` (``u``'s last
axis is ``[a₁, a₂, a₁, a₂, …]``, actuators innermost). ``A = I`` is the
nominal plant; a reversed actuator is a −1 on the diagonal, swapped wires a
permutation, cross-talk off-diagonal mass.

Linearized at the current E[A], observed dynamics are linear in A:

    x' − f(s, v₀) + J v₀  =  Σ_pq A_pq · X_pq + ε,    v₀ = E[A]u,  J = ∂f/∂v,

so every latent coordinate of every observed transition is one CT
observation ``y ~ N(aᵀx, w⁻¹)`` with known regressor ``x``, ``a = vec A``
and noise precision ``w``. q(a) and q(w) are the CT node's :a and :W
messages (`ct_parameter_messages`) from the summed moments, multiplied into
the prior, relinearized each sweep. The posterior is the next prior, so
evidence accumulates across calls. Planning uses E[A]: goals must not send
messages into beliefs about the plant.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import WISHART_DF_MARGIN
from ..distributions import Gaussian, Wishart
from ..fg import Fn
from .dynamics import Dynamics
from .transition import CTCache, CTMeta, CTStats, ct_parameter_messages

SWEEPS = 5


def apply_matrix(u, A):
    """``A`` applied to every command block of ``u``."""
    return (u.reshape(*u.shape[:-1], -1, A.shape[-1]) @ A.T).reshape(u.shape)


@jax.tree_util.register_pytree_node_class
class Actuator:
    def __init__(self, q_a: Gaussian, q_w: Wishart):
        self.q_a, self.q_w = q_a, q_w

    @classmethod
    def nominal(cls, n_actuators, var=1.0, noise_var=1.0) -> Actuator:
        """``vec A ~ N(vec I, var·I)``; ``E[w] = 1 / noise_var``, the
        dynamics' noise level (a weak prior: a few observations move it)."""
        df = 1 + WISHART_DF_MARGIN
        return cls(Gaussian.from_mean(jnp.eye(n_actuators).ravel(), 1.0 / var),
                   Wishart(df, df * noise_var * jnp.eye(1)))

    def tree_flatten(self):
        return (self.q_a, self.q_w), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def n(self):
        return round(self.q_a.eta.shape[0] ** 0.5)

    @property
    def mean(self):
        """E[A]."""
        return self.q_a.mean.reshape(self.n, self.n)

    @property
    def sd(self):
        """Posterior standard deviation of each entry of A."""
        return jnp.sqrt(jnp.diag(self.q_a.cov)).reshape(self.n, self.n)

    def __call__(self, u):
        """The executed actuation of a command expression, at E[A]."""
        return Fn(apply_matrix, u, self.mean)

    def moments(self, model: Dynamics, history, u, x_next, weight=None) -> CTStats:
        """Summed CT moments of the observed transitions (leading axis of
        ``history``, ``u``, ``x_next``), linearized at E[A]; ``weight`` 0
        drops a transition."""
        n, A0 = self.n, self.mean
        N, du = u.shape
        v0 = apply_matrix(u, A0)
        f0 = model.mean(tuple(history), v0)
        J = jax.vmap(jax.jacfwd(lambda v, *h: model.mean(h, v)))(v0, *history)
        X = jnp.einsum("ndbp,nbq->ndpq", J.reshape(N, -1, du // n, n),
                       u.reshape(N, du // n, n)).reshape(N, -1, n * n)
        Y = x_next - f0 + X @ A0.ravel()
        w = jnp.ones(N) if weight is None else jnp.asarray(weight, Y.dtype)
        return CTStats(n=w.sum() * Y.shape[1],
                       xx=jnp.einsum("n,ndi,ndj->ij", w, X, X),
                       yy=jnp.einsum("n,nd,nd->", w, Y, Y)[None, None],
                       yx=jnp.einsum("n,nd,ndj->j", w, Y, X)[None])

    def update(self, model: Dynamics, history, u, x_next, weight=None,
               sweeps=SWEEPS) -> Actuator:
        """q(a) q(w) after the observed transitions, with this belief as the prior."""
        post = self
        for _ in range(sweeps):
            s = post.moments(model, history, u, x_next, weight)
            cache = CTCache(post.q_a, post.q_w, CTMeta(lambda a: a.reshape(1, -1)))
            msg_a, delta, _ = ct_parameter_messages(s, cache)
            post = Actuator(self.q_a * msg_a, Wishart(self.q_w.df + s.n, self.q_w.inv_scale + delta))
        return post
