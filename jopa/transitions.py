"""Transitions: how a state slice evolves.

`KnownPhysics` is a gray-box linearization. `LearnedLinear` is the conjugate
posterior q(A, B, W) of a `ContinuousTransition` node, learned by VMP on the
latent chain.

The chain is a `jopa.fg` graph: a prior on x[0], one observation message per
slice, and per step the CT node's averaged factor
exp E_q log N(x[t+1]; A x[t] + B u[t], W⁻¹), which is the joint rule
`ct_marginal_yx` with no incoming messages. Eliminating the chain in time order
is the forward–backward sweep; its pairwise marginals q(x[t+1], x[t]) give the
CT moments (`ct_moments`), and `ct_parameter_messages` turns them into
q(A, B, W).
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from .config import (
    INIT_A_COV,
    INIT_B_COV,
    PRIOR_A_COV,
    PRIOR_B_COV,
    VMP_ITERATIONS,
    WISHART_DF_MARGIN,
)
from .distributions import (
    Gaussian,
    Wishart,
    gaussian_mean,
    gaussian_mean_cov,
    gaussian_prior,
    vague_gaussian,
)
from .fg import eliminate_chain
from .nodes.dynamics import Linear
from .nodes.transition import (
    CTCache,
    CTMeta,
    ct_marginal_yx,
    ct_moments,
    ct_parameter_messages,
    ct_predict,
)


def _chain(obs: Gaussian, cache: CTCache, us):
    """Smoothed means, covariances and Cov(x[t], x[t+1]) of one chain.

    ``obs`` stacks the (T, d) observation messages, ``us`` the (T − 1, du)
    inputs or None; the prior on x[0] is N(0, I)."""
    T, d = obs.eta.shape
    vague = vague_gaussian(d)
    if us is None:
        factor = jax.tree.map(lambda v: jnp.broadcast_to(v, (T - 1, *v.shape)),
                              ct_marginal_yx(vague, vague, cache))
    else:
        factor = jax.vmap(lambda u: ct_marginal_yx(vague, vague, cache, u))(us)
    lam = obs.lam.at[0].add(jnp.eye(d))
    return eliminate_chain(obs.eta, lam, factor.eta, factor.lam)


_smooth = jax.jit(lambda obs, cache, us: _chain(obs, cache, us)[:2])


def _moments(obs: Gaussian, cache: CTCache, us):
    """CT moments of one chain, summed over its pairwise marginals q(y, x),
    y = x[t+1], x = x[t]."""
    m, V, C = _chain(obs, cache, us)
    mean = jnp.concatenate([m[1:], m[:-1]], axis=-1)
    cov = jnp.concatenate([jnp.concatenate([V[1:], jnp.swapaxes(C, 1, 2)], axis=2),
                           jnp.concatenate([C, V[:-1]], axis=2)], axis=1)
    lam = jnp.linalg.inv(cov)
    q = Gaussian(jnp.einsum("tij,tj->ti", lam, mean), lam)
    stats = (jax.vmap(lambda g: ct_moments(g, cache))(q) if us is None
             else jax.vmap(lambda g, u: ct_moments(g, cache, u))(q, us))
    return jax.tree.map(lambda v: v.sum(0), stats)


@partial(jax.jit, static_argnames=("n_iterations",))
def _fit(groups, priors, q, n_iterations):
    """VMP: alternate the chain sweep with the CT parameter messages.

    ``groups`` holds equal-length trajectories stacked on a leading axis,
    as ``(obs, us)`` pairs; only the summed moments are pooled across them."""
    prior_a, prior_W, prior_b = priors
    d = prior_W.inv_scale.shape[0]
    meta = CTMeta(lambda a: a.reshape(d, d))

    def sweep(_, q):
        cache = CTCache(q[0], q[1], meta, q[2])
        stats = [jax.vmap(lambda o, u: _moments(o, cache, u))(obs, us) for obs, us in groups]
        s = jax.tree.map(lambda *v: sum(x.sum(0) for x in v), *stats)
        msg_a, delta, msg_b = ct_parameter_messages(s, cache)
        return (prior_a * msg_a, Wishart(prior_W.df + s.n, prior_W.inv_scale + delta),
                None if prior_b is None else prior_b * msg_b)

    return jax.lax.fori_loop(0, n_iterations, sweep, q)


class KnownPhysics:
    """Gray-box dynamics: `linearize(belief_mean) -> (A, B, offset)` defines a
    local linear-Gaussian map  s' = A·s + B·u + offset + ε  re-evaluated per step.
    """
    learned = False

    def __init__(self, dim, linearize: Callable, du=0,
                 process_std=1e-2):
        self.dim = dim
        self.du = du
        self.linearize = linearize
        self.Q = process_std ** 2 * jnp.eye(dim)

    def predict(self, belief: Gaussian, u=None) -> Gaussian:
        A, B, offset = self.linearize(gaussian_mean(belief))
        mu, cov = gaussian_mean_cov(belief)
        mean = A @ mu + offset
        if u is not None and B is not None:
            mean = mean + B @ jnp.asarray(u)
        lam = jnp.linalg.inv(A @ cov @ A.T + self.Q)
        return Gaussian(eta=lam @ mean, lam=lam)


class LearnedLinear:
    """Linear-Gaussian dynamics  x' ~ N(A·x + B·u [+ c], W⁻¹)  learned by the CT node.

    `learn` returns the joint posterior q(A,B,W | all trajectories).

    ``offset=True`` adds a learned constant drift `c` (wind, current, bias):
    a pinned-1 control channel is appended internally, so `c` is B's last
    column and gets the same conjugate treatment — posterior, credible
    intervals, planning — with no extra message rules.
    """
    learned = True

    def __init__(self, dim, du=0, n_iterations=VMP_ITERATIONS, offset=False,
                 prior_a_cov=PRIOR_A_COV, prior_a_mean=None, init_a_cov=INIT_A_COV,
                 prior_b_cov=PRIOR_B_COV, init_b_cov=INIT_B_COV, prior_W_df=None):
        self.dim = dim
        self.du = du
        self.n_iterations = n_iterations
        self.offset = offset
        self.prior_a_cov, self.prior_a_mean, self.init_a_cov = prior_a_cov, prior_a_mean, init_a_cov
        self.prior_b_cov, self.init_b_cov = prior_b_cov, init_b_cov
        self.prior_W_df = (dim + WISHART_DF_MARGIN if prior_W_df is None
                           else prior_W_df)
        self.q_a = self.q_W = self.q_b = None
        self._carried_a = self._carried_W = self._carried_b = None

    @property
    def _meta(self):
        d = self.dim
        return CTMeta(lambda a: a.reshape(d, d))

    @property
    def eff_du(self):
        """Control width seen by the CT node (+1 pinned channel when offset)."""
        return self.du + 1 if self.offset else self.du

    def _augment_u(self, u=None):
        """Control with the pinned-1 offset channel appended (offset=True)."""
        u = jnp.zeros(self.du) if u is None else jnp.asarray(u).reshape(self.du)
        return jnp.concatenate([u, jnp.ones(1)])

    def _base_priors(self):
        d, du = self.dim, self.eff_du
        prior_a = gaussian_prior(d * d, self.prior_a_cov, self.prior_a_mean)
        prior_W = Wishart(df=self.prior_W_df, inv_scale=jnp.eye(d))
        prior_b = gaussian_prior(d * du, self.prior_b_cov) if du > 0 else None
        return prior_a, prior_W, prior_b

    def _priors(self):
        if self._carried_a is not None:
            return self._carried_a, self._carried_W, self._carried_b
        return self._base_priors()

    def _init_q(self, has_ctrl):
        d, du = self.dim, self.eff_du
        q_a = self.q_a if self.q_a is not None else gaussian_prior(d * d, self.init_a_cov, self.prior_a_mean)
        q_W = self.q_W if self.q_W is not None else Wishart(df=self.prior_W_df, inv_scale=jnp.eye(d))
        q_b = None
        if has_ctrl:
            q_b = self.q_b if self.q_b is not None else gaussian_prior(d * du, self.init_b_cov)
        return q_a, q_W, q_b

    def _controls(self, controls, T):
        """The CT node's inputs for a T-slice chain: (T − 1, eff_du), with the
        pinned offset channel appended; None when the chain has no inputs."""
        if self.offset:
            return jnp.stack([self._augment_u(u) for u in
                              ([None] * (T - 1) if controls is None else controls)])
        if controls is None or self.du == 0:
            return None
        return jnp.stack([jnp.asarray(u, jnp.float32).reshape(self.du) for u in controls])

    def _cache(self):
        return CTCache(self.q_a, self.q_W, self._meta, self.q_b)

    def remember(self, forget=1.0, forget_a=None, forget_W=None, diffuse=0.0):
        """Fold the posterior into the prior for subsequent `learn` calls.

        Conjugate continual learning: q(A, B, W) becomes the next fit's prior,
        discounted toward the base prior by ``forget`` ∈ (0, 1] — exponential
        forgetting in natural parameters. ``forget=1`` accumulates evidence
        without bound (a static world); smaller values bound the effective
        memory so the posterior can track drifting dynamics. The tempered
        posterior also replaces the current variational state, so predictions
        and plans between chunks use it, and the next `learn` starts from the
        carried prior rather than warm-starting in the previous optimum.

        ``forget_a`` / ``forget_W`` override ``forget`` for q(A) / q(W) —
        e.g. ``remember(forget=0.1, forget_a=1.0)`` discards stale evidence
        about inputs and disturbances (B, c) while trusting the learned
        structure (A). Without it, a drift change can be mis-attributed along
        the degenerate A-vs-c direction.

        ``diffuse`` (requires ``offset=True``) is a Gaussian random-walk
        prediction step on the drift `c`: its posterior variance grows by
        ``diffuse`` per call, so a changing disturbance stays trackable —
        non-stationarity expressed as a prior, not a heuristic.
        """
        if self.q_a is None:
            raise ValueError("remember: nothing to remember — call learn() first.")
        forget_a = forget if forget_a is None else forget_a
        forget_W = forget if forget_W is None else forget_W
        for name, f in (("forget", forget), ("forget_a", forget_a), ("forget_W", forget_W)):
            if not 0.0 < f <= 1.0:
                raise ValueError(f"remember: {name} must be in (0, 1], got {f}")
        if diffuse and not self.offset:
            raise ValueError("remember: diffuse requires offset=True (there is no drift c).")
        base_a, base_W, base_b = self._base_priors()

        def blend(q, base, f):
            return Gaussian(eta=base.eta + f * (q.eta - base.eta),
                            lam=base.lam + f * (q.lam - base.lam))

        self._carried_a = blend(self.q_a, base_a, forget_a)
        self._carried_W = Wishart(
            df=base_W.df + forget_W * (self.q_W.df - base_W.df),
            inv_scale=base_W.inv_scale + forget_W * (self.q_W.inv_scale - base_W.inv_scale))
        self._carried_b = (blend(self.q_b, base_b, forget)
                           if self.q_b is not None and base_b is not None else None)
        if diffuse and self._carried_b is not None:
            mb, Vb = gaussian_mean_cov(self._carried_b)
            idx = jnp.arange(self.dim) * self.eff_du + (self.eff_du - 1)
            Vb = Vb.at[idx, idx].add(diffuse)
            lam = jnp.linalg.inv(Vb)
            self._carried_b = Gaussian(eta=lam @ mb, lam=lam)
        self.q_a, self.q_W, self.q_b = self._carried_a, self._carried_W, self._carried_b

    def learn(self, msg_seqs, ctrl_seqs=None):
        """One E-step. Returns concatenated smoothed `(means, covs)` (M-step input)."""
        obs = [Gaussian(jnp.stack([g.eta for g in seq]), jnp.stack([g.lam for g in seq]))
               for seq in msg_seqs]
        ctrl = [None] * len(msg_seqs) if ctrl_seqs is None else ctrl_seqs
        us = [self._controls(c, len(seq)) for c, seq in zip(ctrl, msg_seqs, strict=True)]
        has_ctrl = us[0] is not None
        by_length = {}
        for i, seq in enumerate(msg_seqs):
            by_length.setdefault(len(seq), []).append(i)
        groups = tuple(
            (Gaussian(jnp.stack([obs[i].eta for i in idx]), jnp.stack([obs[i].lam for i in idx])),
             jnp.stack([us[i] for i in idx]) if has_ctrl else None)
            for idx in by_length.values())
        prior_a, prior_W, prior_b = self._priors()
        def f32(tree):
            return jax.tree.map(lambda v: jnp.asarray(v, jnp.float32), tree)
        priors = f32((prior_a, prior_W, prior_b if has_ctrl else None))
        self.q_a, self.q_W, self.q_b = _fit(groups, priors, f32(self._init_q(has_ctrl)),
                                            self.n_iterations)
        cache = self._cache()
        means, covs = zip(*(_smooth(o, cache, u) for o, u in zip(obs, us, strict=True)), strict=True)
        return jnp.concatenate(means), jnp.concatenate(covs)

    def smooth(self, msgs, controls=None):
        """Smoothed ``(means, covs)`` of one observed chain under q(A, B, W)."""
        obs = Gaussian(jnp.stack([g.eta for g in msgs]), jnp.stack([g.lam for g in msgs]))
        return _smooth(obs, self._cache(), self._controls(controls, len(msgs)))

    def predict(self, belief: Gaussian, u=None) -> Gaussian:
        """Posterior predictive of the next state (`ct_predict`)."""
        cache = self._cache()
        if self.offset:
            u = self._augment_u(u)
        elif u is not None:
            u = jnp.asarray(u)
        return ct_predict(belief, cache, u)

    @property
    def dynamics(self) -> Linear:
        """The transition at its posterior mean, for planning."""
        B = jnp.zeros((self.dim, 0)) if self.B is None else self.B
        c = jnp.zeros(self.dim) if self.c is None else self.c
        return Linear(self.A, B, c)

    @property
    def A(self):
        return gaussian_mean(self.q_a).reshape(self.dim, self.dim)

    @property
    def A_std(self):
        """Marginal posterior standard deviation of each transition weight."""
        _, cov = gaussian_mean_cov(self.q_a)
        return jnp.sqrt(jnp.diag(cov)).reshape(self.dim, self.dim)

    @property
    def B(self):
        if self.q_b is None or self.du == 0:
            return None
        return gaussian_mean(self.q_b).reshape(self.dim, self.eff_du)[:, :self.du]

    @property
    def B_std(self):
        """Marginal posterior standard deviation of each physical control weight."""
        if self.q_b is None or self.du == 0:
            return None
        _, cov = gaussian_mean_cov(self.q_b)
        return jnp.sqrt(jnp.diag(cov)).reshape(
            self.dim, self.eff_du)[:, :self.du]

    @property
    def c(self):
        """Learned constant drift — B's pinned-channel column (offset=True)."""
        if not self.offset or self.q_b is None:
            return None
        return gaussian_mean(self.q_b).reshape(self.dim, self.eff_du)[:, -1]

    @property
    def c_std(self):
        """Marginal posterior standard deviation of the learned drift.

        Shrinks as `learn`/`remember` accumulate evidence about a constant
        disturbance, and grows again under `remember(diffuse=...)`. Watching it
        is how you see whether the drift is being identified or merely tracked.
        """
        if not self.offset or self.q_b is None:
            return None
        _, cov = gaussian_mean_cov(self.q_b)
        return jnp.sqrt(jnp.diag(cov)).reshape(self.dim, self.eff_du)[:, -1]
