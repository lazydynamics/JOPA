"""Factor graph and inference: Gaussian sum-product with iterated linearization.

`Graph.infer` alternates two steps until the estimate stops improving:

1. **Linearize** every node at the current estimate (`Normal.linearize`),
   which makes the whole graph Gaussian.
2. **Sum-product** on that Gaussian graph by variable elimination along an
   ordering: eliminating a variable sends the message ∫ ∏(touching
   potentials) d(variable) to its separator — the junction-tree forward
   pass — and back-substitution is the backward pass. The result is exact
   for the linearized graph: the posterior mean and, at the end, marginal
   covariances.

`Box` nodes constrain variables to a box: estimates are projected onto it
after every step. The new mean is accepted through a line search on the graph energy
Σ ½ rᵀPr (the negative log joint), so every accepted step lowers it;
`Graph.infer_restarts` runs several initial estimates in parallel and keeps
the lowest. For an affine graph one iteration is exact.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..distributions import Gaussian
from .factors import Box, Normal, Potential

DEFAULT_STEPS = (1.0, 0.5, 0.25)


class Marginals(NamedTuple):
    """Posterior per variable: ``mean[name]``, ``cov[name]``; ``energy`` at the mean."""
    mean: dict
    cov: dict
    energy: jnp.ndarray

    def gaussian(self, name) -> Gaussian:
        """The marginal of ``name`` as a message, e.g. the next prior."""
        lam = jnp.linalg.inv(self.cov[name])
        return Gaussian(lam @ self.mean[name], lam)


def _scatter(potentials, scope, dims):
    """Sum ``potentials`` into one potential over ``scope``."""
    offsets, n = {}, 0
    for v in scope:
        offsets[v] = n
        n += dims[v]
    eta = jnp.zeros(n)
    lam = jnp.zeros((n, n))
    for p in potentials:
        idx = jnp.concatenate([jnp.arange(dims[v]) + offsets[v] for v in p.vars])
        eta = eta.at[idx].add(p.eta)
        lam = lam.at[idx[:, None], idx[None, :]].add(p.lam)
    return eta, lam


def eliminate(potentials, order, dims, covariance=True, pairs=()):
    """Exact Gaussian sum-product over ``potentials`` by eliminating ``order``.

    Returns ``(mean, cov)`` dicts keyed by variable name; ``cov`` is empty
    when ``covariance=False``. ``pairs`` adds cross-covariances ``cov[v, w]``
    for pairs where ``w`` is in ``v``'s separator, e.g. ``(x[t], x[t+1])``
    on a chain eliminated in time order.
    """
    pool = list(potentials)
    rank = {v: i for i, v in enumerate(order)}
    eliminated = []                     # (v, separator, Λvv, ΛvS, ηv)
    for v in order:
        touching = [p for p in pool if v in p.vars]
        pool = [p for p in pool if v not in p.vars]
        sep = tuple(sorted({w for p in touching for w in p.vars} - {v}, key=rank.__getitem__))
        eta, lam = _scatter(touching, (v, *sep), dims)
        n = dims[v]
        Lvv, LvS, ev = lam[:n, :n], lam[:n, n:], eta[:n]
        if sep:
            G = jnp.linalg.solve(Lvv, jnp.concatenate([LvS, ev[:, None]], axis=1))
            pool.append(Potential(sep, eta[n:] - LvS.T @ G[:, -1], lam[n:, n:] - LvS.T @ G[:, :-1]))
        eliminated.append((v, sep, Lvv, LvS, ev))

    mean, cov = {}, {}
    for v, sep, Lvv, LvS, ev in reversed(eliminated):
        if not sep:
            mean[v] = jnp.linalg.solve(Lvv, ev)
            if covariance:
                cov[v, v] = jnp.linalg.inv(Lvv)
            continue
        xS = jnp.concatenate([mean[w] for w in sep])
        mean[v] = jnp.linalg.solve(Lvv, ev - LvS @ xS)
        if covariance:
            # Σ_vS = −Λvv⁻¹ ΛvS Σ_SS,  Σ_vv = Λvv⁻¹ + Λvv⁻¹ ΛvS Σ_SS ΛSv Λvv⁻¹
            S = jnp.block([[cov[a, b] for b in sep] for a in sep])
            G = jnp.linalg.solve(Lvv, LvS)
            C = -G @ S
            cov[v, v] = jnp.linalg.inv(Lvv) - C @ G.T
            o = 0
            for w in sep:
                cov[v, w] = C[:, o:o + dims[w]]
                cov[w, v] = cov[v, w].T
                o += dims[w]
    if not covariance:
        return mean, {}
    return mean, {**{v: cov[v, v] for v in order}, **{p: cov[p] for p in pairs}}


def eliminate_chain(eta, lam, pair_eta, pair_lam):
    """`eliminate` on a chain ``x[0] … x[T−1]`` in time order, its repeated step
    rolled into a scan, so the cost of compiling it does not grow with T.

    ``eta`` (T, d), ``lam`` (T, d, d) are the potentials on each ``x[t]``;
    ``pair_eta`` (T−1, 2d), ``pair_lam`` (T−1, 2d, 2d) the potential over
    ``(x[t+1], x[t])``. Returns the means (T, d), covariances (T, d, d) and
    cross-covariances Cov(x[t], x[t+1]) (T−1, d, d).
    """
    d = eta.shape[1]

    def forward(msg, step):
        e, L, pe, pL = step
        Lvv, LvS, ev = L + msg[1] + pL[d:, d:], pL[d:, :d], e + msg[0] + pe[d:]
        G = jnp.linalg.solve(Lvv, jnp.concatenate([LvS, ev[:, None]], axis=1))
        out = (pe[:d] - LvS.T @ G[:, -1], pL[:d, :d] - LvS.T @ G[:, :-1])
        return out, (Lvv, LvS, ev)

    zero = (jnp.zeros(d, eta.dtype), jnp.zeros((d, d), eta.dtype))
    msg, (Lvv, LvS, ev) = jax.lax.scan(forward, zero, (eta[:-1], lam[:-1], pair_eta, pair_lam))
    L_last = lam[-1] + msg[1]
    m_last = jnp.linalg.solve(L_last, eta[-1] + msg[0])
    S_last = jnp.linalg.inv(L_last)

    def backward(nxt, step):
        m_next, S_next = nxt
        Lvv, LvS, ev = step
        m = jnp.linalg.solve(Lvv, ev - LvS @ m_next)
        # Σ_vS = −Λvv⁻¹ ΛvS Σ_SS,  Σ_vv = Λvv⁻¹ + Λvv⁻¹ ΛvS Σ_SS ΛSv Λvv⁻¹
        G = jnp.linalg.solve(Lvv, LvS)
        C = -G @ S_next
        S = jnp.linalg.inv(Lvv) - C @ G.T
        return (m, S), (m, S, C)

    _, (m, S, C) = jax.lax.scan(backward, (m_last, S_last), (Lvv, LvS, ev), reverse=True)
    return (jnp.concatenate([m, m_last[None]]), jnp.concatenate([S, S_last[None]]), C)


class Graph:
    """A set of `Normal` (and `Box`) nodes over named variables.

    ``order`` fixes the elimination ordering (default: first appearance).
    Any ordering gives the same answer; a good one keeps separators small.
    """

    def __init__(self, nodes=(), order=None):
        self.nodes: list[Normal] = []
        self.constraints: list[Box] = []
        self.dims: dict[str, int] = {}
        self._order = order
        for n in nodes:
            self.add(n)

    def add(self, node):
        for v in node.vars:
            if self.dims.setdefault(v.name, v.dim) != v.dim:
                raise ValueError(f"variable {v.name!r} used with dims {self.dims[v.name]} and {v.dim}")
        (self.constraints if isinstance(node, Box) else self.nodes).append(node)
        return node

    def project(self, values: dict) -> dict:
        for c in self.constraints:
            values = c.project(values)
        return values

    @property
    def order(self):
        if self._order is None:
            return tuple(self.dims)
        if set(self._order) != set(self.dims):
            raise ValueError(f"order {self._order} does not match variables {tuple(self.dims)}")
        return tuple(self._order)

    def energy(self, values: dict):
        return sum(n.energy(values) for n in self.nodes)

    def _solve(self, values, covariance):
        return eliminate([n.linearize(values) for n in self.nodes], self.order, self.dims,
                         covariance)

    def infer(self, init: dict, iterations: int = 10, steps=DEFAULT_STEPS) -> Marginals:
        """Iterated-linearization sum-product from ``init`` (one value per
        variable). Means are the final estimate; covariances are those of
        the graph linearized there (the Laplace approximation)."""
        values = self.project({v: jnp.asarray(init[v], dtype=jnp.result_type(float))
                               for v in self.dims})
        energy = self.energy(values)

        def iteration(_, carry):
            values, energy = carry
            target, _ = self._solve(values, covariance=False)
            accepted = jnp.zeros((), bool)
            new, new_energy = values, energy
            for s in steps:
                cand = self.project({v: values[v] + s * (target[v] - values[v]) for v in values})
                e = self.energy(cand)
                take = ~accepted & (e < energy)
                new = jax.tree.map(lambda c, o, take=take: jnp.where(take, c, o), cand, new)
                new_energy = jnp.where(take, e, new_energy)
                accepted = accepted | take
            return new, new_energy

        if iterations:
            values, energy = jax.lax.fori_loop(0, iterations, iteration, (values, energy))
        _, cov = self._solve(values, covariance=True)
        return Marginals(values, cov, energy)

    def infer_restarts(self, inits: dict, iterations: int = 10, steps=DEFAULT_STEPS) -> Marginals:
        """`infer` from each of several initial estimates (leading axis of
        every ``inits`` value) in parallel; returns the lowest-energy result."""
        runs = jax.vmap(lambda init: self.infer(init, iterations, steps))(inits)
        best = jnp.argmin(runs.energy)
        return jax.tree.map(lambda x: x[best], runs)
