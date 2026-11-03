"""Graph container for joint-state Bayesian models.

A `Block` pairs a transition (how a state slice evolves) with an `Observation`
(raw datum → Gaussian message on that slice). A `JointModel` composes blocks
and provides four message-passing operations:

    model.learn(trajectories)   # E-step (VMP) + optional M-step (observation params)
    model.smooth(observations)  # forward-backward over the chain
    model.filter(stream)        # predict-update belief over the stream
    model.plan(observations)    # action inference toward a goal/setpoint

`LinearCoupling` factors information across slices. Smoothing, planning and
coupling are each one `jopa.fg` graph, solved by the same elimination.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .distributions import Gaussian, combine_gaussians, gaussian_mean, wishart_mean
from .fg import Fn, Graph, Normal, Variable
from .nodes.dynamics import Linear
from .observations import _as_observation

# `jax.enable_x64` is the context manager from JAX 0.7 on; the last releases for
# Python 3.10 only ship the experimental one.
_enable_x64 = getattr(jax, "enable_x64", None)
if _enable_x64 is None:
    from jax.experimental import enable_x64 as _enable_x64


class LinearCoupling:
    """Gaussian factor  z = M·s + b + ε,  ε ~ N(0, P⁻¹), linking two block
    slices. `fuse(b_from, b_to)` returns the marginalised beliefs after
    applying the factor — information flows both ways.

    `offset` defaults to zero, giving the purely linear `z = M·s + ε`. Use
    `LinearCoupling.fit(...)` to estimate a reusable affine coupling from
    paired samples.
    """

    def __init__(self, from_name: str, to_name: str, M, noise_prec, offset=None):
        self.from_name = from_name
        self.to_name = to_name
        self.M = jnp.asarray(M)
        self.noise_prec = float(noise_prec)
        d_z = self.M.shape[0]
        self.offset = jnp.zeros(d_z) if offset is None else jnp.asarray(offset).reshape(d_z)

    @classmethod
    def fit(
        cls,
        from_name: str,
        to_name: str,
        x_from,
        x_to,
        ridge: float = 1e-3,
        affine: bool = True,
        noise_floor: float = 1e-8,
    ):
        """Fit `x_to ~= M @ x_from + offset` by ridge regression.

        Returns `(coupling, diagnostics)` where diagnostics includes residual
        variance and offset norm. `x_from` and `x_to` are arrays with shape
        `(n_samples, dim)`.
        """
        if noise_floor <= 0:
            raise ValueError("noise_floor must be strictly positive")
        x_from = jnp.asarray(x_from)
        x_to = jnp.asarray(x_to)
        n = min(x_from.shape[0], x_to.shape[0])
        x_from, x_to = x_from[:n], x_to[:n]
        if affine:
            design = jnp.concatenate([x_from, jnp.ones((n, 1), dtype=x_from.dtype)], axis=1)
        else:
            design = x_from
        reg = ridge * jnp.eye(design.shape[1], dtype=design.dtype)
        if affine:
            reg = reg.at[-1, -1].set(0.0)
        coef = jnp.linalg.solve(design.T @ design + reg, design.T @ x_to).T
        M = coef[:, :-1] if affine else coef
        offset = coef[:, -1] if affine else jnp.zeros(x_to.shape[1], dtype=x_to.dtype)
        pred = design @ coef.T
        residual = x_to - pred
        noise_var = max(float(jnp.mean(residual ** 2)), noise_floor)
        coupling = cls(from_name, to_name, M=M, offset=offset, noise_prec=1.0 / noise_var)
        diagnostics = {
            "noise_var": noise_var,
            "offset_norm": float(jnp.linalg.norm(offset)),
            "residual_rmse": float(jnp.sqrt(jnp.mean(residual ** 2))),
        }
        return coupling, diagnostics

    def fuse(self, b_from: Gaussian, b_to: Gaussian):
        s = Variable("s", b_from.eta.shape[0])
        z = Variable("z", b_to.eta.shape[0])
        post = Graph([
            Normal.from_gaussian(s, b_from), Normal.from_gaussian(z, b_to),
            Normal(z, Fn(lambda v: self.M @ v + self.offset, s), self.noise_prec),
        ]).infer({"s": jnp.zeros(s.dim), "z": jnp.zeros(z.dim)}, iterations=1)
        return post.gaussian("s"), post.gaussian("z")


@dataclass
class Block:
    name: str
    transition: object                      # KnownPhysics | LearnedLinear
    observe: object                         # Callable | Observation (auto-wrapped)

    def __post_init__(self):
        self.observation = _as_observation(self.observe)

    @property
    def dim(self):
        return self.transition.dim


def _default_action_prior(B, msgs, start_mean):
    """Action prior from start→goal latent shift: var(u_i) = ‖μ_e − μ_s‖² / ‖B_col_i‖²."""
    du = B.shape[1]
    b_col_var = jnp.sum(B ** 2, axis=0) + 1e-8
    present = [m for m in msgs if m is not None]
    if start_mean is not None and present:
        mu_s, mu_e = start_mean, gaussian_mean(present[-1])
    elif len(present) >= 2:
        mu_s, mu_e = gaussian_mean(present[0]), gaussian_mean(present[-1])
    else:
        return Gaussian(eta=jnp.zeros(du), lam=jnp.diag(b_col_var))
    shift = jnp.maximum(
        jnp.sum((mu_e - mu_s) ** 2), 1e-8)
    return Gaussian(eta=jnp.zeros(du), lam=jnp.diag(b_col_var / shift))


class JointModel:
    """Composes Blocks. Cross-block coupling (optional) is applied during filter."""

    def __init__(self, blocks, coupling=None):
        self.blocks = list(blocks)
        self.couplings = ([] if coupling is None
                          else list(coupling) if isinstance(coupling, (list, tuple))
                          else [coupling])
        self._by_name = {b.name: b for b in self.blocks}
        self.diagnostics: dict = {}

    def __getitem__(self, name) -> Block:
        return self._by_name[name]

    def learn(self, trajectories, n_em=1):
        """Variational EM. Each trajectory is `{block_name: [raw obs ...]}`
        (+ optional `"control": [u per transition]`). E-step refines each
        learned block; M-step refines each learnable observation."""
        ctrl = ([t["control"] for t in trajectories]
                if trajectories and "control" in trajectories[0] else None)
        self.diagnostics = {b.name: {"det_history": [], "eig_history": []}
                            for b in self.blocks if b.transition.learned}
        for _ in range(n_em):
            for b in self.blocks:
                if not b.transition.learned:
                    continue
                msg_seqs = [[b.observation.message(d) for d in traj[b.name]] for traj in trajectories]
                means, covs = b.transition.learn(msg_seqs, ctrl)
                A = b.transition.A
                self.diagnostics[b.name]["det_history"].append(float(jnp.linalg.det(A)))
                self.diagnostics[b.name]["eig_history"].append(jnp.linalg.eigvals(A))
                if b.observation.learnable:
                    raw = [d for traj in trajectories for d in traj[b.name]]
                    b.observation.update(raw, means, covs)

    def remember(self, forget=1.0, **kwargs):
        """Fold each learned block's posterior into its prior — conjugate
        continual learning across successive `learn` calls (see
        `LearnedLinear.remember`). ``forget`` ∈ (0, 1] discounts old evidence."""
        for b in self.blocks:
            if b.transition.learned:
                b.transition.remember(forget, **kwargs)

    def _controllable(self) -> Block:
        ctrl = [b for b in self.blocks if getattr(b.transition, "du", 0) > 0]
        if not ctrl:
            raise ValueError("plan: no controllable block (transition.du > 0)")
        if len(ctrl) > 1:
            raise NotImplementedError("plan: multi-controllable planning requires coupling resolution")
        return ctrl[0]

    def plan(self, observations, prior_x=None, prior_u=None):
        """Infer the action sequence — planning as inference. `observations[block]`
        is a length-T list of raw observations or `None` (unobserved). `prior_x`
        is the carried belief (defaults to N(0, I)). `prior_u` defaults to a
        start→goal-shift prior for learned dynamics; supply explicitly for
        KnownPhysics / setpoint regulation.

        The graph holds the states, the actions and the transition at its
        posterior mean, and is linear-Gaussian, so the action posterior is one
        elimination sweep (LQG duality). Returns actions of shape `(T-1, du)`.
        """
        b = self._controllable()
        tr = b.transition
        d, du = b.dim, tr.du
        msgs = [None if o is None else b.observation.message(o) for o in observations[b.name]]
        start_mean = None if prior_x is None else gaussian_mean(prior_x)
        if prior_x is None:
            prior_x = Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))
        if tr.learned:
            if tr.q_b is None:
                raise ValueError(
                    f"plan: block '{b.name}' has du={du} but q(B) was never learned — "
                    "include 'control' in the training trajectories so B is learned.")
            model, W = tr.dynamics, wishart_mean(tr.q_W)
            if prior_u is None:
                prior_u = _default_action_prior(model.B, msgs, start_mean)
        else:
            A, B, offset = tr.linearize(gaussian_mean(prior_x))
            model = Linear(jnp.asarray(A), jnp.asarray(B).reshape(d, du), jnp.asarray(offset))
            W = jnp.linalg.inv(tr.Q)
            if prior_u is None:
                prior_u = Gaussian(eta=jnp.zeros(du), lam=jnp.eye(du) * 1e-2)

        T = len(msgs)
        xs = [Variable(f"x{t}", d) for t in range(T)]
        us = [Variable(f"u{t}", du) for t in range(T - 1)]
        nodes = [Normal.from_gaussian(xs[0], prior_x)]
        nodes += [Normal.from_gaussian(x, m) for x, m in zip(xs, msgs, strict=True) if m is not None]
        for t in range(T - 1):
            nodes += [Normal.from_gaussian(us[t], prior_u),
                      Normal(xs[t + 1], model.step((xs[t],), us[t]), W)]
        order = [v.name for t in range(T - 1) for v in (xs[t], us[t])] + [xs[-1].name]
        init = {v.name: jnp.zeros(v.dim) for v in xs + us}
        # float64: transition, goal and action precisions span more than float32 resolves
        with _enable_x64(True):
            post = Graph(nodes, order=order).infer(init, iterations=1)
            return jnp.stack([post.mean[u.name] for u in us]).astype(jnp.float32)

    def smooth(self, observations, n_predict=0, controls=None, predict_controls=None):
        """Forward-backward smoothing (+ optional rollout). Returns
        `{block_name: {"means", "covs", "predictions"}}` per learned block.

        The sweep covers the observed slices only; the ``n_predict`` tail is
        the posterior predictive (`ct_predict`) rolled forward from the last
        smoothed marginal. Appending unobserved slices to the VMP sweep instead
        would let each one's parameter-uncertainty term pull the observed
        estimates toward the origin."""
        out = {}
        for b in self.blocks:
            if not b.transition.learned:
                continue
            tr = b.transition
            means, covs = tr.smooth([b.observation.message(o) for o in observations[b.name]],
                                    controls)
            if n_predict:
                future = (list(predict_controls) if predict_controls is not None
                          else [None] * n_predict)
                belief = Gaussian.from_mean(means[-1], jnp.linalg.inv(covs[-1]))
                rolled = []
                for k in range(n_predict):
                    belief = tr.predict(belief, future[k])
                    rolled.append(belief)
                means = jnp.concatenate([means, jnp.stack([g.mean for g in rolled])])
                covs = jnp.concatenate([covs, jnp.stack([g.cov for g in rolled])])
            preds = ([b.observation.decode(means[t]) for t in range(means.shape[0])]
                     if hasattr(b.observation, "decode") else None)
            out[b.name] = {"means": means, "covs": covs, "predictions": preds}
        return out

    def filter(self, stream):
        """Per-block predict-update filter over `stream` (list of
        `{block_name: raw obs}` (+ optional `"control"`)). Applies cross-block
        couplings after each step. Returns the final belief per block."""
        beliefs = {b.name: None for b in self.blocks}
        for step in stream:
            u = step.get("control")
            for b in self.blocks:
                msg = b.observation.message(step[b.name])
                if beliefs[b.name] is None:
                    beliefs[b.name] = msg
                else:
                    beliefs[b.name] = combine_gaussians(b.transition.predict(beliefs[b.name], u), msg)
            for c in self.couplings:
                beliefs[c.from_name], beliefs[c.to_name] = c.fuse(
                    beliefs[c.from_name], beliefs[c.to_name])
        return beliefs
