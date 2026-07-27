"""Graph container for joint-state Bayesian models.

A `Block` pairs a transition (how a state slice evolves) with an `Observation`
(raw datum → Gaussian message on that slice). A `JointModel` composes blocks
and provides four message-passing operations:

    model.learn(trajectories)   # E-step (VMP) + optional M-step (observation params)
    model.smooth(observations)  # forward-backward over the chain
    model.filter(stream)        # predict-update belief over the stream
    model.plan(observations)    # action inference toward a goal/setpoint

`LinearCoupling` factors information across slices.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .config import (
    PLAN_OFFSET_PIN_PRECISION,
    PLAN_VMP_ITERATIONS,
    POINT_ESTIMATE_COV,
)
from .distributions import (
    Gaussian,
    Wishart,
    combine_gaussians,
    gaussian_mean,
    gaussian_mean_cov,
    gaussian_prior,
    vague_gaussian,
)
from .message_passing import (
    compute_marginals,
    forward_backward,
    infer_actions,
    infer_actions_exact,
)
from .nodes.transition import CTCache, CTMeta
from .observations import _as_observation
from .transitions import KnownPhysics


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
        d_s, d_z = b_from.eta.shape[0], b_to.eta.shape[0]
        P = self.noise_prec * jnp.eye(d_z)
        MtP = self.M.T @ P
        Lam = jnp.block([
            [b_from.lam + MtP @ self.M, -MtP],
            [-MtP.T,                     b_to.lam + P],
        ])
        eta = jnp.concatenate([
            b_from.eta - MtP @ self.offset,
            b_to.eta + P @ self.offset,
        ])
        cov = jnp.linalg.inv(Lam)
        mu = cov @ eta
        lam_s = jnp.linalg.inv(cov[:d_s, :d_s])
        lam_z = jnp.linalg.inv(cov[d_s:, d_s:])
        return (Gaussian(eta=lam_s @ mu[:d_s], lam=lam_s),
                Gaussian(eta=lam_z @ mu[d_s:], lam=lam_z))


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


def _identity_meta(d):
    return CTMeta(lambda a: a.reshape(d, d))


def _known_plan_cache(tr: KnownPhysics, prior_x: Gaussian):
    """Build a point-estimate cache for a KnownPhysics block at the current
    belief. Returns (cache, eff_du) where eff_du = du + 1 (the extra channel
    carries the affine offset via a pinned-1 control)."""
    d, du = tr.dim, tr.du
    A, B, offset = tr.linearize(gaussian_mean(prior_x))
    A = jnp.asarray(A)
    B = jnp.asarray(B).reshape(d, du)
    offset = jnp.asarray(offset).reshape(d, 1)
    B_aug = jnp.concatenate([B, offset], axis=1)
    eff_du = du + 1

    df = 100.0
    q_a = gaussian_prior(d * d, POINT_ESTIMATE_COV, A.ravel())
    q_b = gaussian_prior(d * eff_du, POINT_ESTIMATE_COV, B_aug.ravel())
    q_W = Wishart(df=df, inv_scale=df * tr.Q)         # E[W] = Q⁻¹
    return CTCache(q_a, q_W, _identity_meta(d), q_b), eff_du


def _augment_prior_u(prior_u: Gaussian, eff_du: int) -> Gaussian:
    """Append a tight pinned-1 channel for the constant offset."""
    du = prior_u.eta.shape[0]
    eta = jnp.concatenate(
        [prior_u.eta, jnp.array([PLAN_OFFSET_PIN_PRECISION])])
    lam = (jnp.zeros((eff_du, eff_du))
           .at[:du, :du].set(prior_u.lam)
           .at[-1, -1].set(PLAN_OFFSET_PIN_PRECISION))
    return Gaussian(eta=eta, lam=lam)


def _default_action_prior(d, du, q_b, raw, msgs, start_mean, eff_du=None):
    """Action prior from start→goal latent shift: var(u_i) = ‖μ_e − μ_s‖² / ‖B_col_i‖²."""
    mB = gaussian_mean(q_b).reshape(d, eff_du or du)[:, :du]
    b_col_var = jnp.sum(mB ** 2, axis=0) + 1e-8
    present = [i for i, o in enumerate(raw) if o is not None]
    if start_mean is not None and present:
        mu_s, mu_e = start_mean, gaussian_mean(msgs[present[-1]])
    elif len(present) >= 2:
        mu_s, mu_e = gaussian_mean(msgs[present[0]]), gaussian_mean(msgs[present[-1]])
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

    def plan(self, observations, n_iterations=PLAN_VMP_ITERATIONS,
             prior_x=None, prior_u=None, method="exact"):
        """Infer the action sequence — planning as inference. `observations[block]`
        is a length-T list of raw observations or `None` (vague). `prior_x` is
        the carried belief (defaults to N(0, I)). `prior_u` defaults to a
        start→goal-shift prior for learned dynamics; supply explicitly for
        KnownPhysics / setpoint regulation.

        ``method="exact"`` (default) computes the exact Gaussian action
        posterior in one solve (LQG duality) — no iteration, no mean-field.
        ``method="vmp"`` uses the iterative coordinate-ascent VMP updates
        (`n_iterations` applies only there).

        Returns the planned actions of shape `(T-1, du)`.
        """
        b = self._controllable()
        tr = b.transition
        d, du = b.dim, tr.du

        raw = observations[b.name]
        msgs = [vague_gaussian(d) if o is None else b.observation.message(o) for o in raw]
        if prior_x is None:
            prior_x = Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))
            start_mean = None
        else:
            start_mean = gaussian_mean(prior_x)

        def _solve(cache, pu):
            if method == "exact":
                return infer_actions_exact(prior_x, msgs, cache, pu)
            return infer_actions(prior_x, msgs, cache, pu, n_iterations=n_iterations)

        if tr.learned:
            if tr.q_b is None and prior_u is None:
                raise ValueError(
                    f"plan: block '{b.name}' has du={du} but q(B) was never learned — "
                    "include 'control' in the training trajectories so B is learned, "
                    "or pass prior_u explicitly.")
            cache = CTCache(tr.q_a, tr.q_W, _identity_meta(d), tr.q_b)
            if prior_u is None:
                prior_u = _default_action_prior(d, du, tr.q_b, raw, msgs, start_mean,
                                                eff_du=tr.eff_du)
            if tr.offset:                     # pin the learned-drift channel to 1
                return _solve(cache, _augment_prior_u(prior_u, tr.eff_du))[:, :du]
            return _solve(cache, prior_u)

        # KnownPhysics: re-linearize at current belief, fold offset into a
        # pinned-1 control channel.
        cache, eff_du = _known_plan_cache(tr, prior_x)
        if prior_u is None:
            prior_u = Gaussian(
                eta=jnp.zeros(du),
                lam=jnp.eye(du) * 1e-2)
        return _solve(cache, _augment_prior_u(prior_u, eff_du))[:, :du]

    def smooth(self, observations, n_predict=0, controls=None, predict_controls=None):
        """Forward-backward smoothing (+ optional rollout). Returns
        `{block_name: {"means", "covs", "predictions"}}` per learned block."""
        out = {}
        for b in self.blocks:
            if not b.transition.learned:
                continue
            d, tr = b.dim, b.transition
            cache = CTCache(tr.q_a, tr.q_W, _identity_meta(d), tr.q_b)
            raw = observations[b.name]
            msgs = [b.observation.message(o) for o in raw] + [vague_gaussian(d)] * n_predict
            all_ctrl = None
            offset = getattr(tr, "offset", False)
            if controls is not None or offset:
                base = (list(controls) if controls is not None
                        else [jnp.zeros(tr.du)] * (len(raw) - 1))
                pc = predict_controls if predict_controls is not None else [jnp.zeros(tr.du)] * n_predict
                all_ctrl = [jnp.asarray(u) for u in (base + list(pc))]
                if offset:
                    all_ctrl = [tr._augment_u(u) for u in all_ctrl]
            prior_x = Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))
            alphas, betas, _, _ = forward_backward(prior_x, msgs, cache, all_ctrl)
            marg = compute_marginals(alphas, betas, msgs)
            means, covs = jax.vmap(gaussian_mean_cov)(marg)
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
