"""Composable block API for joint-state Bayesian models.

A `Block` pairs a `Transition` (how a state slice evolves) with an `Observation`
(raw datum → Gaussian message on that slice). A `JointModel` composes blocks
and provides four message-passing operations:

    model.learn(trajectories)   # E-step (VMP) + optional M-step (observation params)
    model.smooth(observations)  # forward-backward over the chain
    model.filter(stream)        # predict-update belief over the stream
    model.plan(observations)    # VMP on the action sequence to a goal/setpoint

`LinearCoupling` factors information across slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from .distributions import (
    Gaussian, Wishart, combine_gaussians, vague_gaussian,
    gaussian_mean, gaussian_mean_cov, gaussian_prior, wishart_mean,
)
from .defaults import PRIOR_A_COV, PRIOR_B_COV, PRIOR_W_DF, INIT_A_COV, INIT_B_COV
from .nodes.transition import CTMeta, CTCache, ct_forward
from .message_passing import (
    forward_backward, accumulate_vmp_messages, compute_marginals, infer_actions,
)
from .nn.vae import LOG_STD_CLIP, PROB_CLIP, _as_batch, _latest_frame


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class Observation:
    """Maps a raw datum to a Gaussian message on its block's state slice.

    Subclasses define `message(datum) -> Gaussian`. Learnable observations
    additionally define `update(data, post_means, post_covs)` for the M-step.
    """
    learnable = False


class Frozen(Observation):
    """Fixed encoder: wraps `encode: datum -> Gaussian` and optional `decode: z -> datum`."""

    def __init__(self, encode: Callable, decode: Callable | None = None):
        self.encode = encode
        if decode is not None:
            self.decode = decode

    def message(self, datum) -> Gaussian:
        return self.encode(datum)


def _kl_diag_vs_full(mu1, log_std1, mu2, cov2):
    """KL( N(mu1, diag(σ1²)) || N(mu2, Σ2) )."""
    d = mu1.shape[0]
    var1 = jnp.exp(2.0 * log_std1)
    prec2 = jnp.linalg.inv(cov2)
    diff = mu2 - mu1
    return 0.5 * (jnp.sum(jnp.diag(prec2) * var1) + diff @ prec2 @ diff
                  - d + jnp.linalg.slogdet(cov2)[1] - 2.0 * jnp.sum(log_std1))


def _vae_m_step_loss(params, model, images, post_means, post_covs, z_rng, beta_recon):
    """ELBO for the VAE M-step: BCE reconstruction + KL(encoder ‖ smoothed posterior).

    `images` is the encoder input — a window `(B, K, 28, 28)` (multi-frame)
    or `(B, 28, 28)` (single-frame). The decoder reconstructs the full input —
    the whole K-frame window for multi-frame, so the latent must encode motion."""
    mu_enc, log_std_enc = model.apply(params, images, method=model.encode)
    log_std_enc = jnp.clip(log_std_enc, *LOG_STD_CLIP)
    z = mu_enc + jnp.exp(log_std_enc) * jax.random.normal(z_rng, mu_enc.shape)
    recon = jnp.clip(model.apply(params, z, method=model.decode), *PROB_CLIP)
    target = images                       # reconstruct the whole window (was last frame only)
    flat = target.reshape(target.shape[0], -1)
    bce = -jnp.sum(flat * jnp.log(recon) + (1 - flat) * jnp.log(1 - recon), axis=-1)
    kl = jax.vmap(_kl_diag_vs_full)(mu_enc, log_std_enc, post_means, post_covs)
    return jnp.mean(beta_recon * bce + kl)


class LearnedVAE(Observation):
    """VAE observation whose encoder produces messages and whose weights are
    refined in the M-step. Follows the VAE's `n_frames` setting — when the
    underlying VAE is multi-frame, the encoder input is a K-frame window
    `(K, 28, 28)` and the decoder reconstructs the whole window; `decode`
    returns its latest frame."""
    learnable = True

    def __init__(self, model, params, lr=5e-5, n_m_steps=20,
                 beta_recon=1.0, seed=0):
        self.model = model
        self.params = params
        self.n_frames = getattr(model, "n_frames", 1)
        self.n_m_steps = n_m_steps
        self.beta_recon = beta_recon
        self._tx = optax.adam(lr)
        self._opt_state = self._tx.init(params)
        self._rng = jax.random.PRNGKey(seed)
        self.loss_history: list[float] = []

    def _shape(self, x):
        return _as_batch(x, self.n_frames)

    def message(self, image) -> Gaussian:
        mu, log_std = self.model.apply(self.params, self._shape(image), method=self.model.encode)
        log_std = jnp.clip(log_std[0], *LOG_STD_CLIP)
        lam = jnp.diag(1.0 / jnp.exp(2.0 * log_std))
        return Gaussian(eta=lam @ mu[0], lam=lam)

    def update(self, data, post_means, post_covs):
        images = jnp.stack([jnp.asarray(im) for im in data])

        @jax.jit
        def step(params, opt_state, z_rng):
            loss, grads = jax.value_and_grad(_vae_m_step_loss)(
                params, self.model, images, post_means, post_covs, z_rng, self.beta_recon)
            updates, opt_state = self._tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for _ in range(self.n_m_steps):
            self._rng, z_rng = jax.random.split(self._rng)
            self.params, self._opt_state, loss = step(self.params, self._opt_state, z_rng)
            self.loss_history.append(float(loss))

    def decode(self, z):
        return _latest_frame(self.model.apply(self.params, jnp.asarray(z).reshape(1, -1),
                                              method=self.model.decode))


def _as_observation(obj) -> Observation:
    return obj if isinstance(obj, Observation) else Frozen(obj)


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

class KnownPhysics:
    """Gray-box dynamics: `linearize(belief_mean) -> (A, B, offset)` defines a
    local linear-Gaussian map  s' = A·s + B·u + offset + ε  re-evaluated per step.
    """
    learned = False

    def __init__(self, dim, linearize: Callable, du=0, process_std=1e-2):
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
    """Linear-Gaussian dynamics  x' ~ N(A·x + B·u, W⁻¹)  learned by the CT node.

    Two aggregation modes pick the statistical question to answer:

    * ``mode="joint"`` (default) — joint posterior q(A,B,W | all trajectories).
      Variance-optimal when one global (A,B,W) generated the data; right for
      system identification.

    * ``mode="per_trajectory"`` — sequential VMP, trajectory by trajectory;
      final q is the last trajectory's local fit (NOT a joint posterior).
      Gives the M-step per-trajectory-consistent latent targets — what aligns
      a VAE encoder across trajectories — and leaves q(B) as a local
      linearization a receding-horizon planner can use even when the encoder
      isn't globally affine.
    """
    learned = True

    def __init__(self, dim, du=0, n_iterations=50, mode="joint",
                 prior_a_cov=PRIOR_A_COV, prior_a_mean=None, init_a_cov=INIT_A_COV,
                 prior_b_cov=PRIOR_B_COV, init_b_cov=INIT_B_COV, prior_W_df=PRIOR_W_DF):
        if mode not in ("joint", "per_trajectory"):
            raise ValueError(f"mode must be 'joint' or 'per_trajectory', got {mode!r}")
        self.dim = dim
        self.du = du
        self.n_iterations = n_iterations
        self.mode = mode
        self.prior_a_cov, self.prior_a_mean, self.init_a_cov = prior_a_cov, prior_a_mean, init_a_cov
        self.prior_b_cov, self.init_b_cov, self.prior_W_df = prior_b_cov, init_b_cov, prior_W_df
        self.q_a = self.q_W = self.q_b = None

    @property
    def _meta(self):
        d = self.dim
        return CTMeta(lambda a: a.reshape(d, d))

    def _priors(self):
        d, du = self.dim, self.du
        prior_a = gaussian_prior(d * d, self.prior_a_cov, self.prior_a_mean)
        prior_W = Wishart(df=self.prior_W_df, inv_scale=jnp.eye(d))
        prior_b = gaussian_prior(d * du, self.prior_b_cov) if du > 0 else None
        return prior_a, prior_W, prior_b

    def _init_q(self, has_ctrl):
        d, du = self.dim, self.du
        q_a = self.q_a if self.q_a is not None else gaussian_prior(d * d, self.init_a_cov, self.prior_a_mean)
        q_W = self.q_W if self.q_W is not None else Wishart(df=self.prior_W_df, inv_scale=jnp.eye(d))
        q_b = self.q_b if self.q_b is not None else (gaussian_prior(d * du, self.init_b_cov) if has_ctrl else None)
        return q_a, q_W, q_b

    def learn(self, msg_seqs, ctrl_seqs=None):
        """One E-step. Returns concatenated smoothed `(means, covs)` (M-step input)."""
        du = self.du
        has_ctrl = ctrl_seqs is not None and du > 0
        prior_a, prior_W, prior_b = self._priors()
        q_a, q_W, q_b = self._init_q(has_ctrl)
        prior_x = Gaussian(eta=jnp.zeros(self.dim), lam=jnp.eye(self.dim))
        meta = self._meta

        def _ctrl_for(i):
            return [jnp.asarray(a) for a in ctrl_seqs[i]] if has_ctrl else None

        def _bp(cache):
            mxs, mys, ab = [], [], []
            for i, seq in enumerate(msg_seqs):
                al, be, m_x, m_y = forward_backward(prior_x, seq, cache, _ctrl_for(i))
                mxs.append(m_x); mys.append(m_y); ab.append((al, be, seq))
            return mxs, mys, ab

        def _stack(seq_of_gs):
            return Gaussian(eta=jnp.concatenate([g.eta for g in seq_of_gs]),
                            lam=jnp.concatenate([g.lam for g in seq_of_gs]))

        if self.mode == "joint":
            actions = (jnp.concatenate([jnp.stack([jnp.asarray(a) for a in c])
                                        for c in ctrl_seqs]) if has_ctrl else None)
            for _ in range(self.n_iterations):
                cache = CTCache(q_a, q_W, meta, q_b)
                mxs, mys, _ = _bp(cache)
                q_a, q_W, q_b = accumulate_vmp_messages(
                    _stack(mxs), _stack(mys), cache, actions, prior_a, prior_W, prior_b)
            cache = CTCache(q_a, q_W, meta, q_b)
            _, _, ab = _bp(cache)
        else:                              # per_trajectory
            ab = []
            for i, seq in enumerate(msg_seqs):
                u = _ctrl_for(i)
                for _ in range(self.n_iterations):
                    cache = CTCache(q_a, q_W, meta, q_b)
                    _, _, m_xs, m_ys = forward_backward(prior_x, seq, cache, u)
                    q_a, q_W, q_b = accumulate_vmp_messages(
                        m_xs, m_ys, cache, u, prior_a, prior_W, prior_b)
                cache = CTCache(q_a, q_W, meta, q_b)
                al, be, _, _ = forward_backward(prior_x, seq, cache, u)
                ab.append((al, be, seq))

        self.q_a, self.q_W, self.q_b = q_a, q_W, q_b
        means, covs = [], []
        for al, be, seq in ab:
            m, c = jax.vmap(gaussian_mean_cov)(compute_marginals(al, be, seq))
            means.append(m); covs.append(c)
        return jnp.concatenate(means), jnp.concatenate(covs)

    def predict(self, belief: Gaussian, u=None) -> Gaussian:
        cache = CTCache(self.q_a, self.q_W, self._meta, self.q_b)
        return ct_forward(belief, cache, jnp.asarray(u) if u is not None else None)

    @property
    def A(self):
        return gaussian_mean(self.q_a).reshape(self.dim, self.dim)

    @property
    def B(self):
        return None if self.q_b is None else gaussian_mean(self.q_b).reshape(self.dim, self.du)


class LearnedAffine:
    """Fully-observed Bayesian linear regression  y = A·x + B·u + ε  via the
    same CT-factor VMP — rectangular A `(output_dim, input_dim)`. Used when x
    and y are observed (no chain smoothing) — e.g. learning physics from
    proprioception with x = [sinθ, ω], y = ω', u = τ.
    """
    learned = True

    def __init__(self, input_dim, output_dim, du=0, n_iterations=20,
                 prior_a_cov=PRIOR_A_COV, prior_b_cov=PRIOR_B_COV,
                 prior_W_df=PRIOR_W_DF, obs_prec=1e6):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.du = du
        self.n_iterations = n_iterations
        self.prior_a_cov = prior_a_cov
        self.prior_b_cov = prior_b_cov
        self.prior_W_df = prior_W_df
        self.obs_prec = obs_prec
        self.q_a = self.q_W = self.q_b = None

    @property
    def _meta(self):
        dx, dy = self.input_dim, self.output_dim
        return CTMeta(lambda a: a.reshape(dy, dx))

    def learn(self, x_seqs, y_seqs, u_seqs=None):
        """Refine q(A,B,W) from `(x, y, u)` triplets. Warm-starts across calls."""
        dx, dy, du = self.input_dim, self.output_dim, self.du
        has_ctrl = u_seqs is not None and du > 0
        p = self.obs_prec

        x = jnp.concatenate([jnp.asarray(s) for s in x_seqs], axis=0)
        y = jnp.concatenate([jnp.asarray(s) for s in y_seqs], axis=0)
        u = (jnp.concatenate([jnp.asarray(s) for s in u_seqs], axis=0)
             if has_ctrl else None)
        n = x.shape[0]

        m_xs = Gaussian(eta=p * x, lam=jnp.broadcast_to(p * jnp.eye(dx), (n, dx, dx)))
        m_ys = Gaussian(eta=p * y, lam=jnp.broadcast_to(p * jnp.eye(dy), (n, dy, dy)))

        prior_a = gaussian_prior(dx * dy, self.prior_a_cov)
        prior_W = Wishart(df=self.prior_W_df, inv_scale=jnp.eye(dy))
        prior_b = gaussian_prior(dy * du, self.prior_b_cov) if has_ctrl else None

        q_a = self.q_a if self.q_a is not None else prior_a
        q_W = self.q_W if self.q_W is not None else prior_W
        q_b = self.q_b if self.q_b is not None else prior_b

        for _ in range(self.n_iterations):
            cache = CTCache(q_a, q_W, self._meta, q_b)
            q_a, q_W, q_b = accumulate_vmp_messages(
                m_xs, m_ys, cache, u, prior_a, prior_W, prior_b)
        self.q_a, self.q_W, self.q_b = q_a, q_W, q_b
        return self

    @property
    def A(self):
        return gaussian_mean(self.q_a).reshape(self.output_dim, self.input_dim)

    @property
    def B(self):
        return None if self.q_b is None else gaussian_mean(self.q_b).reshape(self.output_dim, self.du)

    @property
    def noise_cov(self):
        return jnp.linalg.inv(wishart_mean(self.q_W))


# ---------------------------------------------------------------------------
# Cross-block factors
# ---------------------------------------------------------------------------

class LinearCoupling:
    """Gaussian factor  z = M·s + b + ε,  ε ~ N(0, P⁻¹), linking two block
    slices. `fuse(b_from, b_to)` returns the marginalised beliefs after
    applying the factor — information flows both ways.

    `offset` defaults to zero, preserving the original linear `z = M·s + ε`
    behavior. Use `LinearCoupling.fit(...)` to estimate a reusable affine
    coupling from paired samples.
    """

    def __init__(self, from_name: str, to_name: str, M, noise_prec, offset=None):
        self.from_name = from_name
        self.to_name = to_name
        self.M = jnp.asarray(M)
        self.noise_prec = float(noise_prec)
        d_z = self.M.shape[0]
        self.offset = jnp.zeros(d_z) if offset is None else jnp.asarray(offset).reshape(d_z)

    @classmethod
    def fit(cls, from_name: str, to_name: str, x_from, x_to, ridge: float = 1e-3, affine: bool = True):
        """Fit `x_to ~= M @ x_from + offset` by ridge regression.

        Returns `(coupling, diagnostics)` where diagnostics includes residual
        variance and offset norm. `x_from` and `x_to` are arrays with shape
        `(n_samples, dim)`.
        """
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
        noise_var = float(jnp.mean(residual ** 2) + ridge)
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


# ---------------------------------------------------------------------------
# Block + JointModel
# ---------------------------------------------------------------------------

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
    q_a = gaussian_prior(d * d, 1e-8, A.ravel())
    q_b = gaussian_prior(d * eff_du, 1e-8, B_aug.ravel())
    q_W = Wishart(df=df, inv_scale=df * tr.Q)         # E[W] = Q⁻¹
    return CTCache(q_a, q_W, _identity_meta(d), q_b), eff_du


def _augment_prior_u(prior_u: Gaussian, eff_du: int) -> Gaussian:
    """Append a tight pinned-1 channel for the constant offset."""
    du = prior_u.eta.shape[0]
    eta = jnp.concatenate([prior_u.eta, jnp.array([1e8])])
    lam = (jnp.zeros((eff_du, eff_du))
           .at[:du, :du].set(prior_u.lam)
           .at[-1, -1].set(1e8))
    return Gaussian(eta=eta, lam=lam)


def _default_action_prior(d, du, q_b, raw, msgs, start_mean):
    """Action prior from start→goal latent shift: var(u_i) = ‖μ_e − μ_s‖² / ‖B_col_i‖²."""
    mB = gaussian_mean(q_b).reshape(d, du)
    b_col_var = jnp.sum(mB ** 2, axis=0) + 1e-8
    present = [i for i, o in enumerate(raw) if o is not None]
    if start_mean is not None and present:
        mu_s, mu_e = start_mean, gaussian_mean(msgs[present[-1]])
    elif len(present) >= 2:
        mu_s, mu_e = gaussian_mean(msgs[present[0]]), gaussian_mean(msgs[present[-1]])
    else:
        return Gaussian(eta=jnp.zeros(du), lam=jnp.diag(b_col_var))
    shift = jnp.maximum(jnp.sum((mu_e - mu_s) ** 2), 1e-8)
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

    # ---- learning ----------------------------------------------------------

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

    # ---- planning ----------------------------------------------------------

    def _controllable(self) -> Block:
        ctrl = [b for b in self.blocks if getattr(b.transition, "du", 0) > 0]
        if not ctrl:
            raise ValueError("plan: no controllable block (transition.du > 0)")
        if len(ctrl) > 1:
            raise NotImplementedError("plan: multi-controllable planning requires coupling resolution")
        return ctrl[0]

    def plan(self, observations, n_iterations=50, prior_x=None, prior_u=None):
        """Infer the action sequence via VMP on `u`. `observations[block]` is
        a length-T list of raw observations or `None` (vague). `prior_x` is
        the carried belief (defaults to N(0, I)). `prior_u` defaults to a
        start→goal-shift prior for learned dynamics; supply explicitly for
        KnownPhysics / setpoint regulation.

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

        if tr.learned:
            if tr.q_b is None and prior_u is None:
                raise ValueError(
                    f"plan: block '{b.name}' has du={du} but q(B) was never learned — "
                    "include 'control' in the training trajectories so B is learned, "
                    "or pass prior_u explicitly.")
            cache = CTCache(tr.q_a, tr.q_W, _identity_meta(d), tr.q_b)
            if prior_u is None:
                prior_u = _default_action_prior(d, du, tr.q_b, raw, msgs, start_mean)
            return infer_actions(prior_x, msgs, cache, prior_u, n_iterations=n_iterations)

        # KnownPhysics: re-linearize at current belief, fold offset into a
        # pinned-1 control channel.
        cache, eff_du = _known_plan_cache(tr, prior_x)
        if prior_u is None:
            prior_u = Gaussian(eta=jnp.zeros(du), lam=jnp.eye(du) * 1e-2)
        u_aug = infer_actions(prior_x, msgs, cache, _augment_prior_u(prior_u, eff_du),
                              n_iterations=n_iterations)
        return u_aug[:, :du]

    # ---- smoothing & filtering --------------------------------------------

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
            if controls is not None:
                pc = predict_controls if predict_controls is not None else [jnp.zeros(tr.du)] * n_predict
                all_ctrl = [jnp.asarray(u) for u in (list(controls) + list(pc))]
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
