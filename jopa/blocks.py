"""Modular joint-state model: compose a system from independent `Block`s.

Each block is a slice of the joint state with a `transition` (how it evolves)
and an `observe` map (raw datum → Gaussian message on that slice). Learned
blocks learn their dynamics through the CT-node VMP (the same engine as
`infer`); known blocks just filter. Adding a modality = appending a `Block`.

    model = JointModel([
        Block("proprio", KnownPhysics(2, du=1, linearize=phys), observe=read),
        Block("image",   LearnedLinear(dim=4),                   observe=encode),
    ])
    model.learn(trajectories)        # learned blocks learn their dynamics
    beliefs = model.filter(stream)   # per-block fused beliefs

`coupling` between slices is the single extension point (left for later).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from .distributions import (
    Gaussian, Wishart, combine_gaussians, vague_gaussian,
    gaussian_mean, gaussian_mean_cov, gaussian_prior,
)
from .defaults import PRIOR_A_COV, PRIOR_B_COV, PRIOR_W_DF, INIT_A_COV, INIT_B_COV
from .nodes.transition import CTMeta, CTCache, ct_forward
from .message_passing import (
    forward_backward, accumulate_vmp_messages, compute_marginals, infer_actions,
)
from .nn.vae import LOG_STD_CLIP
from .em import _m_step_loss


def _concat(gaussians):
    """Concatenate stacked Gaussians along the leading (time) axis."""
    return Gaussian(
        eta=jnp.concatenate([g.eta for g in gaussians]),
        lam=jnp.concatenate([g.lam for g in gaussians]),
    )


# ---------------------------------------------------------------------------
# Observations — raw datum → Gaussian message on a state slice
# ---------------------------------------------------------------------------

class Observation:
    """Maps a raw datum to a Gaussian message on its block's state slice.

    ``learnable`` observations (e.g. a VAE, added in B2) additionally implement
    an M-step ``update`` given the smoothed states; frozen ones do not.
    """
    learnable = False

    def message(self, datum) -> Gaussian:
        raise NotImplementedError


class Frozen(Observation):
    """A fixed encoder: wraps a callable ``datum -> Gaussian``."""

    def __init__(self, encode):
        self.encode = encode

    def message(self, datum) -> Gaussian:
        return self.encode(datum)


class LearnedVAE(Observation):
    """A learnable observation: a VAE whose encoder produces the message and
    whose weights are refined in the M-step. Makes `JointModel.learn`
    Variational EM for the block it sits on.
    """
    learnable = True

    def __init__(self, model, params, lr=5e-5, n_m_steps=20, beta_recon=1.0, seed=0):
        self.model = model
        self.params = params
        self.n_m_steps = n_m_steps
        self.beta_recon = beta_recon
        self._tx = optax.adam(lr)
        self._opt_state = self._tx.init(params)
        self._rng = jax.random.PRNGKey(seed)
        self.loss_history = []

    def message(self, image) -> Gaussian:
        mu, log_std = self.model.apply(
            self.params, jnp.asarray(image).reshape(1, 28, 28), method=self.model.encode)
        log_std = jnp.clip(log_std[0], *LOG_STD_CLIP)
        var = jnp.exp(2.0 * log_std)
        lam = jnp.diag(1.0 / var)
        return Gaussian(eta=lam @ mu[0], lam=lam)

    def update(self, images, post_means, post_covs):
        """M-step: refine the VAE weights against the smoothed posterior."""
        images = jnp.stack([jnp.asarray(im) for im in images])

        @jax.jit
        def step(params, opt_state, z_rng):
            loss, grads = jax.value_and_grad(_m_step_loss)(
                params, self.model, images, post_means, post_covs, z_rng, self.beta_recon)
            updates, opt_state = self._tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for _ in range(self.n_m_steps):
            self._rng, z_rng = jax.random.split(self._rng)
            self.params, self._opt_state, loss = step(self.params, self._opt_state, z_rng)
            self.loss_history.append(float(loss))

    def decode(self, z):
        return self.model.apply(
            self.params, jnp.asarray(z).reshape(1, -1), method=self.model.decode)[0].reshape(28, 28)


def _as_observation(obj) -> Observation:
    return obj if isinstance(obj, Observation) else Frozen(obj)


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

class KnownPhysics:
    """A slice whose dynamics are known (gray-box), re-linearized per step.

    ``linearize(mean) -> (A, B, offset)`` gives the local linear-Gaussian map
    s' = A·s + B·u + offset. Nothing is learned — this slice is only filtered.
    """
    learned = False

    def __init__(self, dim, linearize, du=0, process_std=1e-2):
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
        new_cov = A @ cov @ A.T + self.Q
        lam = jnp.linalg.inv(new_cov)
        return Gaussian(eta=lam @ mean, lam=lam)


class LearnedLinear:
    """A slice with unknown linear-Gaussian dynamics, learned by the CT node.

    x' ~ N(A·x + B·u, W⁻¹), with conjugate q(A), q(W), q(B) — exactly the
    system-identification `infer` does, exposed as a composable block.
    """
    learned = True

    def __init__(self, dim, du=0, n_iterations=50,
                 prior_a_cov=PRIOR_A_COV, prior_a_mean=None, init_a_cov=INIT_A_COV,
                 prior_b_cov=PRIOR_B_COV, init_b_cov=INIT_B_COV, prior_W_df=PRIOR_W_DF):
        self.dim = dim
        self.du = du
        self.n_iterations = n_iterations
        self.prior_a_cov, self.prior_a_mean, self.init_a_cov = prior_a_cov, prior_a_mean, init_a_cov
        self.prior_b_cov, self.init_b_cov, self.prior_W_df = prior_b_cov, init_b_cov, prior_W_df
        self.q_a = self.q_W = self.q_b = None

    @property
    def _meta(self):
        d = self.dim
        return CTMeta(lambda a: a.reshape(d, d))

    def learn(self, msg_seqs, ctrl_seqs=None):
        """One E-step: refine q(A), q(W), q(B) from observation-message sequences
        and return the concatenated smoothed states ``(means, covs)`` (for the
        M-step). Warm-starts from the current posteriors across EM iterations.

        ``msg_seqs``  : list of trajectories, each a list[Gaussian] of length T.
        ``ctrl_seqs`` : optional list of trajectories, each list[(du,)] len T-1.
        """
        d, du = self.dim, self.du
        meta = self._meta
        has_ctrl = ctrl_seqs is not None and du > 0
        prior_a = gaussian_prior(d * d, self.prior_a_cov, self.prior_a_mean)
        prior_W = Wishart(df=self.prior_W_df, inv_scale=jnp.eye(d))
        prior_b = gaussian_prior(d * du, self.prior_b_cov) if has_ctrl else None

        # Warm-start from current posteriors (EM), else from the priors.
        q_a = self.q_a if self.q_a is not None else gaussian_prior(d * d, self.init_a_cov, self.prior_a_mean)
        q_W = self.q_W if self.q_W is not None else prior_W
        q_b = self.q_b if self.q_b is not None else (gaussian_prior(d * du, self.init_b_cov) if has_ctrl else None)
        prior_x = Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))

        actions = (jnp.concatenate([jnp.stack([jnp.asarray(a) for a in c]) for c in ctrl_seqs])
                   if has_ctrl else None)

        def _bp(cache):
            mxs, mys, alphas_betas = [], [], []
            for i, seq in enumerate(msg_seqs):
                u = [jnp.asarray(a) for a in ctrl_seqs[i]] if has_ctrl else None
                al, be, m_x, m_y = forward_backward(prior_x, seq, cache, u)
                mxs.append(m_x); mys.append(m_y); alphas_betas.append((al, be, seq))
            return mxs, mys, alphas_betas

        for _ in range(self.n_iterations):
            cache = CTCache(q_a, q_W, meta, q_b)
            mxs, mys, _ = _bp(cache)
            q_a, q_W, q_b = accumulate_vmp_messages(
                _concat(mxs), _concat(mys), cache, actions, prior_a, prior_W, prior_b)
        self.q_a, self.q_W, self.q_b = q_a, q_W, q_b

        # Smoothed marginals per trajectory, concatenated (aligned with raw data).
        cache = CTCache(q_a, q_W, meta, q_b)
        _, _, ab = _bp(cache)
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


# ---------------------------------------------------------------------------
# Block + JointModel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Coupling — cross-block factor (the multimodal fusion seam)
# ---------------------------------------------------------------------------

class LinearCoupling:
    """A Gaussian factor  z = M·s + ε,  ε ~ N(0, P⁻¹)  linking two block slices.

    At each filter step we form the joint Gaussian over (s, z) under this
    factor and the two slices' incoming beliefs, then marginalise back out —
    information flows both ways. ``learnable=True`` (a learned M via conjugate
    regression on paired (s, z) data) is the documented extension hook.
    """

    def __init__(self, from_name: str, to_name: str, M, noise_prec, learnable=False):
        self.from_name = from_name
        self.to_name = to_name
        self.M = jnp.asarray(M)
        self.noise_prec = float(noise_prec)
        self.learnable = learnable

    def fuse(self, b_from: Gaussian, b_to: Gaussian):
        d_s = b_from.eta.shape[0]
        d_z = b_to.eta.shape[0]
        P = self.noise_prec * jnp.eye(d_z)
        MtP = self.M.T @ P
        Lam = jnp.block([
            [b_from.lam + MtP @ self.M, -MtP],
            [-MtP.T,                    b_to.lam + P],
        ])
        eta = jnp.concatenate([b_from.eta, b_to.eta])
        cov = jnp.linalg.inv(Lam)
        mu = cov @ eta
        cov_s, cov_z = cov[:d_s, :d_s], cov[d_s:, d_s:]
        lam_s, lam_z = jnp.linalg.inv(cov_s), jnp.linalg.inv(cov_z)
        return (Gaussian(eta=lam_s @ mu[:d_s], lam=lam_s),
                Gaussian(eta=lam_z @ mu[d_s:], lam=lam_z))


@dataclass
class Block:
    name: str
    transition: object              # KnownPhysics | LearnedLinear (Transition)
    observe: object                 # Callable | Observation (auto-wrapped)

    def __post_init__(self):
        self.observation = _as_observation(self.observe)

    @property
    def dim(self):
        return self.transition.dim


class JointModel:
    """Compose blocks into one joint-state model. ``coupling`` links slices
    (left for a later extension; ``None`` = independent chains)."""

    def __init__(self, blocks, coupling=None):
        self.blocks = list(blocks)
        # Accept a single coupling or a list (for multi-link fusion).
        self.couplings = ([] if coupling is None
                          else list(coupling) if isinstance(coupling, (list, tuple))
                          else [coupling])
        self._by_name = {b.name: b for b in self.blocks}

    def __getitem__(self, name) -> Block:
        return self._by_name[name]

    def learn(self, trajectories, n_em=1):
        """Variational EM over the composed blocks.

        Each trajectory is a dict ``{block_name: [raw obs per step]}`` (+ optional
        ``"control": [u per transition]``). Per EM iteration: an E-step refines
        each learned block's dynamics (yielding smoothed states), then an M-step
        refines each *learnable* observation against those states. With only
        frozen observations, ``n_em=1`` is plain system identification.
        """
        ctrl = ([t["control"] for t in trajectories]
                if trajectories and "control" in trajectories[0] else None)
        for _ in range(n_em):
            for b in self.blocks:
                if not b.transition.learned:
                    continue
                msg_seqs = [[b.observation.message(d) for d in traj[b.name]] for traj in trajectories]
                means, covs = b.transition.learn(msg_seqs, ctrl)
                if b.observation.learnable:
                    raw = [d for traj in trajectories for d in traj[b.name]]
                    b.observation.update(raw, means, covs)

    def _controllable(self):
        ctrl = [b for b in self.blocks if getattr(b.transition, "du", 0) > 0 and b.transition.learned]
        if not ctrl:
            raise ValueError("plan: no controllable learned block (transition.du > 0)")
        if len(ctrl) > 1:
            raise NotImplementedError("plan: multi-controllable planning is a coupling-layer concern")
        return ctrl[0]

    def plan(self, observations, n_iterations=50, prior_x=None):
        """Planning-as-inference on the composed model.

        ``observations`` is a dict ``{block_name: [obs or None per horizon step]}``
        — typically ``[start, None, ..., goal]`` for the controllable block.
        Returns the inferred action sequence (shape ``(n_ct, du)``).
        """
        b = self._controllable()
        tr = b.transition
        d, du = b.dim, tr.du
        meta = CTMeta(lambda a: a.reshape(d, d))
        cache = CTCache(tr.q_a, tr.q_W, meta, tr.q_b)

        raw = observations[b.name]
        msgs = [vague_gaussian(d) if o is None else b.observation.message(o) for o in raw]
        prior_x = prior_x if prior_x is not None else Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))

        # Action prior from the start→goal latent shift (same heuristic as inference.plan).
        mB = gaussian_mean(tr.q_b).reshape(d, du)
        b_col_var = jnp.sum(mB ** 2, axis=0) + 1e-8
        present = [i for i, o in enumerate(raw) if o is not None]
        if len(present) >= 2:
            mu_s = gaussian_mean(msgs[present[0]])
            mu_e = gaussian_mean(msgs[present[-1]])
            shift = jnp.maximum(jnp.sum((mu_e - mu_s) ** 2), 1e-8)
        else:
            shift = jnp.array(1.0)
        prior_u = Gaussian(eta=jnp.zeros(du), lam=jnp.diag(b_col_var / shift))

        return infer_actions(prior_x, msgs, cache, prior_u, n_iterations=n_iterations)

    def filter(self, stream):
        """Run an independent predict-update filter per block over the stream
        (list of {block_name: raw obs} (+ optional "control")). Returns the
        final belief per block."""
        beliefs = {b.name: None for b in self.blocks}
        for step in stream:
            u = step.get("control")
            for b in self.blocks:
                msg = b.observation.message(step[b.name])
                if beliefs[b.name] is None:
                    beliefs[b.name] = msg
                else:
                    beliefs[b.name] = combine_gaussians(b.transition.predict(beliefs[b.name], u), msg)
            # Cross-block fusion: apply each coupling factor to its slice pair.
            for c in self.couplings:
                beliefs[c.from_name], beliefs[c.to_name] = c.fuse(
                    beliefs[c.from_name], beliefs[c.to_name])
        return beliefs
