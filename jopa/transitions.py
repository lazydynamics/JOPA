"""Transitions: how a state slice evolves.

`KnownPhysics` is a gray-box linearization; the `Learned*` classes are
conjugate Gaussian/Wishart posteriors over `(A, B, W)` fitted by the CT-factor
VMP updates in `jopa.message_passing`.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .config import (
    AFFINE_VMP_ITERATIONS,
    DEFAULT_DELAY,
    DELAY_INIT_A_COV,
    DELAY_INIT_B_COV,
    DELAY_LEARNED_A_COV,
    DELAY_LEARNED_B_COV,
    DELAY_PROCESS_STD,
    DELAY_REPLAY_OBS_PRECISION,
    DELAY_REPLAY_REFRESH_EVERY,
    DELAY_REPLAY_VMP_ITERATIONS,
    DELAY_SHIFT_COV,
    DELAY_SHIFT_PROCESS_STD,
    DELAY_VMP_ITERATIONS,
    DELAY_W_DF_FLOOR,
    INIT_A_COV,
    INIT_B_COV,
    KNOWN_PHYSICS_PROCESS_STD,
    LEARN_OBSERVED_PRECISION,
    OBSERVED_FIT_PRECISION,
    PRIOR_A_COV,
    PRIOR_B_COV,
    PRIOR_W_DF,
    REPLAY_NEIGHBORS,
    REPLAY_OBS_PRECISION,
    REPLAY_REFRESH_DISTANCE,
    REPLAY_REFRESH_EVERY,
    REPLAY_VMP_ITERATIONS,
    VMP_ITERATIONS,
    WHITENING_STD_FLOOR,
    WISHART_DF_MARGIN,
)
from .distributions import (
    Gaussian,
    Wishart,
    gaussian_mean,
    gaussian_mean_cov,
    gaussian_prior,
    wishart_mean,
)
from .message_passing import (
    accumulate_vmp_messages,
    compute_marginals,
    fit_vmp_grouped_controlled,
    forward_backward,
)
from .nodes.transition import CTCache, CTMeta, ct_forward


class KnownPhysics:
    """Gray-box dynamics: `linearize(belief_mean) -> (A, B, offset)` defines a
    local linear-Gaussian map  s' = A·s + B·u + offset + ε  re-evaluated per step.
    """
    learned = False

    def __init__(self, dim, linearize: Callable, du=0,
                 process_std=KNOWN_PHYSICS_PROCESS_STD):
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

    ``offset=True`` adds a learned constant drift `c` (wind, current, bias):
    a pinned-1 control channel is appended internally, so `c` is B's last
    column and gets the same conjugate treatment — posterior, credible
    intervals, planning — with no extra message rules.
    """
    learned = True

    def __init__(self, dim, du=0, n_iterations=VMP_ITERATIONS, mode="joint",
                 offset=False,
                 prior_a_cov=PRIOR_A_COV, prior_a_mean=None, init_a_cov=INIT_A_COV,
                 prior_b_cov=PRIOR_B_COV, init_b_cov=INIT_B_COV, prior_W_df=PRIOR_W_DF):
        if mode not in ("joint", "per_trajectory"):
            raise ValueError(f"mode must be 'joint' or 'per_trajectory', got {mode!r}")
        self.dim = dim
        self.du = du
        self.n_iterations = n_iterations
        self.mode = mode
        self.offset = offset
        self.prior_a_cov, self.prior_a_mean, self.init_a_cov = prior_a_cov, prior_a_mean, init_a_cov
        self.prior_b_cov, self.init_b_cov, self.prior_W_df = prior_b_cov, init_b_cov, prior_W_df
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
        q_b = self.q_b if self.q_b is not None else (gaussian_prior(d * du, self.init_b_cov) if has_ctrl else None)
        return q_a, q_W, q_b

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
        du = self.du
        has_ctrl = (ctrl_seqs is not None and du > 0) or self.offset
        prior_a, prior_W, prior_b = self._priors()
        q_a, q_W, q_b = self._init_q(has_ctrl)
        prior_x = Gaussian(eta=jnp.zeros(self.dim), lam=jnp.eye(self.dim))
        meta = self._meta

        def _ctrl_for(i):
            if not has_ctrl:
                return None
            us = (ctrl_seqs[i] if ctrl_seqs is not None and du > 0
                  else [None] * (len(msg_seqs[i]) - 1))
            return ([self._augment_u(a) for a in us] if self.offset
                    else [jnp.asarray(a) for a in us])

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
            actions = (jnp.concatenate([jnp.stack(_ctrl_for(i))
                                        for i in range(len(msg_seqs))]) if has_ctrl else None)
            if has_ctrl:
                groups = {}
                for i, seq in enumerate(msg_seqs):
                    groups.setdefault(len(seq), []).append(i)
                obs_groups, action_groups = [], []
                for indices in groups.values():
                    obs_groups.append(Gaussian(
                        eta=jnp.stack([
                            jnp.stack([g.eta for g in msg_seqs[i]])
                            for i in indices]),
                        lam=jnp.stack([
                            jnp.stack([g.lam for g in msg_seqs[i]])
                            for i in indices])))
                    action_groups.append(jnp.stack([
                        jnp.stack(_ctrl_for(i)) for i in indices]))
                q_a, q_W, q_b = fit_vmp_grouped_controlled(
                    prior_x, tuple(obs_groups), tuple(action_groups),
                    prior_a, prior_W, prior_b, q_a, q_W, q_b,
                    self.n_iterations)
            else:
                for _ in range(self.n_iterations):
                    cache = CTCache(q_a, q_W, meta, q_b)
                    mxs, mys, _ = _bp(cache)
                    q_a, q_W, q_b = accumulate_vmp_messages(
                        _stack(mxs), _stack(mys), cache, actions,
                        prior_a, prior_W, prior_b)
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
        if self.offset:
            u = self._augment_u(u)
        elif u is not None:
            u = jnp.asarray(u)
        return ct_forward(belief, cache, u)

    def attach_replay(self, state_seqs, ctrl_seqs, neighbors=REPLAY_NEIGHBORS,
                      refresh_every=REPLAY_REFRESH_EVERY,
                      obs_prec=REPLAY_OBS_PRECISION,
                      n_iterations=REPLAY_VMP_ITERATIONS,
                      refresh_distance=REPLAY_REFRESH_DISTANCE):
        """Attach encoded replay for local conjugate refits at the belief.

        Replay holds only sensor-encoded states and the agent's actions. At
        each ``localize`` call the nearest transitions in whitened state space
        are refit with the same Gaussian/Wishart VMP update — a local
        conjugate posterior, not a point-estimate regression. A single global
        linear fit averages configuration-dependent couplings over the whole
        workspace (where they can cancel); local refits keep them live.
        """
        if neighbors < 1 or refresh_every < 1:
            raise ValueError("neighbors and refresh_every must be positive")
        xs, ys, us = [], [], []
        for states, controls in zip(state_seqs, ctrl_seqs):
            states = np.asarray(states, dtype=np.float32)
            controls = np.asarray(controls, dtype=np.float32)
            if states.ndim != 2 or states.shape[1] != self.dim:
                raise ValueError(f"replay states must be (T,{self.dim})")
            if controls.shape != (len(states) - 1, self.du):
                raise ValueError(
                    "each replay sequence needs one control per transition")
            xs.append(states[:-1]); ys.append(states[1:]); us.append(controls)
        x = np.concatenate(xs); y = np.concatenate(ys); u = np.concatenate(us)
        scale = np.maximum(x.std(0), WHITENING_STD_FLOOR)
        self._local_replay = {
            "x": x, "y": y, "u": u, "scaled_x": x / scale, "scale": scale,
            "neighbors": min(int(neighbors), len(x)),
            "refresh_every": int(refresh_every), "obs_prec": float(obs_prec),
            "n_iterations": int(n_iterations),
            "refresh_distance": float(refresh_distance),
        }
        self._local_calls = 0
        self._local_query = None
        return self

    def localize(self, mean):
        """Refresh q(A,B,W) around ``mean`` from attached replay.

        ``refresh_distance`` defines the local model's validity region in
        whitened state space: while the query stays inside it, the current
        posterior remains valid and is NOT refit — which also lets online
        conjugate updates (e.g. drift-only hold adaptation) accumulate
        without being overwritten by an identical replay refit.
        """
        replay = getattr(self, "_local_replay", None)
        if replay is None:
            return False
        due = self._local_calls % replay["refresh_every"] == 0
        self._local_calls += 1
        if not due:
            return False
        mean = np.asarray(mean)
        last_query = getattr(self, "_local_query", None)
        if last_query is not None and replay["refresh_distance"] > 0.0:
            moved = float(np.linalg.norm(
                (mean - last_query) / replay["scale"]))
            if moved < replay["refresh_distance"]:
                return False
        self._local_query = mean.copy()
        distance = np.sum(
            (replay["scaled_x"] - mean / replay["scale"]) ** 2, axis=1)
        k = replay["neighbors"]
        selected = np.argpartition(distance, k - 1)[:k]
        # Each selected replay entry is an independent transition, so the
        # conjugate update is one batched rectangular fit. Routing the
        # neighbourhood through `learn_observed` as one two-row trajectory per
        # neighbour is numerically identical but costs a Python loop of tiny
        # device ops per neighbour.
        x = replay["x"][selected]
        y = replay["y"][selected]
        u = replay["u"][selected]
        if self.offset:
            u = np.concatenate(
                [u, np.ones((len(u), 1), dtype=u.dtype)], axis=1)
        # Fit deviations from the neighbourhood centroid. Parameterised around
        # the origin, a local fit must explain the offset of the whole
        # neighbourhood through A, which biases it and leaves a drift that
        # places the model's fixed point far outside the ball it was fitted in.
        reg = LearnedAffine(
            input_dim=self.dim, output_dim=self.dim, du=self.eff_du,
            n_iterations=replay["n_iterations"],
            prior_a_cov=self.prior_a_cov, prior_a_mean=self.prior_a_mean,
            prior_b_cov=self.prior_b_cov, prior_W_df=self.prior_W_df,
            obs_prec=replay["obs_prec"]).learn(
                [x], [y], [u] if self.eff_du else None)
        self.q_a, self.q_W, self.q_b = reg.q_a, reg.q_W, reg.q_b
        self._carried_a = self._carried_W = self._carried_b = None
        return True

    def learn_observed(self, state_seqs, ctrl_seqs=None,
                       obs_prec=LEARN_OBSERVED_PRECISION, n_iterations=None):
        """Conjugate fit from already encoded state/action trajectories.

        This is the fully observed special case of the same CT-factor VMP
        update. It is useful when a frozen pixel encoder has already produced
        Gaussian state means and avoids running redundant chain smoothing.
        """
        if ctrl_seqs is not None and len(state_seqs) != len(ctrl_seqs):
            raise ValueError("state_seqs and ctrl_seqs must have equal length")
        xs, ys, us = [], [], []
        for index, states in enumerate(state_seqs):
            states = jnp.asarray(states)
            if states.ndim != 2 or states.shape[1] != self.dim:
                raise ValueError(
                    f"sequence {index}: expected states (T,{self.dim}), "
                    f"got {states.shape}")
            xs.append(states[:-1])
            ys.append(states[1:])
            if self.du:
                if ctrl_seqs is None:
                    raise ValueError("controlled dynamics require ctrl_seqs")
                controls = jnp.asarray(ctrl_seqs[index])
                if controls.shape != (len(states) - 1, self.du):
                    raise ValueError(
                        f"sequence {index}: expected controls "
                        f"{(len(states) - 1, self.du)}, got {controls.shape}")
                us.append(jnp.concatenate(
                    [controls, jnp.ones((len(controls), 1))], axis=1)
                    if self.offset else controls)
            elif self.offset:
                us.append(jnp.ones((len(states) - 1, 1)))

        reg = LearnedAffine(
            input_dim=self.dim, output_dim=self.dim, du=self.eff_du,
            n_iterations=(self.n_iterations if n_iterations is None
                          else n_iterations),
            prior_a_cov=self.prior_a_cov,
            prior_a_mean=self.prior_a_mean,
            prior_b_cov=self.prior_b_cov,
            prior_W_df=self.prior_W_df,
            obs_prec=obs_prec).learn(xs, ys, us if self.eff_du else None)
        self.q_a, self.q_W, self.q_b = reg.q_a, reg.q_W, reg.q_b
        self._carried_a = self._carried_W = self._carried_b = None
        return self

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


class LearnedDelayLinear(LearnedLinear):
    """Conjugate controlled dynamics with a companion delay-state prior.

    For chronological ``z=[h[t-K+1], ..., h[t]]``, the first ``K-1`` output
    blocks are known history shifts. Only the final feature block needs to
    learn dynamics and control. Tight Gaussian priors preserve those shift
    rows while the ordinary CT-node VMP learns the final row, ``B``, affine
    drift and process precision. The resulting posterior has the same full
    ``q(A,B,W)`` interface used by filtering and exact planning.
    """

    def __init__(self, feature_dim, delay=DEFAULT_DELAY, du=0,
                 n_iterations=DELAY_VMP_ITERATIONS,
                 offset=True, learned_a_cov=DELAY_LEARNED_A_COV,
                 learned_b_cov=DELAY_LEARNED_B_COV,
                 init_a_cov=DELAY_INIT_A_COV, init_b_cov=DELAY_INIT_B_COV,
                 shift_cov=DELAY_SHIFT_COV,
                 process_std=DELAY_PROCESS_STD,
                 shift_process_std=DELAY_SHIFT_PROCESS_STD,
                 prior_W_df=None, mode="joint"):
        self.feature_dim = int(feature_dim)
        self.delay = int(delay)
        if self.feature_dim < 1 or self.delay < 2:
            raise ValueError("feature_dim must be positive and delay at least 2")
        self.learned_a_cov = float(learned_a_cov)
        self.learned_b_cov = float(learned_b_cov)
        self.shift_cov = float(shift_cov)
        self.process_std = float(process_std)
        self.shift_process_std = float(shift_process_std)
        dim = self.feature_dim * self.delay
        super().__init__(
            dim=dim, du=du, n_iterations=n_iterations, mode=mode, offset=offset,
            prior_a_cov=learned_a_cov, init_a_cov=init_a_cov,
            prior_b_cov=learned_b_cov, init_b_cov=init_b_cov,
            prior_W_df=(dim + WISHART_DF_MARGIN if prior_W_df is None
                        else prior_W_df))
        self._local_replay = None
        self._local_calls = 0

    @property
    def companion_mean(self):
        d, h = self.dim, self.feature_dim
        A = jnp.zeros((d, d))
        A = A.at[:d - h, h:].set(jnp.eye(d - h))
        # A neutral initial prediction copies the latest learned feature.
        return A.at[d - h:, d - h:].set(jnp.eye(h))

    @staticmethod
    def _diag_gaussian(mean, variance):
        mean, variance = jnp.asarray(mean).ravel(), jnp.asarray(variance).ravel()
        lam = jnp.diag(1.0 / variance)
        return Gaussian(eta=lam @ mean, lam=lam)

    def _a_gaussian(self, learned_cov):
        d, h = self.dim, self.feature_dim
        variance = jnp.full((d, d), self.shift_cov)
        variance = variance.at[d - h:, :].set(learned_cov)
        return self._diag_gaussian(self.companion_mean, variance)

    def _b_gaussian(self, learned_cov):
        d, h = self.dim, self.feature_dim
        variance = jnp.full((d, self.eff_du), self.shift_cov)
        variance = variance.at[d - h:, :].set(learned_cov)
        return self._diag_gaussian(jnp.zeros((d, self.eff_du)), variance)

    def _wishart_prior(self):
        d, h = self.dim, self.feature_dim
        variance = jnp.concatenate([
            jnp.full(d - h, self.shift_process_std ** 2),
            jnp.full(h, self.process_std ** 2),
        ])
        df = max(float(self.prior_W_df), float(d + WISHART_DF_MARGIN))
        return Wishart(df=df, inv_scale=df * jnp.diag(variance))

    def _base_priors(self):
        return (self._a_gaussian(self.learned_a_cov), self._wishart_prior(),
                self._b_gaussian(self.learned_b_cov) if self.eff_du else None)

    def _init_q(self, has_ctrl):
        q_a = self.q_a if self.q_a is not None else self._a_gaussian(self.init_a_cov)
        q_W = self.q_W if self.q_W is not None else self._wishart_prior()
        q_b = (self.q_b if self.q_b is not None else self._b_gaussian(self.init_b_cov)) \
            if has_ctrl else None
        return q_a, q_W, q_b

    def learn_observed(self, state_seqs, ctrl_seqs,
                       obs_prec=LEARN_OBSERVED_PRECISION, n_iterations=None):
        """Fast conjugate bootstrap from frozen-encoder delay vectors.

        Consecutive delay vectors share their first ``K-1`` blocks exactly, so
        fitting a full square transition through a chain is redundant and
        poorly conditioned. This update learns only the unknown newest-feature
        row with :class:`LearnedAffine`, then embeds that Gaussian/Wishart
        posterior into the full companion transition expected by filtering and
        exact planning. Controls and an optional pinned-one drift channel are
        learned jointly.
        """
        if len(state_seqs) != len(ctrl_seqs):
            raise ValueError("state_seqs and ctrl_seqs must have equal length")
        h, d = self.feature_dim, self.dim
        xs, ys, us = [], [], []
        for i, (states, controls) in enumerate(zip(state_seqs, ctrl_seqs)):
            states, controls = jnp.asarray(states), jnp.asarray(controls)
            if states.ndim != 2 or states.shape[1] != d:
                raise ValueError(f"sequence {i}: expected states (T,{d}), got {states.shape}")
            if controls.shape != (len(states) - 1, self.du):
                raise ValueError(
                    f"sequence {i}: expected controls {(len(states) - 1, self.du)}, "
                    f"got {controls.shape}")
            xs.append(states[:-1])
            ys.append(states[1:, -h:])
            us.append(jnp.concatenate([controls, jnp.ones((len(controls), 1))], axis=1)
                      if self.offset else controls)

        reg = LearnedAffine(
            input_dim=d, output_dim=h, du=self.eff_du,
            n_iterations=(self.n_iterations if n_iterations is None else n_iterations),
            prior_a_cov=self.learned_a_cov, prior_b_cov=self.learned_b_cov,
            prior_W_df=max(float(self.prior_W_df),
                           float(h + WISHART_DF_MARGIN)),
            obs_prec=obs_prec).learn(xs, ys, us)

        def embed(base, learned, n_learned):
            base_mean, _ = gaussian_mean_cov(base)
            learned_mean = gaussian_mean(learned)
            mean = base_mean.at[-n_learned:].set(learned_mean)
            # The embedded covariance is block diagonal: fixed shift rows and
            # the learned rectangular posterior. Construct its precision
            # directly to avoid a large dense inverse of mostly deterministic
            # parameters on every local refresh.
            lam = jnp.eye(mean.shape[0]) / self.shift_cov
            lam = lam.at[-n_learned:, -n_learned:].set(learned.lam)
            return Gaussian(eta=lam @ mean, lam=lam)

        base_a, _, base_b = self._base_priors()
        self.q_a = embed(base_a, reg.q_a, h * d)
        self.q_b = embed(base_b, reg.q_b, h * self.eff_du)

        top_cov = jnp.linalg.inv(wishart_mean(reg.q_W))
        full_cov = jnp.diag(jnp.full(d, self.shift_process_std ** 2))
        full_cov = full_cov.at[-h:, -h:].set(top_cov)
        df = max(float(d + WISHART_DF_MARGIN), DELAY_W_DF_FLOOR)
        self.q_W = Wishart(df=df, inv_scale=df * full_cov)
        self._carried_a = self._carried_W = self._carried_b = None
        return self

    def attach_replay(self, state_seqs, ctrl_seqs, neighbors=REPLAY_NEIGHBORS,
                      refresh_every=DELAY_REPLAY_REFRESH_EVERY,
                      obs_prec=DELAY_REPLAY_OBS_PRECISION,
                      n_iterations=DELAY_REPLAY_VMP_ITERATIONS):
        """Attach frozen-encoder replay for online local conjugate fits.

        Replay contains only delay vectors and the agent's actions. At refresh,
        nearest transitions are selected in whitened delay space and passed to
        :meth:`learn_observed`; selection is local, but parameter learning is
        still the same Gaussian/Wishart VMP update.
        """
        if neighbors < 1 or refresh_every < 1:
            raise ValueError("neighbors and refresh_every must be positive")
        xs, ys, us = [], [], []
        for states, controls in zip(state_seqs, ctrl_seqs):
            states, controls = jnp.asarray(states), jnp.asarray(controls)
            if controls.shape != (len(states) - 1, self.du):
                raise ValueError("each replay sequence needs one control per transition")
            xs.append(np.asarray(states[:-1])); ys.append(np.asarray(states[1:])); us.append(np.asarray(controls))
        x = np.concatenate(xs); y = np.concatenate(ys); u = np.concatenate(us)
        scale = np.maximum(x.std(0), WHITENING_STD_FLOOR)
        self._local_replay = {
            "x": x, "y": y, "u": u, "scaled_x": x / scale, "scale": scale,
            "neighbors": min(int(neighbors), len(x)),
            "refresh_every": int(refresh_every), "obs_prec": float(obs_prec),
            "n_iterations": int(n_iterations),
        }
        self._local_calls = 0
        return self

    def localize(self, mean):
        """Refresh q(A,B,W) around ``mean``; return whether it changed."""
        if self._local_replay is None:
            return False
        replay = self._local_replay
        due = self._local_calls % replay["refresh_every"] == 0
        self._local_calls += 1
        if not due:
            return False
        mean = np.asarray(mean)
        distance = np.sum((replay["scaled_x"] - mean / replay["scale"]) ** 2, axis=1)
        k = replay["neighbors"]
        selected = np.argpartition(distance, k - 1)[:k]
        pairs = [np.stack([replay["x"][i], replay["y"][i]]) for i in selected]
        controls = [replay["u"][i:i + 1] for i in selected]
        self.learn_observed(pairs, controls, obs_prec=replay["obs_prec"],
                            n_iterations=replay["n_iterations"])
        return True


class LearnedAffine:
    """Fully-observed Bayesian linear regression  y = A·x + B·u + ε  via the
    same CT-factor VMP — rectangular A `(output_dim, input_dim)`. Used when x
    and y are observed (no chain smoothing) — e.g. learning physics from
    proprioception with x = [sinθ, ω], y = ω', u = τ.
    """
    learned = True

    def __init__(self, input_dim, output_dim, du=0,
                 n_iterations=AFFINE_VMP_ITERATIONS,
                 prior_a_cov=PRIOR_A_COV, prior_a_mean=None,
                 prior_b_cov=PRIOR_B_COV,
                 prior_W_df=PRIOR_W_DF, obs_prec=OBSERVED_FIT_PRECISION):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.du = du
        self.n_iterations = n_iterations
        self.prior_a_cov = prior_a_cov
        self.prior_a_mean = prior_a_mean
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

        prior_a = gaussian_prior(
            dx * dy, self.prior_a_cov, self.prior_a_mean)
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
