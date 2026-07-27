"""Closed-loop agent on one factor graph: sense → filter → plan → act,
learning continually.

Every step is inference on the same linear-Gaussian chain:

* **filter** — the transition message (`ct_forward`) predicts the belief, the
  observation message updates it (`combine_gaussians`);
* **adapt** — the transition posterior q(A, B, W) is re-estimated online by
  conjugate VMP (`learn`) with exponential forgetting (`remember`), so the
  model tracks the world it is acting in;
* **plan** — the action sequence is the exact Gaussian posterior given a goal
  message (`infer_actions_exact`, one α/β sweep), re-planned every step.

The relearning window is cleared on every `goal(...)` switch: evidence has a
regime, not just an age — data gathered holding the previous goal describes a
different operating point than the reach that follows.

Minimal loop::

    agent = Agent(model, horizon=6, forget=0.5)
    agent.goal(goal_observation)
    for obs in env:
        u = agent.step(obs)
"""
import jax.numpy as jnp
import numpy as np

from .graph import _identity_meta
from .config import (
    AGENT_ACTION_PRECISION,
    AGENT_FORGET,
    AGENT_GOAL_PRECISION,
    AGENT_HOLD_DRIFT_DIFFUSION,
    AGENT_HOLD_FORGET,
    AGENT_HORIZON,
    AGENT_INITIAL_GOAL_DISTANCE,
    AGENT_MIN_EXCITATION,
    AGENT_OFFSET_PIN_PRECISION,
    AGENT_PLAN_BELIEF_PRECISION,
    AGENT_PLAN_PRECISION_SCALE,
    AGENT_RELIN_EVERY,
    AGENT_RELIN_RADIUS,
    AGENT_RELIN_SURPRISE,
    AGENT_U_CLIP,
    AGENT_WINDOW,
    GOAL_PRECISION_PSD_TOLERANCE,
    GOAL_PRECISION_SYMMETRY_ATOL,
    GOAL_PRECISION_SYMMETRY_RTOL,
    SURPRISE_EMA_DECAY,
    SURPRISE_EMA_WEIGHT,
)
from .distributions import (
    Gaussian,
    Wishart,
    combine_gaussians,
    gaussian_mean,
    gaussian_mean_cov,
)
from .message_passing import (
    infer_actions_exact,
    infer_actions_exact_posterior,
)
from .nodes.transition import CTCache, ct_forward


class Agent:
    """Receding-horizon agent for a `JointModel` with one controllable block.

    Parameters
    ----------
    model : JointModel with a learned, controllable transition (warm-started
        via `model.learn`).
    horizon : planning horizon (steps).
    forget : exponential forgetting for the online conjugate updates;
        1.0 accumulates forever (static world), smaller tracks change.
    window, relin_every : online relearning uses the last `window` filtered
        states, refit every `relin_every` steps.
    action_precision : prior precision on each action (Gaussian, zero mean) —
        scalar, per-dimension vector, or a callable
        ``f(goal_distance) -> scalar | vector``.  The callable form is a
        conditional prior: expected effort may legitimately depend on how far
        the belief is from the goal, so one static precision need not trade
        travel against holding.  It is evaluated on the *latent* goal distance
        (``goal_metric``), so no physical units enter.
    goal_precision : precision of the goal message — scalar, per-dimension
        vector, or a full positive-semidefinite matrix.  A full matrix can,
        for example, carry the local image geometry induced by a decoder
        Jacobian while retaining correlations between pose coordinates.
    u_clip : actuator saturation applied to the planned action (None: off).
    subgoal : optional `f(belief_mean, goal_mean) -> setpoint mean` hook —
        a trust region for far goals (domain adapters live in examples).
    goal_metric : optional `f(belief_mean, goal_mean) -> distance` deciding
        whether online dynamics adaptation is far from the goal (default:
        Euclidean norm on the full latent).
    condition_on_estimate : plan against a point condition on E[x | y] rather
        than the filtered belief as a soft prior, and place goal evidence only
        on future states. Certainty equivalence; without it the goal factors
        re-infer the present state toward the goal and the commanded action
        shrinks. Off by default because it changes the closed-loop behaviour
        of existing models.
    plan_precision_scale : sharpen q(W) for planning only. The optimal gain is
        independent of process noise, but a finite transition precision lets
        the inferred plan satisfy the goal with noise instead of action.
        1.0 leaves the learned posterior untouched.
    goal_schedule : placement of goal evidence on the planning chain.
        ``"stage"`` places the same goal factor at every future
        state, the standard finite-horizon regulation objective. ``"terminal"``
        places it only at the horizon. ``"adaptive"`` retains the legacy
        distance-switched behavior and remains the backward-compatible default;
        learned pixel systems should choose explicitly because latent distance
        may not be a calibrated physical metric.
    """

    def __init__(self, model, horizon=AGENT_HORIZON, forget=AGENT_FORGET,
                 window=AGENT_WINDOW, relin_every=AGENT_RELIN_EVERY,
                 action_precision=AGENT_ACTION_PRECISION,
                 goal_precision=AGENT_GOAL_PRECISION,
                 u_clip=AGENT_U_CLIP, subgoal=None, goal_metric=None,
                 min_excitation=AGENT_MIN_EXCITATION,
                 relin_surprise=AGENT_RELIN_SURPRISE,
                 relin_radius=AGENT_RELIN_RADIUS,
                 adapt_hold=True, track_uncertainty=False,
                 goal_schedule="adaptive", condition_on_estimate=False,
                 plan_belief_precision=AGENT_PLAN_BELIEF_PRECISION,
                 plan_precision_scale=AGENT_PLAN_PRECISION_SCALE):
        self.model = model
        self._block = model._controllable()
        self._tr = self._block.transition
        if not self._tr.learned or self._tr.q_b is None:
            raise ValueError("Agent: the controllable transition must be "
                             "learned (with controls) before acting.")
        self.horizon = int(horizon)
        self.forget = float(forget)
        self.window = int(window)
        self.relin_every = int(relin_every)
        self.u_clip = u_clip
        self.subgoal = subgoal
        self.goal_metric = goal_metric or (
            lambda m, g: float(np.linalg.norm(m - g)))
        self.min_excitation = float(min_excitation)
        self.relin_surprise = float(relin_surprise)
        self.relin_radius = float(relin_radius)
        self.adapt_hold = bool(adapt_hold)
        self.track_uncertainty = bool(track_uncertainty)
        if goal_schedule not in ("stage", "terminal", "adaptive"):
            raise ValueError(
                "goal_schedule must be 'stage', 'terminal', or 'adaptive', "
                f"got {goal_schedule!r}")
        self.goal_schedule = goal_schedule
        self.condition_on_estimate = bool(condition_on_estimate)
        self.plan_belief_precision = float(plan_belief_precision)
        self.plan_precision_scale = float(plan_precision_scale)
        self._surprise_ema = 0.0
        d, du = self._tr.dim, self._tr.du
        self._d, self._du = d, du
        self._meta = _identity_meta(d)
        self._action_precision = action_precision
        self._prior_u = self._action_prior(
            action_precision(AGENT_INITIAL_GOAL_DISTANCE)
            if callable(action_precision) else action_precision)
        self.last_action_precision = None
        self._pg = self._goal_precision_matrix(goal_precision, d)
        self._vague = Gaussian(eta=jnp.zeros(d), lam=jnp.zeros((d, d)))
        self._cache = CTCache(self._tr.q_a, self._tr.q_W, self._meta,
                              self._tr.q_b)
        self.belief = None
        self.surprise = 0.0
        self._goal_mean = None
        self._u_prev = None
        self._means = []
        self._msgs = []
        self._acts = []
        self._t = 0
        self.last_observation_cov = None
        self.last_belief_cov = None
        self.last_action_cov = None
        # Full action posterior over the horizon from the last planning sweep
        # (requires track_uncertainty); u[0] is what gets executed.
        self.last_plan = None

    @property
    def du(self):
        """Number of physical control channels.

        `last_plan` is the posterior over the *augmented* control, so when the
        transition carries a drift (`offset=True`) it has one extra pinned
        channel held at 1; slice to `du` for the actions that are applied.
        """
        return self._du

    def _plan_prior(self) -> Gaussian:
        """The state condition used for planning.

        Certainty equivalence conditions the controller on E[x | y]. Passing the
        filtered belief as a *soft* prior instead leaves x[0] free, so the goal
        factors re-infer the present state toward the goal and the planner acts
        on a smaller error than the real one.
        """
        if not self.condition_on_estimate:
            return self.belief
        mean = jnp.asarray(gaussian_mean(self.belief))
        lam = self.plan_belief_precision * jnp.eye(self._d)
        return Gaussian(eta=lam @ mean, lam=lam)

    def _plan_cache(self) -> CTCache:
        """Transition precision used for planning only.

        The optimal gain does not depend on process noise, but a finite
        transition precision lets the inferred plan violate the dynamics and
        pay for the goal with process noise instead of action — the planner
        then under-actuates. Sharpening q(W) for planning restores the gain
        without touching the learned posterior used for filtering.
        """
        if self.plan_precision_scale == 1.0:
            return self._cache
        sharpened = Wishart(
            df=self._tr.q_W.df,
            inv_scale=self._tr.q_W.inv_scale / self.plan_precision_scale)
        return CTCache(self._tr.q_a, sharpened, self._meta, self._tr.q_b)

    def _action_prior(self, value) -> Gaussian:
        """Zero-mean action prior from a scalar/vector precision."""
        precision = np.asarray(value, dtype=np.float32)
        if precision.ndim == 0:
            lam = np.eye(self._du, dtype=np.float32) * float(precision)
        elif precision.shape == (self._du,):
            lam = np.diag(precision)
        else:
            raise ValueError(
                "action precision must be a scalar or a vector of length "
                f"{self._du}, got shape {precision.shape}")
        if not np.all(np.isfinite(lam)) or float(np.min(np.diag(lam))) <= 0.0:
            raise ValueError("action precision must be finite and positive")
        self.last_action_precision = lam
        return Gaussian(eta=jnp.zeros(self._du), lam=jnp.asarray(lam))

    @staticmethod
    def _goal_precision_matrix(goal_precision, dim):
        """Normalize scalar/vector/matrix goal precisions to a PSD matrix.

        Scalar and vector handling intentionally matches the historical API.
        Matrix validation happens once at agent construction, outside the
        inference loop, so exact planning receives the matrix unchanged.
        """
        value = np.asarray(goal_precision, dtype=np.float32)
        if value.ndim == 0:
            matrix = np.eye(dim, dtype=np.float32) * float(value)
        elif value.ndim == 1:
            if value.shape != (dim,):
                raise ValueError(
                    "goal_precision vector must have shape "
                    f"({dim},), got {value.shape}")
            matrix = np.diag(value)
        elif value.ndim == 2:
            if value.shape != (dim, dim):
                raise ValueError(
                    "goal_precision matrix must have shape "
                    f"({dim}, {dim}), got {value.shape}")
            if not np.allclose(value, value.T,
                               rtol=GOAL_PRECISION_SYMMETRY_RTOL,
                               atol=GOAL_PRECISION_SYMMETRY_ATOL):
                raise ValueError("goal_precision matrix must be symmetric")
            matrix = 0.5 * (value + value.T)
        else:
            raise ValueError(
                "goal_precision must be a scalar, vector, or matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("goal_precision must contain only finite values")
        tolerance = GOAL_PRECISION_PSD_TOLERANCE * max(
            1.0, float(np.max(np.abs(matrix))))
        if float(np.linalg.eigvalsh(matrix).min()) < -tolerance:
            raise ValueError("goal_precision matrix must be positive-semidefinite")
        return jnp.asarray(matrix, dtype=jnp.float32)

    # -- goals ---------------------------------------------------------------
    @property
    def transition(self):
        """The active learned transition, including its parameter posterior."""
        return self._tr

    def goal(self, observation):
        """Set the goal from a raw observation (encoded by the block's sensor).
        Clears the relearning window — a new goal is a new regime."""
        msg = self._block.observation.message(observation)
        self._goal_mean = gaussian_mean(msg)
        self._means, self._msgs, self._acts = [], [], []

    # -- the loop ------------------------------------------------------------
    def step(self, observation):
        """One closed-loop step: filter → adapt → plan → act."""
        if self._goal_mean is None:
            raise ValueError("Agent.step: set a goal first (agent.goal(...)).")
        obs_msg = self._block.observation.message(observation)
        if self.track_uncertainty:
            _, self.last_observation_cov = gaussian_mean_cov(obs_msg)
        # Replay neighborhoods live in raw observation-latent space. Refresh
        # before prediction so a stale model cannot pull the filtered belief
        # away and thereby choose its own (wrong) neighborhood.
        localize = getattr(self._tr, "localize", None)
        if localize is not None and localize(np.asarray(gaussian_mean(obs_msg))):
            self._cache = CTCache(self._tr.q_a, self._tr.q_W, self._meta,
                                  self._tr.q_b)
        # filter: transition predict ⊗ observation update
        if self.belief is None or self._u_prev is None:
            self.belief = obs_msg
        else:
            pred = ct_forward(self.belief, self._cache,
                              self._augment_u(self._u_prev))
            self.surprise = float(jnp.linalg.norm(
                gaussian_mean(obs_msg) - gaussian_mean(pred)))
            self._surprise_ema = (SURPRISE_EMA_DECAY * self._surprise_ema
                                  + SURPRISE_EMA_WEIGHT * self.surprise)
            self.belief = combine_gaussians(pred, obs_msg)
        m = np.asarray(gaussian_mean(self.belief))
        if self.track_uncertainty:
            _, self.last_belief_cov = gaussian_mean_cov(self.belief)
        # adapt: conjugate relearning from the raw OBSERVATION messages — the
        # E-step inside `learn` smooths noisy channels (e.g. encoder velocity)
        # through the dynamics with their true per-dim precision, instead of
        # trusting filtered point estimates. Relearning trades memory for
        # plasticity (`remember` discounts old evidence), so it only runs when
        # the model is WRONG and the data is INFORMATIVE:
        # (1) surprise — relearn only while the filter innovation exceeds the
        #     threshold (model disagreement; noisy sensors raise the floor);
        # (2) distance — near the setpoint the closed loop makes u collinear
        #     with x (closed-loop identifiability loss), so freeze the model
        #     while holding; a dynamics change re-engages both gates;
        # (3) excitation — a static window carries no evidence at all.
        self._means.append(m)
        self._msgs.append(obs_msg)
        goal_distance = self.goal_metric(m, np.asarray(self._goal_mean))
        far = goal_distance > self.relin_radius
        if callable(self._action_precision):
            self._prior_u = self._action_prior(
                self._action_precision(float(goal_distance)))
        due = (len(self._means) > self.window
               and self._t % self.relin_every == 0)
        if far and due and self._surprise_ema > self.relin_surprise:
            win = np.stack(self._means[-self.window:])
            if float(win.std(axis=0).max()) > self.min_excitation:
                self._tr.learn([self._msgs[-self.window:]],
                               [self._acts[-(self.window - 1):] if self.window > 1 else []])
                self._tr.remember(forget=self.forget)
                self._cache = CTCache(self._tr.q_a, self._tr.q_W, self._meta,
                                      self._tr.q_b)
        elif self.adapt_hold and not far and due and self._tr.offset:
            # hold regime, drift-only adaptation: static closed-loop data
            # cannot identify (A, B) — but it identifies the affine drift c
            # perfectly (the DC component). Accumulate that evidence without
            # forgetting anything (forget=1), with a slow random walk on c
            # (diffuse) so the model's phantom equilibrium at the goal is
            # absorbed into the learned offset instead of being fought.
            self._tr.learn([self._msgs[-self.window:]],
                           [self._acts[-(self.window - 1):] if self.window > 1 else []])
            self._tr.remember(forget=AGENT_HOLD_FORGET,
                              diffuse=AGENT_HOLD_DRIFT_DIFFUSION)
            self._cache = CTCache(self._tr.q_a, self._tr.q_W, self._meta,
                                  self._tr.q_b)
        # Plan by exact inference. Goal-factor placement is explicit and does
        # not silently depend on an uncalibrated learned-latent threshold.
        g = np.asarray(self._goal_mean)
        if far and self.subgoal is not None:
            g = np.asarray(self.subgoal(m, g))
        goal_msg = Gaussian(eta=self._pg @ jnp.asarray(g, dtype=jnp.float32),
                            lam=self._pg)
        terminal_only = (
            self.goal_schedule == "terminal"
            or (self.goal_schedule == "adaptive" and far))
        if terminal_only:
            obs_chain = [self._vague] * self.horizon + [goal_msg]
        elif self.condition_on_estimate:
            # Goal evidence on every FUTURE state. A goal factor on x[0] — the
            # present — competes with the state estimate, and once the belief
            # is not sharper than the goal it can reverse the sign of u[0].
            obs_chain = [self._vague] + [goal_msg] * self.horizon
        else:
            obs_chain = [goal_msg] * (self.horizon + 1)
        plan_prior = self._plan_prior()
        plan_cache = self._plan_cache()
        acts = infer_actions_exact(
            plan_prior, obs_chain, plan_cache,
            self._augment_prior_u())
        if self.track_uncertainty:
            # Actions deliberately come from the mean-only sweep above: the
            # dense posterior agrees to ~1e-12, but this loop amplifies, and
            # the published metrics were measured against this source. Folding
            # the two solves into one belongs with the next re-baseline.
            self.last_plan = infer_actions_exact_posterior(
                plan_prior, obs_chain, plan_cache,
                self._augment_prior_u())
            self.last_action_cov = self.last_plan.marginal_covariance[
                0, :self._du, :self._du]
        u = np.asarray(acts[0])[:self._du]
        if self.u_clip is not None:
            u = np.clip(u, -self.u_clip, self.u_clip)
        self._acts.append(jnp.asarray(u))
        self._u_prev = u
        self._t += 1
        return u

    # -- offset (pinned drift channel) support --------------------------------
    def _augment_u(self, u):
        if self._tr.offset:
            return jnp.concatenate([jnp.asarray(u), jnp.ones(1)])
        return jnp.asarray(u)

    def _augment_prior_u(self):
        if not self._tr.offset:
            return self._prior_u
        eff = self._tr.eff_du
        eta = jnp.concatenate(
            [self._prior_u.eta, jnp.array([AGENT_OFFSET_PIN_PRECISION])])
        lam = jnp.zeros((eff, eff)).at[:self._du, :self._du] \
                                   .set(self._prior_u.lam) \
                                   .at[self._du, self._du] \
                                   .set(AGENT_OFFSET_PIN_PRECISION)
        return Gaussian(eta=eta, lam=lam)
