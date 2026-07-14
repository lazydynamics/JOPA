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

from .distributions import Gaussian, combine_gaussians, gaussian_mean
from .message_passing import infer_actions_exact
from .nodes.transition import CTCache, ct_forward
from .blocks import _identity_meta


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
    action_precision : prior precision on each action (Gaussian, zero mean).
    goal_precision : precision of the goal message — scalar or per-dimension
        vector (e.g. firm on position, soft on velocity: "arrive at rest" as
        a soft preference rather than a hard pin).
    u_clip : actuator saturation applied to the planned action (None: off).
    subgoal : optional `f(belief_mean, goal_mean) -> setpoint mean` hook —
        a trust region for far goals (domain adapters live in examples).
    goal_metric : optional `f(belief_mean, goal_mean) -> distance` deciding
        the reach/hold regime (default: Euclidean norm on the full latent —
        supply a domain metric, e.g. wrapped joint-angle distance, when some
        latent dimensions should not count toward "arrived").
    """

    def __init__(self, model, horizon=6, forget=0.5, window=10, relin_every=4,
                 action_precision=0.05, goal_precision=200.0,
                 u_clip=1.0, subgoal=None, goal_metric=None,
                 min_excitation=0.01, relin_surprise=0.02, relin_radius=0.1):
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
        self._surprise_ema = 0.0
        d, du = self._tr.dim, self._tr.du
        self._d, self._du = d, du
        self._meta = _identity_meta(d)
        self._prior_u = Gaussian(eta=jnp.zeros(du),
                                 lam=action_precision * jnp.eye(du))
        self._pg = jnp.diag(jnp.broadcast_to(
            jnp.asarray(goal_precision, dtype=jnp.float32), (d,)))
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

    # -- goals ---------------------------------------------------------------
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
        # filter: transition predict ⊗ observation update
        if self.belief is None or self._u_prev is None:
            self.belief = obs_msg
        else:
            pred = ct_forward(self.belief, self._cache,
                              self._augment_u(self._u_prev))
            self.surprise = float(jnp.linalg.norm(
                gaussian_mean(obs_msg) - gaussian_mean(pred)))
            self._surprise_ema = 0.7 * self._surprise_ema + 0.3 * self.surprise
            self.belief = combine_gaussians(pred, obs_msg)
        m = np.asarray(gaussian_mean(self.belief))
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
        far = self.goal_metric(m, np.asarray(self._goal_mean)) \
            > self.relin_radius
        due = (len(self._means) > self.window
               and self._t % self.relin_every == 0)
        if far and due and self._surprise_ema > self.relin_surprise:
            win = np.stack(self._means[-self.window:])
            if float(win.std(axis=0).max()) > self.min_excitation:
                self._tr.learn([self._msgs[-self.window:]],
                               [self._acts[-(self.window - 1):]])
                self._tr.remember(forget=self.forget)
                self._cache = CTCache(self._tr.q_a, self._tr.q_W, self._meta,
                                      self._tr.q_b)
        elif not far and due and self._tr.offset:
            # hold regime, drift-only adaptation: static closed-loop data
            # cannot identify (A, B) — but it identifies the affine drift c
            # perfectly (the DC component). Accumulate that evidence without
            # forgetting anything (forget=1), with a slow random walk on c
            # (diffuse) so the model's phantom equilibrium at the goal is
            # absorbed into the learned offset instead of being fought.
            self._tr.learn([self._msgs[-self.window:]],
                           [self._acts[-(self.window - 1):]])
            self._tr.remember(forget=1.0, diffuse=1e-3)
            self._cache = CTCache(self._tr.q_a, self._tr.q_W, self._meta,
                                  self._tr.q_b)
        # plan: exact action posterior toward the goal message. Two regimes,
        # both pure inference — they differ only in WHERE the goal evidence
        # sits on the chain:
        # * reach (far): goal at the horizon end, optionally trust-regioned by
        #   `subgoal` — intermediate slices stay free so motion is allowed;
        # * hold (near): goal pinned at EVERY slice — per-step deviation is
        #   penalised, so the plan leans on the accurate 1-step model instead
        #   of compounding H-step predictions (which noisy-sensor relearning
        #   cannot support).
        g = np.asarray(self._goal_mean)
        if far and self.subgoal is not None:
            g = np.asarray(self.subgoal(m, g))
        goal_msg = Gaussian(eta=self._pg @ jnp.asarray(g, dtype=jnp.float32),
                            lam=self._pg)
        if far:
            obs_chain = [self._vague] * self.horizon + [goal_msg]
        else:
            obs_chain = [goal_msg] * (self.horizon + 1)
        acts = infer_actions_exact(self.belief, obs_chain, self._cache,
                                   self._augment_prior_u())
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
        eta = jnp.concatenate([self._prior_u.eta, jnp.array([1e6])])
        lam = jnp.zeros((eff, eff)).at[:self._du, :self._du] \
                                   .set(self._prior_u.lam) \
                                   .at[self._du, self._du].set(1e6)
        return Gaussian(eta=eta, lam=lam)
