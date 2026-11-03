"""Closed-loop agent on one factor graph: sense → filter → plan → act,
learning continually.

Every step is inference on the same linear-Gaussian chain:

* **filter** — the posterior predictive (`ct_predict`) forecasts the belief, the
  observation message updates it (`combine_gaussians`);
* **adapt** — the transition posterior q(A, B, W) is re-estimated online by
  conjugate VMP (`learn`) with exponential forgetting (`remember`), so the
  model tracks the world it is acting in;
* **plan** — the action sequence is the posterior mode of the reaching graph
  (`jopa.Planner`): the transition at its posterior mean, a goal on every
  future state, the actuator limits as a box. Re-planned every step.

The relearning window is cleared on every `goal(...)` switch: evidence has a
regime, not just an age — data gathered holding the previous goal describes a
different operating point than the reach that follows.

Minimal loop::

    agent = Agent(model, horizon=6, forget=0.5)
    agent.goal(goal_observation)
    for obs in env:
        u = agent.step(obs)
"""
import jax
import jax.numpy as jnp
import numpy as np

from .distributions import combine_gaussians, gaussian_mean, gaussian_mean_cov
from .planning import Planner


class Agent:
    """Receding-horizon agent for a `JointModel` with one controllable block.

    Planning is certainty-equivalent: the plan rolls the posterior-mean
    transition forward from E[x | y], and goal evidence sits on every future
    state. Conditioning on the filtered belief as a soft prior instead would
    let the goal factors re-infer the present state toward the goal, and the
    commanded action would shrink; a noisy transition would let the plan reach
    the goal with noise instead of action.

    Parameters
    ----------
    model : JointModel with a learned, controllable transition (warm-started
        via `model.learn`).
    horizon : planning horizon (steps).
    forget : exponential forgetting for the online conjugate updates;
        1.0 accumulates forever (static world), smaller tracks change.
    window, relin_every : online relearning uses the last `window`
        observations, refit every `relin_every` steps.
    action_precision : prior precision on each action (zero mean) — scalar or
        per-dimension vector.
    goal_precision : precision of the goal message — scalar or per-dimension
        vector.
    u_clip : actuator limit, a box on every planned action (None: off).
    min_excitation, relin_surprise, relin_radius : relearning runs only when
        the data is informative (the window's spread exceeds
        ``min_excitation``), the model is wrong (smoothed filter innovation
        above ``relin_surprise``) and the belief is away from the goal (beyond
        ``relin_radius``, where closed-loop data stays identifiable).
    track_uncertainty : record ``last_observation_cov``, ``last_belief_cov``
        and ``last_action_cov`` (the Laplace covariance of the first action)
        every step.
    """

    def __init__(self, model, horizon=6, forget=0.5, window=10, relin_every=4,
                 action_precision=0.05, goal_precision=200.0, u_clip=1.0,
                 min_excitation=0.01, relin_surprise=0.02, relin_radius=0.1,
                 track_uncertainty=False):
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
        self.min_excitation = float(min_excitation)
        self.relin_surprise = float(relin_surprise)
        self.relin_radius = float(relin_radius)
        self.track_uncertainty = bool(track_uncertainty)
        self._d, self._du = self._tr.dim, self._tr.du
        self._planner = Planner(
            horizon=self.horizon, starts=1,
            action_precision=self._precision(action_precision, self._du),
            goal_precision=self._precision(goal_precision, self._d),
            limit=None if u_clip is None else float(u_clip))
        self._plan = jax.jit(self._first_action)
        self.belief = None
        self.surprise = 0.0
        self._surprise_ema = 0.0
        self._goal_mean = None
        self._u_prev = None
        self._means, self._msgs, self._acts = [], [], []
        self._absorbed = 0          # transitions already in q(A, B, W)
        self._t = 0
        self.last_observation_cov = None
        self.last_belief_cov = None
        self.last_action_cov = None

    @property
    def du(self):
        """Number of physical control channels."""
        return self._du

    @staticmethod
    def _precision(value, dim):
        """A scalar or per-dimension precision, as a per-dimension vector."""
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 0:
            value = np.full(dim, float(value), dtype=np.float32)
        if value.shape != (dim,):
            raise ValueError(f"precision must be a scalar or have shape ({dim},), "
                             f"got {value.shape}")
        if not np.all(np.isfinite(value)) or float(value.min()) < 0.0:
            raise ValueError("precision must be finite and non-negative")
        return jnp.asarray(value)

    def _first_action(self, model, x, goal):
        """Mean and covariance of the first planned action."""
        graph = self._planner.graph(model, (x,), goal, self._du)
        post = graph.infer({f"u{k}": jnp.zeros(self._du) for k in range(self.horizon)},
                           self._planner.iterations)
        return post.mean["u0"], post.cov["u0"]

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
        self._absorbed = 0

    # -- the loop ------------------------------------------------------------
    def step(self, observation):
        """One closed-loop step: filter → adapt → plan → act."""
        if self._goal_mean is None:
            raise ValueError("Agent.step: set a goal first (agent.goal(...)).")
        obs_msg = self._block.observation.message(observation)
        if self.track_uncertainty:
            _, self.last_observation_cov = gaussian_mean_cov(obs_msg)
        # filter: transition predict ⊗ observation update
        if self.belief is None or self._u_prev is None:
            self.belief = obs_msg
        else:
            pred = self._tr.predict(self.belief, self._u_prev)
            self.surprise = float(jnp.linalg.norm(gaussian_mean(obs_msg) - gaussian_mean(pred)))
            self._surprise_ema = 0.7 * self._surprise_ema + 0.3 * self.surprise
            self.belief = combine_gaussians(pred, obs_msg)
        m = np.asarray(gaussian_mean(self.belief))
        if self.track_uncertainty:
            _, self.last_belief_cov = gaussian_mean_cov(self.belief)
        # adapt: conjugate relearning from the raw observation messages, gated
        # so it runs only when it can help (see the class docstring).
        self._means.append(m)
        self._msgs.append(obs_msg)
        far = float(np.linalg.norm(m - np.asarray(self._goal_mean))) > self.relin_radius
        due = len(self._means) > self.window and self._t % self.relin_every == 0
        if far and due and self._surprise_ema > self.relin_surprise:
            win = np.stack(self._means[-self.window:])
            if float(win.std(axis=0).max()) > self.min_excitation:
                self._absorb()
        # plan: from E[x | y], with the goal on every future state
        u, cov = self._plan(self._tr.dynamics, jnp.asarray(m), jnp.asarray(self._goal_mean))
        if self.track_uncertainty:
            self.last_action_cov = cov
        u = np.asarray(u)
        self._acts.append(jnp.asarray(u))
        self._u_prev = u
        self._t += 1
        return u

    def _absorb(self):
        """Conjugate online update: discount q(A, B, W) by ``forget``, then
        condition it on the transitions not yet absorbed (within ``window``).

        The current posterior is the prior, so offline evidence is kept, and
        each transition enters the posterior exactly once."""
        start = max(self._absorbed, len(self._msgs) - self.window)
        if len(self._msgs) - start < 2:
            return
        self._tr.remember(forget=self.forget)
        self._tr.learn([self._msgs[start:]], [self._acts[start:]])
        self._absorbed = len(self._msgs) - 1
