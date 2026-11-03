"""JOPA as a drop-in ``solver=`` for stable-worldmodel's ``eval_wm.py``.

The adapter encodes frames with LeWM's frozen encoder (torch) and converts
units; the rest is the library: each replan is inference on the reaching
graph (`jopa.Planner`), and with ``adapt=True`` the transitions seen since
the last replan first update the `jopa.Actuator` belief q(A) by the
ContinuousTransition rules (needs ``history_len = 3``).
"""
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # share the GPU with torch

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jopa import Actuator, Planner, load_neural_linear


class HarnessSolver:
    """The configure/property surface the harness's ``WorldModelPolicy`` expects."""

    def __init__(self, cost, device="cuda"):
        self.cost, self.device = cost, device

    def configure(self, *, action_space, n_envs, config):
        self._action_space, self._n_envs, self._config = action_space, n_envs, config
        self._action_dim = int(np.prod(action_space.shape[1:]))

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)


class JopaSolver(HarnessSolver):
    def __init__(self, cost, model_path, adapt=False, n_actuators=2, actuator_prior_var=1.0,
                 starts=8, iterations=20, action_precision=1e-2, act_sd=0.5774,
                 device="cuda", seed=0):
        super().__init__(cost, device)
        self.model, extra = load_neural_linear(model_path)
        self.centre, self.scale = torch.tensor(extra["centre"], device=device), extra["scale"]
        self.adapt = adapt
        self.actuator = Actuator.nominal(n_actuators, actuator_prior_var)
        self.planner_args = dict(starts=starts, iterations=iterations,
                                 action_precision=action_precision)
        self.n_actuators, self.act_sd = n_actuators, act_sd     # harness actions: raw / act_sd
        self.key = jax.random.key(seed)
        self._plan = self._update = None

    def _compile(self):
        planner, du = Planner(horizon=self.horizon, **self.planner_args), self.action_dim
        # one problem per episode; model and actuator are arguments, so one compile per run
        self._plan = jax.jit(jax.vmap(
            lambda m, act, prev, cur, goal, key: planner(m, (prev, cur), goal, key, du,
                                                         actuator=act),
            in_axes=(None, None, 0, 0, 0, 0)))
        self._update = jax.jit(lambda act, m, *data: act.update(m, *data))

    def _encode(self, px):
        with torch.no_grad():
            e = self.cost.model.encode({"pixels": px.to(self.device)})["emb"]
        return jnp.asarray(((e.float() - self.centre) / self.scale).cpu().numpy())

    def solve(self, info_dict, init_action=None):
        t0 = time.time()
        if self._plan is None:
            self._compile()
        e = self._encode(info_dict["pixels"])
        # The harness passes only unfinished episodes; padding to a fixed batch
        # (repeating the last row) keeps one compiled graph for the whole run.
        n = len(e)
        def pad(x):
            return jnp.concatenate([x, jnp.repeat(x[-1:], self.n_envs - n, 0)])
        if self.adapt and e.shape[1] >= 3 and "action_history" in info_dict:
            u = info_dict["action_history"][:, -1].double().cpu().numpy() * self.act_sd
            u = jnp.asarray(u, jnp.float32)
            weight = jnp.arange(self.n_envs) < n                 # padded rows send no message
            self.actuator = self._update(self.actuator, self.model,
                                         (pad(e[:, -3]), pad(e[:, -2])), pad(u), pad(e[:, -1]),
                                         weight)
            print(f"actuator posterior mean {np.round(self.actuator.mean, 3).tolist()} "
                  f"sd {np.round(self.actuator.sd, 4).tolist()}")
        goal = self._encode(info_dict["goal"])[:, -1]
        prev, cur = (e[:, -2], e[:, -1]) if e.shape[1] > 1 else (e[:, -1], e[:, -1])
        self.key, k = jax.random.split(self.key)
        act = self.actuator if self.adapt else Actuator.nominal(self.n_actuators)
        u, energy = self._plan(self.model, act, pad(prev), pad(cur), pad(goal),
                               jax.random.split(k, self.n_envs))
        print(f"JOPA solve time: {time.time() - t0:.4f} seconds")
        a = np.asarray(u, np.float64)[:n] / self.act_sd     # unit conversions in float64
        return {"actions": torch.tensor(a, dtype=torch.float32),
                "costs": np.asarray(energy)[:n].tolist()}
