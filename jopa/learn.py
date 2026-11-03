"""Fitting a `NeuralLinear` transition from trajectories.

Two stages, matching the node's split between learned and inferred parts:

1. **Features** — φ (a two-layer GELU MLP plus a constant feature) and a
   point head W₀ are trained by gradient descent on ``horizon``-step
   open-loop rollouts, so φ is good for multi-step prediction, not just one
   step. These are the only gradients in JOPA.
2. **Last layer** — with φ frozen, W's prior is ``MN(W₀, (λI)⁻¹, σ²I)`` and
   every one-step transition multiplies in its exact conjugate message. σ² is
   the residual variance under the posterior mean.

A trajectory is ``(states, actions)``, ``states`` (T, d), ``actions``
(T, du), ``actions[t]`` driving ``states[t] → states[t+1]``.
"""
from __future__ import annotations

import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .distributions import MatrixNormal
from .nodes.neural_linear import NeuralLinear


class Features(nn.Module):
    """``[h(x), 1]``, h a two-layer GELU MLP; the 1 carries the bias."""
    width: int

    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.gelu(nn.Dense(self.width)(x), approximate=False)
        return jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)


def windows(trajectories, order, horizon):
    """Every ``(order + horizon)``-frame window and its ``horizon`` actions."""
    frames, acts = [], []
    for s, a in trajectories:
        for t in range(order - 1, len(s) - horizon):
            frames.append(s[t - order + 1:t + horizon + 1])
            acts.append(a[t:t + horizon])
    return jnp.asarray(np.stack(frames), jnp.float32), jnp.asarray(np.stack(acts), jnp.float32)


def transitions(trajectories, order):
    """One-step ``(history, u, x_next)`` over all trajectories, ``history``
    the ``order`` stacked state arrays, oldest first."""
    hist, us, nxt = [[] for _ in range(order)], [], []
    for s, a in trajectories:
        T = len(s)
        for j in range(order):
            hist[j].append(s[j:T - order + j])
        us.append(a[order - 1:T - 1])
        nxt.append(s[order:])
    def cat(xs):
        return jnp.asarray(np.concatenate(xs), jnp.float32)
    return tuple(cat(h) for h in hist), cat(us), cat(nxt)


def fit_neural_linear(trajectories, *, width=512, order=2, horizon=5, steps=20_000,
                      batch_size=256, lr=1e-3, weight_decay=1e-4, prior_precision=1e4,
                      seed=0, verbose=False) -> NeuralLinear:
    """Train φ on rollouts, then condition W on every transition (module docstring).
    ``lr`` is AdamW's peak rate, cosine-decayed to 0 over ``steps``."""
    frames, acts = windows(trajectories, order, horizon)
    d, du = frames.shape[-1], acts.shape[-1]
    net = Features(width)
    key, init_key = jax.random.split(jax.random.key(seed))
    params = {"phi": net.init(init_key, jnp.zeros((1, order * d + du))),
              "head": jnp.zeros((width + 1, d))}

    def loss(p, f, a):
        model = NeuralLinear(net.apply, p["phi"], None, 1.0, order, weight_mean=p["head"])
        return jnp.mean((model.rollout(tuple(f[:, j] for j in range(order)), a)
                         - f[:, order:]) ** 2)

    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=weight_decay)

    @jax.jit
    def train_step(p, s, k):
        idx = jax.random.randint(k, (batch_size,), 0, frames.shape[0])
        val, g = jax.value_and_grad(loss)(p, frames[idx], acts[idx])
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, val

    state = opt.init(params)
    for i in range(steps):
        key, k = jax.random.split(key)
        params, state, val = train_step(params, state, k)
        if verbose and i % 5000 == 0:
            print(f"step {i} rollout loss {float(val):.5f}", flush=True)

    data = transitions(trajectories, order)
    prior = MatrixNormal.from_mean(params["head"], prior_precision)
    model = NeuralLinear(net.apply, params["phi"], prior, 1.0, order).condition(*data)
    history, u, nxt = data
    noise_var = float(jnp.mean((nxt - model.mean(history, u)) ** 2))
    return NeuralLinear(net.apply, model.params, model.weights, noise_var, order)


def save_neural_linear(path, model: NeuralLinear, **extra):
    """Write the node's arrays, plus ``extra`` metadata, to ``path``."""
    state = {"params": jax.tree.map(np.asarray, model.params),
             "eta": np.asarray(model.weights.eta), "lam": np.asarray(model.weights.lam),
             "noise_var": float(model.noise_var), "order": model.order, **extra}
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_neural_linear(path):
    """``(model, extra)`` from `save_neural_linear`'s file."""
    with open(path, "rb") as f:
        state = pickle.load(f)
    params = jax.tree.map(jnp.asarray, state.pop("params"))
    width = params["params"]["Dense_0"]["kernel"].shape[1]
    weights = MatrixNormal(jnp.asarray(state.pop("eta")), jnp.asarray(state.pop("lam")))
    model = NeuralLinear(Features(width).apply, params, weights, state.pop("noise_var"),
                         state.pop("order"))
    return model, state
