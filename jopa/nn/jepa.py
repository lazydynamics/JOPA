"""Predictive (JEPA-style) encoder — the observation bootstrap for models
whose latent must obey *linear* dynamics.

Where the VAE learns a latent by reconstructing pixels (`jopa.nn.vae`), this
learns one by **predicting its own future in latent space**: encode two
consecutive frames, require the second to be a linear-Gaussian step from the
first, and regularize the latent's variance/covariance (VICReg) so it can't
collapse. The linear predictor `(A, B)` used here is *throwaway* — it only
shapes the representation; the deployed dynamics posterior q(A, B, W) is
learned afterwards by conjugate VMP (`LearnedLinear`).

This is the same *kind* of non-message-passing bootstrap the VAE always was —
a fixed sensor that turns an image into a Gaussian message — but its objective
matches what the message-passing core needs: a latent in which the dynamics
are linear. After pretraining the encoder is frozen; learning, filtering,
prediction, planning and online adaptation all remain variational message
passing.
"""
from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as serialization
import numpy as np
import optax
from tqdm import tqdm


class JEPAEncoder(nn.Module):
    """Deterministic conv encoder  `(B, [K,] S, S) → (B, d)`  (S = img_size,
    28|64). `n_frames=K>1` stacks K consecutive frames as input channels so the
    latent can encode motion (velocity) — required for second-order systems
    (torque-controlled arms), where a single-frame position latent is
    velocity-blind and control is unidentifiable. Mirrors the VAE encoder
    trunk so both bootstraps share an architecture."""
    latent_dim: int = 4
    ch: int = 32
    img_size: int = 28
    n_frames: int = 1

    @nn.compact
    def __call__(self, x):
        c = self.ch
        if x.ndim == 3:                                   # (B, S, S) → (B, 1, S, S)
            x = x[:, None, :, :]
        x = jnp.transpose(x, (0, 2, 3, 1)) * 2.0 - 1.0    # (B, S, S, K), centered
        x = nn.relu(nn.Conv(c,     (4, 4), strides=2, padding="SAME")(x))   # S/2
        x = nn.relu(nn.Conv(c * 2, (4, 4), strides=2, padding="SAME")(x))   # S/4
        if self.img_size == 64:
            x = nn.relu(nn.Conv(c * 2, (4, 4), strides=2, padding="SAME")(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))
        return nn.Dense(self.latent_dim)(x)


def _vicreg(z):
    """Variance (each dim std ≥ 1) + covariance (decorrelate) — anti-collapse."""
    std = jnp.sqrt(z.var(0) + 1e-4)
    var_loss = jnp.mean(jax.nn.relu(1.0 - std))
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    off = cov - jnp.diag(jnp.diag(cov))
    cov_loss = (off ** 2).sum() / z.shape[1]
    return var_loss, cov_loss


def train_jepa(
    frame_seqs,
    control_seqs=None,
    *,
    latent_dim: int = 4,
    ch: int = 32,
    img_size: int = 28,
    n_frames: int = 1,
    iterations: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    var_weight: float = 10.0,
    cov_weight: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[JEPAEncoder, dict]:
    """Pretrain the encoder by latent one-step prediction + VICReg.

    `frame_seqs`: list of `(T, S, S)` arrays (one per trajectory).
    `control_seqs`: optional matching list of `(T-1, du)` controls; when given,
    the throwaway predictor includes a `B·u` term so control-driven motion is
    not forced into the representation as state.

    `n_frames=K>1` encodes K-frame windows (stacked channels) so the latent
    captures velocity — needed for second-order (torque-controlled) systems.
    A prediction step then relates window ending at t to the window ending at
    t+1 under the control applied at t.

    Returns `(model, params)`. The internal predictor `(A, B, c)` is discarded;
    only the encoder params are kept — the deployed dynamics are learned by
    conjugate VMP downstream.
    """
    seqs = [np.asarray(s, dtype=np.float32) for s in frame_seqs]
    S, K = img_size, n_frames
    control_arrs = ([np.asarray(c, dtype=np.float32) for c in control_seqs]
                    if control_seqs is not None else None)
    du = control_arrs[0].shape[1] if control_arrs else 0

    # Build K-frame windows per trajectory and consecutive (window_t, window_{t+1})
    # pairs; the control on that transition is the one applied at the window's
    # last frame (controls[i + K - 1]).
    win_list, w0, w1, uu = [], [], [], []
    off = 0
    for k, s in enumerate(seqs):
        T = s.shape[0]
        if T < K + 1:
            continue
        wins = np.stack([s[t:t + K] for t in range(T - K + 1)])   # (T-K+1, K, S, S)
        nb = wins.shape[0]
        for i in range(nb - 1):
            w0.append(off + i); w1.append(off + i + 1)
            uu.append(control_arrs[k][i + K - 1] if du else np.zeros(0, np.float32))
        win_list.append(wins if K > 1 else wins[:, 0])
        off += nb
    windows = jnp.asarray(np.concatenate(win_list, axis=0))
    w0 = np.asarray(w0); w1 = np.asarray(w1)
    ctrl = jnp.asarray(np.stack(uu)) if du else None

    enc = JEPAEncoder(latent_dim=latent_dim, ch=ch, img_size=img_size, n_frames=K)
    rng = jax.random.PRNGKey(seed)
    rng, ir = jax.random.split(rng)
    init_in = jnp.ones((1, S, S)) if K == 1 else jnp.ones((1, K, S, S))
    params = {
        "enc": enc.init(ir, init_in),
        "A": jnp.eye(latent_dim) * 0.9,
        "B": jnp.zeros((latent_dim, du)) if du else jnp.zeros((latent_dim, 0)),
        "c": jnp.zeros(latent_dim),
    }
    tx = optax.adam(lr)
    opt = tx.init(params)

    def loss_fn(p, f0, f1, u):
        z0 = enc.apply(p["enc"], f0)
        z1 = enc.apply(p["enc"], f1)
        pred = z0 @ p["A"].T + p["c"]
        if du:
            pred = pred + u @ p["B"].T
        pred_loss = jnp.mean((z1 - pred) ** 2)
        v0, c0 = _vicreg(z0)
        v1, c1 = _vicreg(z1)
        return pred_loss + var_weight * (v0 + v1) + cov_weight * (c0 + c1)

    @jax.jit
    def step(p, opt, f0, f1, u):
        l, g = jax.value_and_grad(loss_fn)(p, f0, f1, u)
        upd, opt = tx.update(g, opt, p)
        return optax.apply_updates(p, upd), opt, l

    n = len(w0)
    pbar = tqdm(range(iterations), desc="JEPA", disable=not verbose)
    for it in pbar:
        rng, sr = jax.random.split(rng)
        sel = np.asarray(jax.random.choice(sr, n, (min(batch_size, n),), replace=False))
        f0, f1 = windows[w0[sel]], windows[w1[sel]]
        u = ctrl[sel] if du else jnp.zeros((len(sel), 0))
        params, opt, l = step(params, opt, f0, f1, u)
        if verbose:
            pbar.set_postfix(loss=f"{float(l):.4f}")
    return enc, params["enc"]


# ---------------------------------------------------------------------------
# Persistence (mirrors jopa.nn.vae)
# ---------------------------------------------------------------------------

def save_encoder(params, path: str | Path):
    with open(str(path), "wb") as f:
        f.write(serialization.to_bytes(params))


def load_encoder(model: JEPAEncoder, path: str | Path) -> dict:
    S, K = model.img_size, getattr(model, "n_frames", 1)
    init_in = jnp.ones((1, S, S)) if K == 1 else jnp.ones((1, K, S, S))
    template = model.init(jax.random.PRNGKey(0), init_in)
    with open(str(path), "rb") as f:
        return serialization.from_bytes(template, f.read())
