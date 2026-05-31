"""Convolutional VAE for 28×28 frames. Supports stacked-frame input via
`n_frames` — the encoder sees K consecutive frames as input channels and the
decoder reconstructs the whole K-frame window, so the latent must encode
motion (velocity), not just the latest pose. (`n_frames=1` is the plain
single-frame VAE.)"""
from __future__ import annotations
from typing import Callable, NamedTuple
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as serialization
import optax
import numpy as np
from tqdm import tqdm


LOG_STD_CLIP = (-6.0, 2.0)
PROB_CLIP = (1e-6, 1.0 - 1e-6)


def _as_batch(window, n_frames):
    """A single (28,28) frame or (K,28,28) window → a 1-element model batch."""
    w = jnp.asarray(window)
    return w.reshape(1, 28, 28) if n_frames == 1 else w.reshape(1, n_frames, 28, 28)


def _latest_frame(recon):
    """The most-recent 28×28 frame from a flat decoder output (… → 28×28)."""
    return jnp.asarray(recon).reshape(-1, 28, 28)[-1]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """Strided-conv encoder.  Input `(B, K, 28, 28)` (K = n_frames stacked as
    input channels) or `(B, 28, 28)` when K=1.  Output `(μ, log σ)` each `(B, d)`."""
    latent_dim: int = 2
    ch: int = 32
    n_frames: int = 1

    @nn.compact
    def __call__(self, x):
        c = self.ch
        # Accept (B, 28, 28) for n_frames=1 or (B, K, 28, 28) for K>=1.
        if x.ndim == 3:
            x = x[:, None, :, :]
        x = jnp.transpose(x, (0, 2, 3, 1))                # (B, 28, 28, K)
        x = nn.relu(nn.Conv(c,     (4, 4), strides=2, padding="SAME")(x))   # 14×14
        x = nn.relu(nn.Conv(c * 2, (4, 4), strides=2, padding="SAME")(x))   #  7×7
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))
        mu      = nn.Dense(self.latent_dim)(x)
        log_std = nn.Dense(self.latent_dim,
                           bias_init=nn.initializers.constant(-1.0))(x)
        return mu, log_std


class _Decoder(nn.Module):
    """Transposed-conv decoder  `(B, d) → (B, n_frames*784)` Bernoulli probs.
    `n_frames=1` (default) emits one 28×28 frame; `n_frames=K>1` reconstructs
    the whole K-frame window, which forces the latent to encode motion
    (velocity), not just the latest pose."""
    ch: int = 32
    n_frames: int = 1

    @nn.compact
    def __call__(self, z):
        c = self.ch
        x = nn.Dense(256)(z)
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(7 * 7 * c * 2)(x))
        x = x.reshape((-1, 7, 7, c * 2))
        x = nn.relu(nn.ConvTranspose(c, (4, 4), strides=(2, 2), padding="SAME")(x))     # 14×14
        x = nn.ConvTranspose(self.n_frames, (4, 4), strides=(2, 2), padding="SAME")(x)  # 28×28×K
        x = jnp.transpose(x, (0, 3, 1, 2))                                              # (B, K, 28, 28)
        return jax.nn.sigmoid(5.0 * x).reshape((z.shape[0], -1))                         # (B, K*784)


class VAE(nn.Module):
    """Convolutional VAE for 28×28 frames.

    `n_frames=1` (default): single-frame encoder + single-frame decoder.
    `n_frames=K>1`: encoder sees K stacked frames and the decoder reconstructs
    the whole K-frame window, so the latent must encode motion (velocity), not
    just the latest pose.
    """
    latent_dim: int = 2
    ch: int = 32
    n_frames: int = 1

    def setup(self):
        self.encoder = _Encoder(self.latent_dim, self.ch, self.n_frames)
        self.decoder = _Decoder(self.ch, self.n_frames)

    def __call__(self, x, z_rng):
        mu, log_std = self.encoder(x)
        z = mu + jnp.exp(log_std) * jax.random.normal(z_rng, mu.shape)
        return self.decoder(z), mu, log_std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Loss & training
# ---------------------------------------------------------------------------

def _elbo_loss(params, model, batch, target, z_rng, beta):
    """ELBO. `batch` is the encoder input (window or single frame); `target`
    is what the decoder must reconstruct — the full input (the whole K-frame
    window for multi-frame VAEs)."""
    recon, mu, log_std = model.apply(params, batch, z_rng)
    log_std = jnp.clip(log_std, *LOG_STD_CLIP)
    flat = target.reshape(target.shape[0], -1)
    p = jnp.clip(recon, *PROB_CLIP)
    bce = -jnp.sum(flat * jnp.log(p) + (1 - flat) * jnp.log(1 - p), axis=-1)
    kl = -0.5 * jnp.sum(1 + 2 * log_std - mu ** 2 - jnp.exp(2 * log_std), axis=-1)
    return jnp.mean(bce + beta * kl)


def train_vae(
    images: np.ndarray,
    *,
    latent_dim: int = 2,
    ch: int = 32,
    n_frames: int = 1,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
    callback: Callable | None = None,
) -> tuple[VAE, dict]:
    """Train the VAE (autoencoder ELBO — the decoder reconstructs its input).

    Single-frame: `images` of shape `(N, 28, 28)`.
    Multi-frame: `images` of shape `(N, K, 28, 28)` (the windows); the decoder
    reconstructs the whole K-frame window, so the latent must encode motion.
    `n_frames` must equal K.

    β-annealing from 0.1 → 1.0 over the first 15 epochs.
    """
    images = jnp.asarray(images)
    if images.ndim == 4:
        if images.shape[1] != n_frames:
            raise ValueError(
                f"4D input has {images.shape[1]} frames per window but n_frames={n_frames}; "
                f"expected (N, {n_frames}, 28, 28)")
        if n_frames == 1:
            images = images[:, 0]            # (N, 1, 28, 28) → (N, 28, 28)
    elif n_frames > 1:
        raise ValueError(
            f"n_frames={n_frames} expects a 4D (N, {n_frames}, 28, 28) input, got {images.shape}")
    targets_arr = images                     # autoencode: the decoder reconstructs its input

    model = VAE(latent_dim=latent_dim, ch=ch, n_frames=n_frames)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, z_rng = jax.random.split(rng, 3)
    init_input = (jnp.ones((1, 28, 28)) if n_frames == 1
                  else jnp.ones((1, n_frames, 28, 28)))
    params = model.init({"params": init_rng}, init_input, z_rng)

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, batch, target, z_rng, beta):
        loss, grads = jax.value_and_grad(_elbo_loss)(
            params, model, batch, target, z_rng, beta,
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = images.shape[0]
    pbar = tqdm(range(1, epochs + 1), desc="VAE", disable=not verbose)
    for epoch in pbar:
        rng, perm_rng = jax.random.split(rng)
        idx = jax.random.permutation(perm_rng, n)
        imgs = images[idx]
        targs = targets_arr[idx]
        beta = min(1.0, 0.1 + 0.9 * (epoch - 1) / 15)
        losses = []
        for i in range(0, n, batch_size):
            rng, z_rng = jax.random.split(rng)
            params, opt_state, loss = step(
                params, opt_state,
                imgs[i : i + batch_size], targs[i : i + batch_size],
                z_rng, beta,
            )
            losses.append(float(loss))
        pbar.set_postfix(loss=f"{np.mean(losses):.1f}", beta=f"{beta:.2f}")
        if callback is not None:
            callback(epoch, params, float(np.mean(losses)))
    return model, params


# ---------------------------------------------------------------------------
# Encode / decode helpers
# ---------------------------------------------------------------------------

class VAEAdapter(NamedTuple):
    """Jitted `encode`/`decode` closures bundled with the latent dimension."""
    encode: Callable
    decode: Callable
    latent_dim: int


def make_encode_decode(model, params) -> "VAEAdapter":
    K = getattr(model, "n_frames", 1)

    @jax.jit
    def encode_fn(window):
        mu, ls = model.apply(params, _as_batch(window, K), method=model.encode)
        ls = jnp.clip(ls, *LOG_STD_CLIP)
        return mu[0], ls[0]

    @jax.jit
    def decode_fn(z):
        return _latest_frame(model.apply(params, z.reshape(1, -1), method=model.decode))

    return VAEAdapter(encode=encode_fn, decode=decode_fn, latent_dim=model.latent_dim)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_params(params, path: str | Path):
    with open(str(path), "wb") as f:
        f.write(serialization.to_bytes(params))


def load_params(model: VAE, path: str | Path) -> dict:
    """Read flax msgpack; falls back to legacy `np.savez` (positional `p0..pN`)."""
    path = str(path)
    rng = jax.random.PRNGKey(0)
    K = getattr(model, "n_frames", 1)
    init_input = (jnp.ones((1, 28, 28)) if K == 1 else jnp.ones((1, K, 28, 28)))
    template = model.init({"params": rng}, init_input, rng)
    with open(path, "rb") as f:
        header = f.read(4)
    if header.startswith(b"PK"):
        data = np.load(path)
        leaves = jax.tree.leaves(template)
        flat = [jnp.array(data[f"p{i}"]) for i in range(len(leaves))]
        return jax.tree.unflatten(jax.tree.structure(template), flat)
    with open(path, "rb") as f:
        return serialization.from_bytes(template, f.read())
