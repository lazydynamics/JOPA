"""Checkpoint I/O and jitted encode/decode adapters."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from ..config import (
    CHECKPOINT_HEADER_BYTES,
    DEFAULT_IMG_SIZE,
    DEFAULT_N_FRAMES,
    DEFAULT_SEED,
    LOG_STD_CLIP,
)
from .layers import _as_batch, _latest_frame


class VAEAdapter(NamedTuple):
    """Jitted `encode`/`decode` closures bundled with the latent dimension."""
    encode: Callable
    decode: Callable
    latent_dim: int


def make_encode_decode(model, params) -> VAEAdapter:
    K = getattr(model, "n_frames", DEFAULT_N_FRAMES)
    S = getattr(model, "img_size", DEFAULT_IMG_SIZE)

    @jax.jit
    def encode_fn(window):
        mu, ls = model.apply(params, _as_batch(window, K, S), method=model.encode)
        ls = jnp.clip(ls, *LOG_STD_CLIP)
        return mu[0], ls[0]

    @jax.jit
    def decode_fn(z):
        return _latest_frame(model.apply(params, z.reshape(1, -1), method=model.decode), S)

    return VAEAdapter(encode=encode_fn, decode=decode_fn, latent_dim=model.latent_dim)


def save_params(params, path: str | Path):
    with open(str(path), "wb") as f:
        f.write(serialization.to_bytes(params))


def _validate_checkpoint_tree(template, loaded, path):
    """Reject parameter trees whose structure or tensor shapes changed."""
    expected_structure = jax.tree_util.tree_structure(template)
    loaded_structure = jax.tree_util.tree_structure(loaded)
    if loaded_structure != expected_structure:
        raise ValueError(
            f"checkpoint {path} is incompatible with the model parameter tree")
    expected_leaves = jax.tree_util.tree_leaves(template)
    loaded_leaves = jax.tree_util.tree_leaves(loaded)
    for index, (expected, actual) in enumerate(
            zip(expected_leaves, loaded_leaves)):
        if np.shape(actual) != np.shape(expected):
            raise ValueError(
                f"checkpoint {path} is incompatible at parameter {index}: "
                f"expected shape {np.shape(expected)}, got {np.shape(actual)}")
    return loaded


def load_params(model, path: str | Path) -> dict:
    """Read flax msgpack; falls back to legacy np.savez positional arrays."""
    path = str(path)
    rng = jax.random.PRNGKey(DEFAULT_SEED)
    K = getattr(model, "n_frames", DEFAULT_N_FRAMES)
    S = getattr(model, "img_size", DEFAULT_IMG_SIZE)
    init_input = (jnp.ones((1, S, S)) if K == 1 else jnp.ones((1, K, S, S)))
    template = model.init({"params": rng}, init_input, rng)
    with open(path, "rb") as f:
        header = f.read(CHECKPOINT_HEADER_BYTES)
    if header.startswith(b"PK"):
        with np.load(path) as data:
            leaves = jax.tree.leaves(template)
            flat = [jnp.array(data[f"p{i}"]) for i in range(len(leaves))]
        loaded = jax.tree.unflatten(jax.tree.structure(template), flat)
        return _validate_checkpoint_tree(template, loaded, path)
    with open(path, "rb") as f:
        loaded = serialization.from_bytes(template, f.read())
    return _validate_checkpoint_tree(template, loaded, path)
