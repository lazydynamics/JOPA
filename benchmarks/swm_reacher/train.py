"""Fit JOPA's neural-linear transition on LeWM's frozen latent.

Embeddings are centred and scaled to unit spread; the fit is
`jopa.fit_neural_linear` with its defaults. Prints held-out open-loop R² by
rollout step.

    python train.py --episodes runs/episodes.npz --out runs/jopa_nl.pkl
"""
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from encode import load

from jopa import fit_neural_linear, save_neural_linear
from jopa.learn import windows

HORIZON = 5


def r2(pred, true):
    return 1 - ((pred - true) ** 2).sum((0, 2)) / (true ** 2).sum((0, 2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    train, test = load(args.episodes)
    flat = np.concatenate([e for e, _ in train])
    centre = flat.mean(0)
    scale = float((flat - centre).std())
    def norm(eps):
        return [((e - centre) / scale, a) for e, a in eps]
    model = fit_neural_linear(norm(train), verbose=True)

    f, a = windows(norm(test), model.order, HORIZON)
    idx = np.random.default_rng(0).permutation(len(f))[:2000]
    f, a = f[idx], a[idx]
    true = f[:, 2:]
    def fmt(v):
        return " ".join(f"{float(x):.2f}" for x in v)
    print(f"held-out R² by k: persistence {fmt(r2(jnp.broadcast_to(f[:, 1:2], true.shape), true))}"
          f"   JOPA {fmt(r2(model.rollout((f[:, 0], f[:, 1]), a), true))}"
          f"   σ² {model.noise_var:.2e}", flush=True)
    save_neural_linear(args.out, model, centre=centre, scale=scale)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
