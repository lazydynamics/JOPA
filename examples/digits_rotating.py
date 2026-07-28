"""Learning rotation dynamics of MNIST digits with VAE + Bayesian inference.

Two-stage pipeline: train a VAE, then use message passing (VMP + BP)
to learn a linear dynamical system in latent space and predict future frames.

Matches the RxInfer.jl example:
  https://examples.rxinfer.com/categories/advanced_examples/learning_dynamics_with_vaes/
"""
import argparse
import os

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--smoke", action="store_true",
                help="Tiny end-to-end configuration, for CI.")
_p.add_argument("--checkpoint-dir", default=None)
_p.add_argument("--output-dir", default=None)
args = _p.parse_args()
SMOKE = args.smoke

import jax.numpy as jnp
import numpy as np

from examples.digits_data import load_mnist, rotating_mnist, rotation_sequence
from jopa.blocks import Block, Frozen, JointModel, LearnedLinear
from jopa.distributions import Gaussian
from jopa.nn.vae import VAE, load_params, make_encode_decode, save_params, train_vae

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = args.checkpoint_dir or os.path.join(ROOT, "checkpoints")
OUTPUTS = args.output_dir or os.path.join(ROOT, "outputs")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

print("Creating rotated MNIST dataset …")
train_images, train_labels = rotating_mnist(
    n_digits=1 if SMOKE else 10,
    n_rotations=4 if SMOKE else 36, digits=(0, 1, 8),
)
print(f"  {train_images.shape[0]} training images (10°/step, binarised)")

vae_path = os.path.join(CHECKPOINTS, "vae_d4.npz")
model = VAE(latent_dim=4, ch=4 if SMOKE else 32)

try:
    params = load_params(model, vae_path)
    print(f"Loaded VAE from {vae_path}")
except FileNotFoundError:
    print("Training VAE …")
    model, params = train_vae(train_images, latent_dim=4, ch=4 if SMOKE else 32,
                              epochs=1 if SMOKE else 100, seed=42)
    save_params(params, vae_path)
    print(f"Saved VAE to {vae_path}")

vae = make_encode_decode(model, params)

n_observed = 6 if SMOKE else 100
n_predicted = 2 if SMOKE else 100
step_deg = 360.0 / n_observed

all_imgs, all_labs = load_mnist()
digit_idx = np.where(all_labs == 8)[0][4]
base_img = all_imgs[digit_idx]

sequence = [jnp.array(f) for f in rotation_sequence(base_img, n_observed, step_deg=step_deg)]

def _vae_msg(image):
    mu, log_std = vae.encode(image)
    lam = jnp.diag(1.0 / jnp.exp(2.0 * log_std))
    return Gaussian(eta=lam @ mu, lam=lam)

print(f"Running inference ({n_observed} observed + {n_predicted} predicted) …")
block = Block("z", LearnedLinear(dim=4, n_iterations=50),
              observe=Frozen(_vae_msg, vae.decode))
model = JointModel([block])
model.learn([{"z": sequence}])
out = model.smooth({"z": sequence}, n_predict=n_predicted)["z"]

H = block.transition.A
angle = float(jnp.arctan2(H[1, 0], H[0, 0]) * 180 / jnp.pi)
det = float(jnp.linalg.det(H))

print(f"\nLearned transition matrix:\n{H}")
print(f"Rotation angle per step: {angle:.2f}°  (expected: {step_deg:.2f}°)")
print(f"Determinant: {det:.4f}  (ideal: 1.0)")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i, ax in enumerate(axes[0]):
        idx = i * (n_observed // 10)
        ax.imshow(np.array(out["predictions"][idx]).reshape(28, 28), cmap="gray")
        ax.set_title(f"obs {idx}", fontsize=8)
        ax.axis("off")
    for i, ax in enumerate(axes[1]):
        idx = n_observed + i * (n_predicted // 10)
        ax.imshow(np.array(out["predictions"][idx]).reshape(28, 28), cmap="inferno")
        ax.set_title(f"pred +{idx - n_observed}", fontsize=8, color="red")
        ax.axis("off")
    fig.suptitle("Observations (top) vs Predictions (bottom)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "rotating_digits_result.png"), dpi=150)
    print(f"Saved {OUTPUTS}/rotating_digits_result.png")

    means = np.array(out["means"])
    stds = np.sqrt(np.diagonal(np.array(out["covs"]), axis1=-2, axis2=-1))
    d = means.shape[1]
    t = np.arange(means.shape[0])

    fig2, axes2 = plt.subplots(1, d, figsize=(4 * d, 4))
    if d == 1:
        axes2 = [axes2]
    for dim, ax in enumerate(axes2):
        ax.plot(t[:n_observed], means[:n_observed, dim], "g-", label="posterior (obs)")
        ax.fill_between(t[:n_observed],
                        means[:n_observed, dim] - 2 * stds[:n_observed, dim],
                        means[:n_observed, dim] + 2 * stds[:n_observed, dim],
                        alpha=0.15, color="g")
        ax.plot(t[n_observed:], means[n_observed:, dim], "r-", label="prediction")
        ax.fill_between(t[n_observed:],
                        means[n_observed:, dim] - 2 * stds[n_observed:, dim],
                        means[n_observed:, dim] + 2 * stds[n_observed:, dim],
                        alpha=0.15, color="r")
        ax.axvline(n_observed, color="k", ls="--", alpha=0.3)
        ax.set(xlabel="time step", ylabel=f"z[{dim}]")
        ax.legend(fontsize=8)
    fig2.suptitle("Latent trajectory with uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "latent_trajectory.png"), dpi=150)
    print(f"Saved {OUTPUTS}/latent_trajectory.png")

except ImportError:
    print("(install matplotlib for visualisation)")
