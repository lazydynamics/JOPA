"""Learning rotation dynamics of MNIST digits with VAE + Bayesian inference.

Two-stage pipeline: train a VAE, then use message passing (VMP + BP)
to learn a linear dynamical system in latent space and predict future frames.

Matches the RxInfer.jl example:
  https://examples.rxinfer.com/categories/advanced_examples/learning_dynamics_with_vaes/
"""
import os
import jax.numpy as jnp
import numpy as np

from jopa.inference import infer
from jopa.nn.vae import VAE, train_vae, save_params, load_params, make_encode_decode
from jopa.data import load_mnist, rotate_image, rotating_mnist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# ── 1. Data ────────────────────────────────────────────────────────────────
# Digits 0, 1, 8 — 10 exemplars each, 36 rotations (10°/step)
print("Creating rotated MNIST dataset …")
train_images, train_labels = rotating_mnist(
    n_digits=10, n_rotations=36, digits=(0, 1, 8),
)
print(f"  {train_images.shape[0]} training images (10°/step, binarised)")

# ── 2. VAE ─────────────────────────────────────────────────────────────────
vae_path = os.path.join(CHECKPOINTS, "vae_d4.npz")
model = VAE(latent_dim=4)

try:
    params = load_params(model, vae_path)
    print(f"Loaded VAE from {vae_path}")
except FileNotFoundError:
    print("Training VAE …")
    model, params = train_vae(train_images, latent_dim=4, epochs=100, seed=42)
    save_params(params, vae_path)
    print(f"Saved VAE to {vae_path}")

encode_fn, decode_fn = make_encode_decode(model, params)

# ── 3. Inference ───────────────────────────────────────────────────────────
# Build observation sequence: 100 rotated frames of one digit
n_observed = 100
n_predicted = 100
step_deg = 360.0 / n_observed  # 3.6°/step

all_imgs, all_labs = load_mnist()
# Use the 5th digit-8 exemplar
digit_idx = np.where(all_labs == 8)[0][4]
base_img = all_imgs[digit_idx]

sequence = []
for i in range(n_observed):
    img = rotate_image(base_img, i * step_deg)
    img = (img > 0.5 * img.max()).astype(np.float32) if img.max() > 0 else img
    sequence.append(jnp.array(img))

print(f"Running inference ({n_observed} observed + {n_predicted} predicted) …")
result = infer(
    observations=sequence,
    encode_fn=encode_fn,
    decode_fn=decode_fn,
    transform_fn=lambda a: a.reshape(4, 4),
    latent_dim=4,
    n_predict=n_predicted,
    n_iterations=50,
)

# ── 4. Results ─────────────────────────────────────────────────────────────
H = result.transition_matrix
angle = float(jnp.arctan2(H[1, 0], H[0, 0]) * 180 / jnp.pi)
det = float(jnp.linalg.det(H))

print(f"\nLearned transition matrix:\n{H}")
print(f"Rotation angle per step: {angle:.2f}°  (expected: {step_deg:.2f}°)")
print(f"Determinant: {det:.4f}  (ideal: 1.0)")

# ── 5. Visualise (optional) ────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i, ax in enumerate(axes[0]):
        idx = i * (n_observed // 10)
        ax.imshow(np.array(result.predictions[idx]).reshape(28, 28), cmap="gray")
        ax.set_title(f"obs {idx}", fontsize=8)
        ax.axis("off")
    for i, ax in enumerate(axes[1]):
        idx = n_observed + i * (n_predicted // 10)
        ax.imshow(np.array(result.predictions[idx]).reshape(28, 28), cmap="inferno")
        ax.set_title(f"pred +{idx - n_observed}", fontsize=8, color="red")
        ax.axis("off")
    fig.suptitle("Observations (top) vs Predictions (bottom)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "rotating_digits_result.png"), dpi=150)
    print(f"Saved {OUTPUTS}/rotating_digits_result.png")

    # Latent trajectory
    d = result.latent_means.shape[1]
    means = np.array(result.latent_means)
    stds = np.sqrt(np.diagonal(np.array(result.latent_covs), axis1=-2, axis2=-1))
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
