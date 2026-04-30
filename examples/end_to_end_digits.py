"""Variational EM: jointly refine VAE + learn dynamics.

Starts from a pre-trained VAE, then alternates E-step (VMP+BP inference)
and M-step (VAE param updates against dynamics posterior) so the encoder
produces latent codes consistent with the learned linear dynamics.
"""
import os
import jax.numpy as jnp
import numpy as np

from jopa.nn.vae import VAE, train_vae, save_params, load_params, make_encode_decode
from jopa.data import load_mnist, rotate_image, rotating_mnist
from jopa.em import variational_em
from jopa.inference import infer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# ── 1. Pre-train or load VAE ────────────────────────────────────────────────
latent_dim = 4
vae_path = os.path.join(CHECKPOINTS, "vae_e2e_d4.npz")
model = VAE(latent_dim=latent_dim)

print("Preparing data …")
train_images, _ = rotating_mnist(n_digits=10, n_rotations=36, digits=(0, 1, 8))

try:
    params = load_params(model, vae_path)
    print(f"Loaded VAE from {vae_path}")
except FileNotFoundError:
    print("Pre-training VAE …")
    model, params = train_vae(train_images, latent_dim=latent_dim, epochs=100, seed=42)
    save_params(params, vae_path)
    print(f"Saved VAE to {vae_path}")

params_pretrained = params  # keep a copy for baseline

# ── 2. Build observation sequence ───────────────────────────────────────────
n_observed = 100
n_predicted = 100
step_deg = 360.0 / n_observed

all_imgs, all_labs = load_mnist()
digit_idx = np.where(all_labs == 8)[0][0]
base_img = all_imgs[digit_idx]

sequence = []
for i in range(n_observed):
    img = rotate_image(base_img, i * step_deg)
    img = (img > 0.5 * img.max()).astype(np.float32) if img.max() > 0 else img
    sequence.append(jnp.array(img))

# Ground truth future frames for comparison
gt_future = []
for i in range(n_observed, n_observed + n_predicted):
    img = rotate_image(base_img, i * step_deg)
    img = (img > 0.5 * img.max()).astype(np.float32) if img.max() > 0 else img
    gt_future.append(jnp.array(img))

# ── 3. Baseline: inference-only (frozen VAE) ────────────────────────────────
print("\n── Baseline (frozen VAE) ──")

encode_fn, decode_fn = make_encode_decode(model, params_pretrained)

baseline = infer(
    observations=sequence, encode_fn=encode_fn, decode_fn=decode_fn,
    latent_dim=latent_dim,
    n_predict=n_predicted, n_iterations=50,
)

H_base = baseline.transition_matrix
det_base = float(jnp.linalg.det(H_base))
eigs_base = np.linalg.eigvals(np.array(H_base))
print(f"  det(A)={det_base:.4f}  |λ|={np.abs(eigs_base)}")

# ── 4. Variational EM ──────────────────────────────────────────────────────
print("\n── Variational EM (40 EM iters) ──")
em_result = variational_em(
    model=model,
    params=params_pretrained,
    trajectories=[{"observations": sequence}],
    latent_dim=latent_dim,
    n_em_iterations=40,
    n_vmp_iterations=20,
    n_m_steps=30,
    lr=5e-5,
    beta_recon=1.0,
    seed=42,
)

H_e2e = em_result.transition_matrix
det_e2e = float(jnp.linalg.det(H_e2e))
eigs_e2e = np.linalg.eigvals(np.array(H_e2e))

print(f"\n── Final comparison ──")
print(f"  Baseline:       det(A)={det_base:.4f}  |λ|={np.abs(eigs_base)}")
print(f"  Variational EM: det(A)={det_e2e:.4f}  |λ|={np.abs(eigs_e2e)}")
print(f"  Expected step: {step_deg:.2f}°")

# ── 5. Predictions with refined VAE ────────────────────────────────────────
print("\nRunning prediction with refined VAE …")
encode_e2e, decode_e2e = make_encode_decode(model, em_result.params)

e2e_infer = infer(
    observations=sequence, encode_fn=encode_e2e, decode_fn=decode_e2e,
    latent_dim=latent_dim,
    n_predict=n_predicted, n_iterations=50, verbose=False,
)

# ── 6. Visualise ────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # --- Figure 1: Prediction comparison (4 rows) ---
    fig1, axes = plt.subplots(4, 10, figsize=(20, 8))

    for i, ax in enumerate(axes[0]):
        idx = i * (n_observed // 10)
        ax.imshow(np.array(sequence[idx]).reshape(28, 28), cmap="gray")
        ax.set_title(f"obs {idx}", fontsize=8)
        ax.axis("off")
    axes[0][0].set_ylabel("Observed", fontsize=10, rotation=0, labelpad=60, va="center")

    for i, ax in enumerate(axes[1]):
        idx = i * (n_predicted // 10)
        ax.imshow(np.array(gt_future[idx]).reshape(28, 28), cmap="gray")
        ax.set_title(f"gt +{idx}", fontsize=8, color="green")
        ax.axis("off")
    axes[1][0].set_ylabel("Ground truth", fontsize=10, rotation=0, labelpad=60, va="center")

    for i, ax in enumerate(axes[2]):
        idx = n_observed + i * (n_predicted // 10)
        ax.imshow(np.array(baseline.predictions[idx]).reshape(28, 28), cmap="inferno")
        ax.set_title(f"base +{i * (n_predicted // 10)}", fontsize=8, color="blue")
        ax.axis("off")
    axes[2][0].set_ylabel("Baseline pred", fontsize=10, rotation=0, labelpad=60, va="center")

    for i, ax in enumerate(axes[3]):
        idx = n_observed + i * (n_predicted // 10)
        ax.imshow(np.array(e2e_infer.predictions[idx]).reshape(28, 28), cmap="inferno")
        ax.set_title(f"e2e +{i * (n_predicted // 10)}", fontsize=8, color="red")
        ax.axis("off")
    axes[3][0].set_ylabel("VarEM pred", fontsize=10, rotation=0, labelpad=60, va="center")

    fig1.suptitle(
        f"Digit 8 (d={latent_dim}): baseline det={det_base:.3f} vs VarEM det={det_e2e:.3f}  |  "
        f"expected {step_deg:.1f}°/step",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "varem_predictions.png"), dpi=150)
    print(f"Saved {OUTPUTS}/varem_predictions.png")

    # --- Figure 2: Training diagnostics ---
    fig2 = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig2)

    # 2a: M-step loss over gradient steps
    ax_loss = fig2.add_subplot(gs[0, 0])
    ax_loss.plot(em_result.loss_history, "b-", linewidth=0.5)
    ax_loss.set(xlabel="gradient step", ylabel="loss", title="M-step loss")
    ax_loss.grid(True, alpha=0.3)

    # 2b: det(A) convergence
    ax_det = fig2.add_subplot(gs[0, 1])
    ax_det.plot(range(1, len(em_result.det_history) + 1), em_result.det_history, "r-o", markersize=3)
    ax_det.axhline(1.0, color="k", ls="--", alpha=0.3, label="ideal")
    ax_det.axhline(det_base, color="blue", ls=":", alpha=0.5, label=f"baseline ({det_base:.3f})")
    ax_det.set(xlabel="EM iteration", ylabel="det(A)", title="Transition matrix determinant")
    ax_det.legend(fontsize=8)
    ax_det.grid(True, alpha=0.3)

    # 2c: Eigenvalue moduli
    ax_eig = fig2.add_subplot(gs[0, 2])
    eig_mods = np.array([np.sort(np.abs(e))[::-1] for e in em_result.eigenvalues_history])
    for j in range(latent_dim):
        ax_eig.plot(range(1, len(eig_mods) + 1), eig_mods[:, j], "-o", markersize=3, label=f"|λ{j+1}|")
    ax_eig.axhline(1.0, color="k", ls="--", alpha=0.3)
    ax_eig.set(xlabel="EM iteration", ylabel="|λ|", title="Eigenvalue moduli of A")
    ax_eig.legend(fontsize=8)
    ax_eig.grid(True, alpha=0.3)

    # 2d: Latent trajectories — baseline
    ax_lat_base = fig2.add_subplot(gs[1, 0])
    means_b = np.array(baseline.latent_means)
    t = np.arange(means_b.shape[0])
    for dim in range(latent_dim):
        ax_lat_base.plot(t[:n_observed], means_b[:n_observed, dim], "-", linewidth=1, label=f"z[{dim}]")
        ax_lat_base.plot(t[n_observed:], means_b[n_observed:, dim], "--", linewidth=1, alpha=0.6)
    ax_lat_base.axvline(n_observed, color="k", ls="--", alpha=0.3)
    ax_lat_base.set(xlabel="time", ylabel="z", title="Latent trajectory (baseline)")
    ax_lat_base.legend(fontsize=7, ncol=2)
    ax_lat_base.grid(True, alpha=0.3)

    # 2e: Latent trajectories — VarEM
    ax_lat_e2e = fig2.add_subplot(gs[1, 1])
    means_e = np.array(e2e_infer.latent_means)
    for dim in range(latent_dim):
        ax_lat_e2e.plot(t[:n_observed], means_e[:n_observed, dim], "-", linewidth=1, label=f"z[{dim}]")
        ax_lat_e2e.plot(t[n_observed:], means_e[n_observed:, dim], "--", linewidth=1, alpha=0.6)
    ax_lat_e2e.axvline(n_observed, color="k", ls="--", alpha=0.3)
    ax_lat_e2e.set(xlabel="time", ylabel="z", title="Latent trajectory (VarEM)")
    ax_lat_e2e.legend(fontsize=7, ncol=2)
    ax_lat_e2e.grid(True, alpha=0.3)

    # 2f: Latent 2D phase portrait (first two dims)
    ax_phase = fig2.add_subplot(gs[1, 2])
    ax_phase.plot(means_b[:n_observed, 0], means_b[:n_observed, 1], "b-", alpha=0.5, label="base obs")
    ax_phase.plot(means_b[n_observed:, 0], means_b[n_observed:, 1], "b--", alpha=0.3, label="base pred")
    ax_phase.plot(means_e[:n_observed, 0], means_e[:n_observed, 1], "r-", alpha=0.5, label="VarEM obs")
    ax_phase.plot(means_e[n_observed:, 0], means_e[n_observed:, 1], "r--", alpha=0.3, label="VarEM pred")
    ax_phase.set(xlabel="z[0]", ylabel="z[1]", title="Phase portrait (z0 vs z1)")
    ax_phase.legend(fontsize=7)
    ax_phase.set_aspect("equal")
    ax_phase.grid(True, alpha=0.3)

    fig2.suptitle("Variational EM training diagnostics", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "varem_diagnostics.png"), dpi=150)
    print(f"Saved {OUTPUTS}/varem_diagnostics.png")

    # --- Figure 3: Transition matrices side by side ---
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    im1 = ax1.imshow(np.array(H_base), cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_title(f"Baseline A (det={det_base:.3f})")
    for i in range(latent_dim):
        for j in range(latent_dim):
            ax1.text(j, i, f"{float(H_base[i,j]):.2f}", ha="center", va="center", fontsize=9)
    im2 = ax2.imshow(np.array(H_e2e), cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_title(f"VarEM A (det={det_e2e:.3f})")
    for i in range(latent_dim):
        for j in range(latent_dim):
            ax2.text(j, i, f"{float(H_e2e[i,j]):.2f}", ha="center", va="center", fontsize=9)
    fig3.colorbar(im2, ax=[ax1, ax2], shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "varem_transition_matrices.png"), dpi=150)
    print(f"Saved {OUTPUTS}/varem_transition_matrices.png")

except ImportError:
    print("(install matplotlib for visualisation)")
