"""System identification: learn dynamics A and control matrix B from images + actions.

A digit rotates with varying angular velocity driven by control actions.
The model learns A (autonomous dynamics, prior ~ I) and B (control effect),
then predicts different futures depending on the action applied.
"""
import os
import jax.numpy as jnp
import numpy as np

from jopa.nn.vae import VAE, train_vae, save_params, load_params, make_encode_decode
from jopa.data import load_mnist, rotating_mnist, make_controlled_sequence
from jopa.inference import infer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# ── 1. VAE ─────────────────────────────────────────────────────────────────
latent_dim = 4
vae_path = os.path.join(CHECKPOINTS, "vae_ctrl_d4.npz")
model = VAE(latent_dim=latent_dim)

print("Preparing data …")
train_images, _ = rotating_mnist(n_digits=10, n_rotations=36, digits=(0, 1, 8))

try:
    params = load_params(model, vae_path)
    print(f"Loaded VAE from {vae_path}")
except FileNotFoundError:
    print("Training VAE …")
    model, params = train_vae(train_images, latent_dim=latent_dim, epochs=100, seed=42)
    save_params(params, vae_path)
    print(f"Saved VAE to {vae_path}")

encode_fn, decode_fn = make_encode_decode(model, params)

# ── 2. Generate controlled sequence ────────────────────────────────────────
all_imgs, all_labs = load_mnist()
digit_idx = np.where(all_labs == 8)[0][0]

# Scale actions so u=1 ≈ one latent-unit displacement.
# Empirically ~60° rotation ≈ 1 latent unit, so action_scale=60.
ACTION_SCALE = 60.0

print("Generating controlled rotation sequence …")
frames, actions, angles = make_controlled_sequence(
    digit_idx=digit_idx, n_frames=120, seed=42,
    action_scale=ACTION_SCALE,
)
n_observed = len(frames)

sequence = [jnp.array(f) for f in frames]
jax_actions = [jnp.array(a) for a in actions]

act_values = [float(a[0]) for a in actions]
print(f"  {n_observed} frames, action range: [{min(act_values):.2f}, {max(act_values):.2f}]")
print(f"  angle range: [{angles.min():.0f}°, {angles.max():.0f}°]")

# ── 3. Learn A and B ──────────────────────────────────────────────────────
print("\n── Learning A and B (prior: A ~ I) ──")

vec_I = jnp.eye(latent_dim).ravel()
n_predict = 80

result = infer(
    observations=sequence,
    encode_fn=encode_fn,
    decode_fn=decode_fn,
    transform_fn=lambda a: a.reshape(latent_dim, latent_dim),
    latent_dim=latent_dim,
    actions=jax_actions,
    action_dim=1,
    n_predict=n_predict,
    predict_actions=[jnp.zeros(1)] * n_predict,
    n_iterations=100,
    prior_a_mean=vec_I,           # prior: A ≈ I
    prior_a_cov=0.1,             # tight prior around identity
    init_a_mean=vec_I,
    init_a_cov=0.1,
    prior_b_cov=10.0,
    init_b_cov=100.0,
)

H = result.transition_matrix
B = result.control_matrix
det_A = float(jnp.linalg.det(H))
eigs = np.linalg.eigvals(np.array(H))

print(f"\n  det(A)={det_A:.4f}  |λ|={np.abs(eigs)}")
print(f"  A:\n{H}")
print(f"  B: {B.T[0]}")
print(f"  |B|={float(jnp.linalg.norm(B)):.4f}")

# ── 4. Compare predictions under different actions ────────────────────────
print("\nPredicting under 3 action regimes …")

# Prediction actions — big enough to produce visible divergence
u_fwd = 0.5    # strong forward push
u_rev = -0.5   # strong reverse push

action_regimes = {
    "u=0 (stop)":         [jnp.zeros(1)] * n_predict,
    f"u=+{u_fwd} (fwd)":  [jnp.array([u_fwd])] * n_predict,
    f"u={u_rev} (rev)":   [jnp.array([u_rev])] * n_predict,
}

predictions = {}
latent_trajs = {}
for name, pred_actions in action_regimes.items():
    res = infer(
        observations=sequence,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        transform_fn=lambda a: a.reshape(latent_dim, latent_dim),
        latent_dim=latent_dim,
        actions=jax_actions,
        action_dim=1,
        n_predict=n_predict,
        predict_actions=pred_actions,
        n_iterations=100,
        prior_a_mean=vec_I,
        prior_a_cov=0.1,
        init_a_mean=vec_I,
        init_a_cov=0.1,
        prior_b_cov=10.0,
        init_b_cov=100.0,
        verbose=False,
    )
    predictions[name] = res.predictions
    latent_trajs[name] = np.array(res.latent_means)

# ── 5. Visualise ────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(5, 10, figure=fig, hspace=0.5)

    # Row 0: observations
    for i in range(10):
        ax = fig.add_subplot(gs[0, i])
        idx = i * (n_observed // 10)
        ax.imshow(np.array(sequence[idx]).reshape(28, 28), cmap="gray")
        u_val = act_values[min(idx, len(act_values)-1)]
        ax.set_title(f"t={idx}\nu={u_val:.2f}", fontsize=7)
        ax.axis("off")
    fig.text(0.01, 0.88, "Observed", fontsize=10, va="center", rotation=90)

    # Rows 1-3: predictions under different regimes
    regime_colors = ["green", "red", "blue"]
    for row, ((name, preds), color) in enumerate(zip(predictions.items(), regime_colors)):
        for i in range(10):
            ax = fig.add_subplot(gs[row + 1, i])
            idx = n_observed + i * (n_predict // 10)
            ax.imshow(np.array(preds[idx]).reshape(28, 28), cmap="inferno")
            ax.set_title(f"+{i * (n_predict // 10)}", fontsize=7, color=color)
            ax.axis("off")
        fig.text(0.01, 0.72 - row * 0.16, name, fontsize=9, va="center",
                 rotation=90, color=color)

    # Row 4: diagnostics
    ax_act = fig.add_subplot(gs[4, :3])
    ax_act.fill_between(range(len(act_values)), act_values, alpha=0.3, color="g")
    ax_act.plot(act_values, "g-", linewidth=0.8)
    ax_act.axhline(0, color="gray", ls=":", alpha=0.3)
    ax_act.set(xlabel="t", ylabel="u[t]", title="Observed actions (normalised)")
    ax_act.grid(True, alpha=0.3)

    # Latent time series — show divergence across regimes
    ax_lat = fig.add_subplot(gs[4, 3:7])
    traj0 = list(latent_trajs.values())[0]
    t = np.arange(traj0.shape[0])
    # Pick the latent dim with largest B component
    b_dim = int(np.argmax(np.abs(np.array(B).ravel())))
    ax_lat.plot(t[:n_observed], traj0[:n_observed, b_dim],
                "k-", alpha=0.5, linewidth=1, label="observed")
    for (name, traj), color in zip(latent_trajs.items(), regime_colors):
        ax_lat.plot(t[n_observed:], traj[n_observed:, b_dim],
                    "-", color=color, alpha=0.8, linewidth=2, label=name)
    ax_lat.axvline(n_observed, color="k", ls="--", alpha=0.3)
    ax_lat.set(xlabel="time", ylabel=f"z[{b_dim}]",
               title=f"Latent dim {b_dim} (largest B component: {float(B[b_dim,0]):.3f})")
    ax_lat.legend(fontsize=7)
    ax_lat.grid(True, alpha=0.3)

    # A|B matrix
    ax_AB = fig.add_subplot(gs[4, 7:])
    AB = np.hstack([np.array(H), np.array(B)])
    im = ax_AB.imshow(AB, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_AB.set_title(f"[A | B]  det(A)={det_A:.3f}", fontsize=9)
    ax_AB.set_xticks(range(latent_dim + 1))
    ax_AB.set_xticklabels([f"a{j}" for j in range(latent_dim)] + ["b"], fontsize=8)
    for i in range(latent_dim):
        for j in range(latent_dim + 1):
            ax_AB.text(j, i, f"{float(AB[i,j]):.2f}", ha="center", va="center", fontsize=7)

    fig.suptitle(
        "System identification: same observations, different predicted futures via control",
        fontsize=13,
    )
    plt.savefig(os.path.join(OUTPUTS, "controlled_digits.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUTS}/controlled_digits.png")

except ImportError:
    print("(install matplotlib for visualisation)")
