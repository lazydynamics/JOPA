import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import os as _os
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    from jopa.envs import SimplePendulum
    from jopa.nn.vae import VAE, train_vae, save_params, load_params, make_encode_decode
    from jopa.em import variational_em
    from jopa.inference import plan

    # Use default (white) matplotlib theme — marimo renders on a white page

    _os.makedirs("checkpoints", exist_ok=True)
    _os.makedirs("outputs", exist_ok=True)
    return (
        SimplePendulum,
        VAE,
        jnp,
        load_params,
        make_encode_decode,
        mo,
        np,
        plan,
        plt,
        save_params,
        train_vae,
        variational_em,
    )


@app.cell
def title(SimplePendulum, mo, np):
    # Show pendulum images above the fold — the payoff before the explanation
    _env = SimplePendulum()
    _angles = np.linspace(-2.8, 2.8, 9)
    _frames = []
    for _a in _angles:
        _env.reset(theta=float(_a), theta_dot=0.0)
        _frames.append(_env.render())

    import os as _os
    _ffg_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "docs", "model.png")

    mo.vstack([
        mo.hstack(
            [mo.image(src=_f, width=75) for _f in _frames],
            justify="center", gap=0.3,
        ),
        mo.md("""
    # Poor Man's Active Inference via Message Passing

    **The task**: swing a pendulum to a target angle by observing only a pixelated 28x28 world.

    **No reward function. No policy network. No replay buffer.**
    Just a generative model and message passing.

    The entire pipeline — *learning to see*, *learning physics*, and *planning actions* — runs on **one factor graph**. Planning *is* inference.
    """).center(),
        mo.md("""
    One honest caveat: the VAE observation model is **pre-trained with standard gradient descent** — not message passing. The Variational EM refines it further, but the initial representation comes from deep learning. Everything else — learning dynamics, estimating states, inferring actions — is message passing.
    """).center(),
        mo.md("""
    _Based on [Active Inference for Physical AI Agents](https://arxiv.org/abs/2603.20927) (de Vries, 2026)
    and the [message-passing view of inference and learning](https://doi.org/10.3390/e23070807) (Şenöz et al., 2021).
    Built on [Lazy Dynamics](https://lazydynamics.com/)'s Bayesian inference toolkit.
    CT node rules were derived by hand; everything else was assembled by Claude Code._
    """).callout(kind="info"),
    ])
    return


@app.cell
def ffg_builder(plt):
    """Build a function that draws the FFG with highlighted message arrows."""
    import matplotlib.patches as _mp

    def draw_ffg(phase="none"):
        """Draw FFG. phase: 'none', 'forward', 'backward', 'vmp_params', 'vmp_actions', 'full_sysid', 'full_plan'."""
        _fig, _ax = plt.subplots(figsize=(12, 6))
        _ax.set_xlim(-2.5, 14)
        _ax.set_ylim(-4.0, 5.5)
        _ax.set_aspect("equal")
        _ax.axis("off")

        # Layout constants
        _xs = 3.5  # x spacing
        _yx = 0.0  # latent chain y
        _yv = -2.0  # VAE y
        _yo = -3.2  # observations y
        _yu = 1.2  # controls y
        _yp = 4.0  # params y

        _CT = "#4a90d9"; _LIK = "#5cb85c"; _PRI = "#aaaaaa"
        _DIM = "#666666"; _GLOW_FWD = "#e67e22"; _GLOW_BWD = "#e74c3c"
        _GLOW_VMP = "#9b59b6"; _GLOW_ACT = "#27ae60"

        def _box(_x, _y, _lbl, _c, _w=0.85, _h=0.55):
            _ax.add_patch(_mp.FancyBboxPatch(
                (_x-_w/2, _y-_h/2), _w, _h,
                boxstyle="round,pad=0.25", facecolor=_c, edgecolor="#444444",
                linewidth=1.0, zorder=2))
            _ax.text(_x, _y, _lbl, fontsize=7, ha="center", va="center",
                     color="white", fontweight="bold", zorder=3)

        def _dot(_x, _y, _lbl, _off=(0, 0.35)):
            _ax.plot(_x, _y, "ko", markersize=5, zorder=3)
            _ax.text(_x+_off[0], _y+_off[1], _lbl, fontsize=9,
                     ha="center", va="center", fontweight="bold", zorder=3)

        def _edge(_x1, _y1, _x2, _y2, **kw):
            _kw = dict(color=_DIM, linewidth=1.2, zorder=1)
            _kw.update(kw)
            _ax.plot([_x1, _x2], [_y1, _y2], **_kw)

        def _arrow(_x1, _y1, _x2, _y2, _c, _lw=2.0):
            _ax.annotate("", xy=(_x2, _y2), xytext=(_x1, _y1),
                         arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.15",
                                         color=_c, linewidth=_lw, zorder=10,
                                         connectionstyle="arc3,rad=0"))

        # Nodes
        _xp = [i * _xs for i in range(4)]
        _cp = [(_xp[i] + _xp[i+1]) / 2 for i in range(3)]

        # Prior on x0
        _box(_xp[0] - 1.2, _yx, "prior", _PRI, 0.7, 0.45)
        _edge(_xp[0] - 0.85, _yx, _xp[0] - 0.12, _yx)

        # Latent chain + transition factors
        for _i in range(4):
            _dot(_xp[_i], _yx, f"$x_{_i}$", (0, 0.38))
        for _k in range(3):
            _box(_cp[_k], _yx, "$p(x_t|x_{t-1})$", _CT, _w=1.2)
            _edge(_xp[_k] + 0.12, _yx, _cp[_k] - 0.6, _yx)
            _edge(_cp[_k] + 0.6, _yx, _xp[_k+1] - 0.12, _yx)

        # Likelihood (observation) factors
        for _i in range(4):
            _box(_xp[_i], _yv, "$p(y_t|x_t)$", _LIK, _w=1.1)
            _edge(_xp[_i], _yx - 0.12, _xp[_i], _yv + 0.28)
            _edge(_xp[_i], _yv - 0.28, _xp[_i], _yo + 0.12, linestyle="--")
            _dot(_xp[_i], _yo, f"$y_{_i}$", (0, -0.35))

        # Controls
        for _k in range(3):
            _ux = _cp[_k] - 1.1
            _edge(_cp[_k] - 0.43, _yx + 0.18, _ux + 0.08, _yu - 0.08, linestyle="--")
            _dot(_ux, _yu, f"$u_{_k}$", (-0.32, 0.0))

        # Shared params
        _pc = (_cp[0] + _cp[-1]) / 2
        _pxs = [_pc + d * 1.6 for d in [-0.8, 0.0, 0.8]]
        _plbls = ["$\\mathbf{a}$", "$W$", "$\\mathbf{b}$"]
        _bus_y = _yp - 1.6
        for _j in range(3):
            _dot(_pxs[_j], _yp, _plbls[_j], (0, 0.35))
            _box(_pxs[_j], _yp - 0.9, "prior", _PRI, 0.7, 0.4)
            _edge(_pxs[_j], _yp - 0.12, _pxs[_j], _yp - 0.7)
            _edge(_pxs[_j], _yp - 1.1, _pxs[_j], _bus_y)
        _edge(_pxs[0] - 0.3, _bus_y, _pxs[-1] + 0.3, _bus_y, color="#555555")
        for _c in _cp:
            _edge(_c, _yx + 0.28, _c, _bus_y, linestyle=":", color="#555555", alpha=0.5)

        _ax.text(_xp[-1] + 1.1, _yx, "$\\cdots$", fontsize=16, ha="center", va="center")

        # ── Animated arrows based on phase ────────────────────────
        if phase in ("forward", "full_sysid", "full_plan"):
            for _k in range(3):
                _arrow(_xp[_k] + 0.2, _yx + 0.08, _cp[_k] - 0.65, _yx + 0.08, _GLOW_FWD)
                _arrow(_cp[_k] + 0.65, _yx + 0.08, _xp[_k+1] - 0.2, _yx + 0.08, _GLOW_FWD)

        if phase in ("backward", "full_sysid", "full_plan"):
            for _k in range(3):
                _arrow(_xp[_k+1] - 0.2, _yx - 0.08, _cp[_k] + 0.65, _yx - 0.08, _GLOW_BWD)
                _arrow(_cp[_k] - 0.65, _yx - 0.08, _xp[_k] + 0.2, _yx - 0.08, _GLOW_BWD)

        if phase in ("vmp_params", "full_sysid"):
            for _c in _cp:
                _arrow(_c, _yx + 0.35, _c, _bus_y - 0.1, _GLOW_VMP, 2.0)

        if phase in ("vmp_actions", "full_plan"):
            for _k in range(3):
                _ux = _cp[_k] - 1.1
                _arrow(_cp[_k] - 0.48, _yx + 0.22, _ux + 0.15, _yu - 0.15, _GLOW_ACT, 2.0)

        # Legend based on phase
        _legend = []
        if phase in ("forward", "full_sysid", "full_plan"):
            _legend.append(plt.Line2D([0],[0], color=_GLOW_FWD, lw=2.5, label="forward (α)"))
        if phase in ("backward", "full_sysid", "full_plan"):
            _legend.append(plt.Line2D([0],[0], color=_GLOW_BWD, lw=2.5, label="backward (β)"))
        if phase in ("vmp_params", "full_sysid"):
            _legend.append(plt.Line2D([0],[0], color=_GLOW_VMP, lw=2.5, label="VMP → a, W, b"))
        if phase in ("vmp_actions", "full_plan"):
            _legend.append(plt.Line2D([0],[0], color=_GLOW_ACT, lw=2.5, label="VMP → actions u"))
        if _legend:
            _ax.legend(handles=_legend, loc="upper right", fontsize=8,
                       framealpha=0.95, edgecolor="#cccccc", facecolor="white")

        # Title
        _titles = {
            "none": "Factor graph representation of the model",
            "forward": "Forward messages (α) propagate from observations",
            "backward": "Backward messages (β) propagate from goal",
            "vmp_params": "VMP: update beliefs about dynamics parameters",
            "vmp_actions": "VMP: infer optimal actions",
            "full_sysid": "System identification — all messages",
            "full_plan": "Planning as inference — all messages",
        }
        _fig.suptitle(_titles.get(phase, ""), fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()
        return _fig

    return (draw_ffg,)


@app.cell
def ffg_slider(mo):
    _phases = {
        0: "The model (no messages)",
        1: "Forward messages α →",
        2: "Backward messages ← β",
        3: "System ID: learn A, B, W",
        4: "Planning: infer actions u",
    }
    ffg_phase = mo.ui.slider(
        start=0, stop=4, step=1, value=0,
        label="**Message passing phase**", show_value=True, full_width=True,
    )
    ffg_phase


@app.cell
def ffg_display(draw_ffg, ffg_phase, mo, plt):
    _phase_map = {0: "none", 1: "forward", 2: "backward", 3: "full_sysid", 4: "full_plan"}
    _phase_desc = {
        0: "The factor graph before any messages are passed.",
        1: "**Forward messages (α)** propagate observations from left to right through the transition factors.",
        2: "**Backward messages (β)** propagate the goal observation from right to left.",
        3: "**System identification**: forward-backward (orange/red) estimates states, VMP (purple) updates dynamics parameters A, W, B.",
        4: "**Planning**: same forward-backward, but VMP messages (green) now flow to the **action nodes** u instead of the parameters.",
    }
    _fig = draw_ffg(_phase_map[ffg_phase.value])
    _out = mo.vstack([_fig, mo.md(_phase_desc[ffg_phase.value]).center()])
    plt.close(_fig)
    _out


@app.cell
def config():
    latent_dim = 4
    return (latent_dim,)


@app.cell
def generate_data(SimplePendulum, jnp, np):
    env = SimplePendulum()
    _n_traj = 10
    _traj_len = 80

    trajectories = []
    for _ep in range(_n_traj):
        _rng = np.random.RandomState(_ep)
        env.reset(seed=_ep)
        _obs = [jnp.array(env.render())]
        _acts = []
        for _t in range(_traj_len - 1):
            _torque = _rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0]) + _rng.randn() * 0.5
            _acts.append(jnp.array([_torque]))
            env.step(_torque)
            _obs.append(jnp.array(env.render()))
        trajectories.append({"observations": _obs, "actions": _acts})

    n_frames = sum(len(t["observations"]) for t in trajectories)
    return env, n_frames, trajectories


@app.cell
def show_data(SimplePendulum, mo, n_frames, np):
    _env = SimplePendulum()
    _sample_frames = []
    for _th in np.linspace(-2.5, 2.5, 7):
        _env.reset(theta=float(_th), theta_dot=0.0)
        _sample_frames.append(_env.render())

    mo.vstack([
        mo.md(f"""
    ## Phase 1 — Flailing Around

    The agent applies random torques and watches what happens.
    No strategy. No curriculum. Just chaos and a camera.
    **{n_frames} frames** across 10 trajectories.
    """),
        mo.hstack(
            [mo.image(src=_f, width=110) for _f in _sample_frames],
            justify="center", gap=0.5,
        ),
        mo.md("*Seven pendulum angles from -2.5 to +2.5 radians. This is all the agent sees — 28x28 grayscale.*").center(),
    ])
    return


@app.cell
def train_vae_cell(
    SimplePendulum,
    VAE,
    jnp,
    latent_dim,
    load_params,
    make_encode_decode,
    np,
    save_params,
    train_vae,
    trajectories,
):
    model = VAE(latent_dim=latent_dim)
    _vae_path = "checkpoints/vae_pendulum_d4.npz"

    _env = SimplePendulum()
    vae_sample_imgs = []
    for _theta in [-2.5, -1.2, 0.0, 1.2, 2.5]:
        _env.reset(theta=_theta, theta_dot=0.0)
        vae_sample_imgs.append(_env.render())

    vae_snapshots = []

    def _vae_callback(_epoch, _cb_params, _loss):
        if _epoch == 1 or _epoch % 5 == 0:
            _enc, _dec = make_encode_decode(model, _cb_params)
            _recons = [np.array(_dec(_enc(jnp.array(_img))[0])) for _img in vae_sample_imgs]
            vae_snapshots.append({"epoch": _epoch, "recons": _recons, "loss": float(_loss)})

    try:
        params = load_params(model, _vae_path)
    except FileNotFoundError:
        _all_frames = []
        for _traj in trajectories:
            _all_frames.extend([np.array(o) for o in _traj["observations"]])
        for _theta in np.linspace(-np.pi, np.pi, 200):
            _env.reset(theta=_theta, theta_dot=0.0)
            _all_frames.append(_env.render())
        _images = np.stack(_all_frames)
        model, params = train_vae(
            _images, latent_dim=latent_dim, epochs=100, seed=42,
            callback=_vae_callback,
        )
        save_params(params, _vae_path)

    if not vae_snapshots:
        _enc, _dec = make_encode_decode(model, params)
        _recons = [np.array(_dec(_enc(jnp.array(_img))[0])) for _img in vae_sample_imgs]
        vae_snapshots.append({"epoch": 100, "recons": _recons, "loss": None})
    return model, params, vae_sample_imgs, vae_snapshots


@app.cell
def vae_heading(mo):
    mo.md("""
    ## Phase 2 — Learning to See

    The VAE compresses 28x28 images into a **4D latent space**. This is the observation model — the bridge between pixels and the latent world where dynamics will live.

    Standard deep learning. Scrub the slider to watch reconstructions sharpen:
    """)
    return


@app.cell
def vae_slider(mo, vae_snapshots):
    epoch_slider = mo.ui.slider(
        start=0, stop=len(vae_snapshots) - 1, step=1,
        value=len(vae_snapshots) - 1,
        label="**VAE Epoch**", show_value=True, full_width=True,
    )
    epoch_slider


@app.cell
def vae_display(epoch_slider, mo, plt, vae_sample_imgs, vae_snapshots):
    _snap = vae_snapshots[epoch_slider.value]
    _originals = mo.hstack(
        [mo.image(src=_img, width=110) for _img in vae_sample_imgs],
        justify="center", gap=0.5,
    )
    _recons = mo.hstack(
        [mo.image(src=_snap["recons"][_i], width=110) for _i in range(5)],
        justify="center", gap=0.5,
    )

    _elements = [
        mo.md(f"**Epoch {_snap['epoch']}/100**"),
        mo.md("Originals:"), _originals,
        mo.md("Reconstructions:"), _recons,
    ]

    # Loss plot (skip if loaded from checkpoint)
    if _snap["loss"] is not None and len(vae_snapshots) > 1:
        _losses = [s["loss"] for s in vae_snapshots[:epoch_slider.value + 1] if s["loss"] is not None]
        _epochs = [s["epoch"] for s in vae_snapshots[:epoch_slider.value + 1] if s["loss"] is not None]
        _all_losses = [s["loss"] for s in vae_snapshots if s["loss"] is not None]
        _all_epochs = [s["epoch"] for s in vae_snapshots if s["loss"] is not None]

        _fig, _ax = plt.subplots(figsize=(8, 2.5))
        _ax.fill_between(_epochs, _losses, alpha=0.15, color="#f85149")
        _ax.plot(_epochs, _losses, color="#f85149", linewidth=2)
        _ax.scatter([_epochs[-1]], [_losses[-1]], color="#f85149", s=50, zorder=5, edgecolors="white", linewidths=0.5)
        _ax.set_xlabel("epoch")
        _ax.set_ylabel("ELBO loss")
        _ax.set_xlim(_all_epochs[0] - 1, _all_epochs[-1] + 1)
        _ax.set_ylim(min(_all_losses) * 0.92, max(_all_losses) * 1.05)
        _ax.grid(True)
        plt.tight_layout()
        _elements.append(_fig)

    mo.vstack(_elements)


@app.cell
def vae_gif(mo, np, vae_sample_imgs, vae_snapshots):
    """Auto-playing GIF of VAE learning."""
    from PIL import Image as _PILImage
    import io as _io

    if len(vae_snapshots) > 1:
        _pil_frames = []
        for _snap in vae_snapshots:
            # Stitch originals (top) + reconstructions (bottom) into one wide frame
            _row_orig = np.hstack(vae_sample_imgs)
            _row_recon = np.hstack(_snap["recons"])
            _combined = np.vstack([_row_orig, np.ones((2, _row_orig.shape[1])) * 0.3, _row_recon])
            _uint8 = np.clip(_combined * 255, 0, 255).astype(np.uint8)
            _pil_frames.append(_PILImage.fromarray(_uint8, "L").resize(
                (_uint8.shape[1] * 3, _uint8.shape[0] * 3), _PILImage.NEAREST))

        _buf = _io.BytesIO()
        _pil_frames[0].save(_buf, format="GIF", append_images=_pil_frames[1:],
                            save_all=True, duration=300, loop=0)
        _buf.seek(0)
        mo.vstack([
            mo.md("*Reconstructions sharpening over training (top: originals, bottom: reconstructions):*").center(),
            mo.image(src=_buf, width=500).center(),
        ])
    else:
        # Loaded from checkpoint — show static before/after using the single snapshot
        _snap = vae_snapshots[0]
        mo.vstack([
            mo.md("*(VAE loaded from checkpoint — delete `checkpoints/vae_pendulum_d4.npz` and re-run to see training animation)*").center(),
            mo.md("**Final reconstructions:**").center(),
            mo.hstack(
                [mo.image(src=_snap["recons"][_i], width=110) for _i in range(5)],
                justify="center", gap=0.5,
            ),
        ])
    return


@app.cell
def model_section(mo):
    mo.accordion({
        "The Generative Model (click to expand)": mo.md("""
    A linear dynamical system in the latent space of a VAE:

    $$
    \\begin{aligned}
    \\mathbf{a} &\\sim \\mathcal{N}(\\mu_a, \\Sigma_a), \\quad W \\sim \\text{Wishart}(\\nu_0, I), \\quad \\mathbf{b} \\sim \\mathcal{N}(0, \\Sigma_b) \\\\[4pt]
    x_1 &\\sim \\mathcal{N}(0, I) \\\\
    x_t \\mid x_{t-1}, u_{t-1} &\\sim \\mathcal{N}(A\\, x_{t-1} + B\\, u_{t-1},\\; W^{-1}) \\\\
    y_t \\mid x_t &\\sim p_\\theta(y_t \\mid x_t) \\quad \\text{(VAE decoder)}
    \\end{aligned}
    $$

    where $A = \\text{reshape}(\\mathbf{a})$, $B = \\text{reshape}(\\mathbf{b})$.
    The prior on $\\mathbf{a}$ is centered near the identity matrix — a soft assumption of near-random-walk dynamics.

    Three tasks run on the **same factor graph** — only which variables are latent changes:

    | Task | Inferred | Fixed |
    |------|----------|-------|
    | **System identification** | $A, B, W$ (VMP) | VAE, images, actions |
    | **Variational EM** | $A, B, W$ (VMP) + VAE (gradient M-step) | images, actions |
    | **Planning** | actions $u_t$ (VMP) | $A, B, W$, start + goal |
    """)
    })
    return


@app.cell
def sysid_cell(
    jnp,
    latent_dim,
    make_encode_decode,
    model,
    np,
    params,
    trajectories,
    variational_em,
):
    import pickle as _pickle

    _traj0_obs = trajectories[0]["observations"]
    _traj0_acts = trajectories[0]["actions"]
    _sample_indices = [0, 10, 20, 35, 50]
    sample_pairs = [(jnp.array(_traj0_obs[i]), jnp.array(_traj0_obs[i + 1])) for i in _sample_indices]
    sample_actions = [_traj0_acts[i] for i in _sample_indices]

    _cache_path = "checkpoints/em_result_raw_torque.pkl"
    _vec_I = jnp.eye(latent_dim).ravel()

    # Try loading cached EM result
    _loaded = False
    try:
        with open(_cache_path, "rb") as _f:
            _cached = _pickle.load(_f)
        result = _cached["result"]
        em_snapshots = _cached["em_snapshots"]
        _loaded = True
        print(f"Loaded cached EM result from {_cache_path}")
    except (FileNotFoundError, Exception):
        pass

    if not _loaded:
        em_snapshots = []

        def _em_callback(_it, _cb_params, _mA, _mB, _loss, _det_A):
            _enc, _dec = make_encode_decode(model, _cb_params)
            _preds = []
            for (_img_t, _), _u_t in zip(sample_pairs, sample_actions):
                _z_t, _ = _enc(_img_t)
                _z_next = _mA @ _z_t
                if _mB is not None:
                    _z_next = _z_next + _mB @ _u_t
                _preds.append(np.array(_dec(_z_next)))
            em_snapshots.append({"preds": _preds, "loss": float(_loss), "det_A": float(_det_A)})

        result = variational_em(
            model=model, params=params, trajectories=trajectories,
            latent_dim=latent_dim, action_dim=1,
            n_em_iterations=30, n_vmp_iterations=10, n_m_steps=20, lr=5e-5,
            beta_recon=1.0, prior_a_mean=_vec_I, prior_a_cov=0.5, init_a_cov=0.5,
            prior_b_cov=10.0, init_b_cov=100.0, seed=42,
            callback=_em_callback,
        )

        # Cache for next time
        try:
            with open(_cache_path, "wb") as _f:
                _pickle.dump({"result": result, "em_snapshots": em_snapshots}, _f)
            print(f"Saved EM result to {_cache_path}")
        except Exception:
            pass

    det_A = float(jnp.linalg.det(result.transition_matrix))
    eig_mods = np.abs(np.linalg.eigvals(np.array(result.transition_matrix)))
    norm_B = float(np.linalg.norm(result.control_matrix))
    params_em = result.params
    em_actuals = [np.array(_p[1]) for _p in sample_pairs]
    return (det_A, eig_mods, em_actuals, em_snapshots, norm_B, params_em, result,
            sample_actions, sample_pairs)


@app.cell
def sysid_heading(det_A, eig_mods, mo, norm_B):
    _eig_str = ", ".join(f"{e:.3f}" for e in sorted(eig_mods, reverse=True))
    mo.vstack([
        mo.md(f"""
    ## Phase 3 — Learning Physics

    This is where message passing takes over. **Variational EM** alternates on the factor graph:

    - **E-step**: forward-backward messages on the latent chain (state estimation) + VMP messages to update beliefs about $A$, $B$, $W$ (parameter learning)
    - **M-step**: gradient-based update of the VAE so the encoder agrees with the dynamics posterior

    The E-step messages are derived from a structured mean-field factorisation $q(x) \\, q(\\mathbf{{a}}) \\, q(W) \\, q(\\mathbf{{b}})$ — each message has a closed-form update rule.

    ### What the model discovered

    | | Value | Meaning |
    |---|---|---|
    | det(A) | {det_A:.4f} | Volume contraction per step in latent space |
    | \\|eigenvalues\\| | {_eig_str} | All < 1 → stable autonomous (unforced) dynamics |
    | \\|B\\| | {norm_B:.3f} | How strongly torque affects latent state |

    No explicit physics equations (F=ma, moment of inertia) were provided — the dynamics are learned from data. The linear state-space structure $x_{{t+1}} = Ax + Bu$ is itself an inductive bias, but the specific matrices are discovered from pixels + torques. Scrub to watch predictions improve:
    """),
        mo.accordion({
            "Message passing pseudocode": mo.md("""
    ```
    Forward:   α[t] = CT_forward(combine(α[t-1], vae[t-1]), cache, u[t-1])
    Backward:  β[t] = CT_backward(combine(β[t+1], vae[t+1]), cache, u[t])
    VMP:       q(a) ← prior(a) · ∏_k msg_a(k)
           q(W) ← prior(W) · ∏_k msg_W(k)
           q(b) ← prior(b) · ∏_k msg_b(k)
    ```
    """)
        }),
    ])
    return


@app.cell
def em_slider(em_snapshots, mo):
    em_iter_slider = mo.ui.slider(
        start=0, stop=len(em_snapshots) - 1, step=1,
        value=len(em_snapshots) - 1,
        label="**EM Iteration**", show_value=True, full_width=True,
    )
    em_iter_slider


@app.cell
def em_display(em_actuals, em_iter_slider, em_snapshots, mo, plt):
    _snap = em_snapshots[em_iter_slider.value]
    _n = em_iter_slider.value + 1

    _actuals_row = mo.hstack(
        [mo.image(src=_img, width=110) for _img in em_actuals],
        justify="center", gap=0.5,
    )
    _preds_row = mo.hstack(
        [mo.image(src=_snap["preds"][_i], width=110) for _i in range(5)],
        justify="center", gap=0.5,
    )

    _dets = [s["det_A"] for s in em_snapshots[:_n]]
    _losses = [s["loss"] for s in em_snapshots[:_n]]
    _all_dets = [s["det_A"] for s in em_snapshots]
    _all_losses = [s["loss"] for s in em_snapshots]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
    _ax1.fill_between(range(1, _n + 1), _dets, alpha=0.15, color="#3fb950")
    _ax1.plot(range(1, _n + 1), _dets, color="#3fb950", linewidth=2)
    _ax1.scatter([_n], [_dets[-1]], color="#3fb950", s=50, zorder=5, edgecolors="white", linewidths=0.5)
    _ax1.set_xlim(0.5, len(em_snapshots) + 0.5)
    _ax1.set_ylim(min(_all_dets) - 0.02, max(_all_dets) + 0.02)
    _ax1.set_xlabel("EM iteration")
    _ax1.set_ylabel("det(A)")
    _ax1.grid(True)

    _ax2.fill_between(range(1, _n + 1), _losses, alpha=0.15, color="#f85149")
    _ax2.plot(range(1, _n + 1), _losses, color="#f85149", linewidth=2)
    _ax2.scatter([_n], [_losses[-1]], color="#f85149", s=50, zorder=5, edgecolors="white", linewidths=0.5)
    _ax2.set_xlim(0.5, len(em_snapshots) + 0.5)
    _ax2.set_ylim(min(_all_losses) * 0.95, max(_all_losses) * 1.05)
    _ax2.set_xlabel("EM iteration")
    _ax2.set_ylabel("M-step loss")
    _ax2.grid(True)

    plt.tight_layout()

    mo.vstack([
        mo.md(f"**EM {_n}/{len(em_snapshots)}** — loss {_snap['loss']:.1f}, det(A) {_snap['det_A']:.4f}"),
        mo.md("**Actual t+1 (ground truth):**"), _actuals_row,
        mo.md("**Predicted t+1 (model):**"), _preds_row,
        _fig,
    ])


@app.cell
def em_gif(em_actuals, em_snapshots, mo, np):
    """Auto-playing GIF of dynamics predictions converging."""
    from PIL import Image as _PILImage
    import io as _io

    _pil_frames = []
    for _snap in em_snapshots:
        _row_actual = np.hstack(em_actuals)
        _row_pred = np.hstack(_snap["preds"])
        _combined = np.vstack([_row_actual, np.ones((2, _row_actual.shape[1])) * 0.3, _row_pred])
        _uint8 = np.clip(_combined * 255, 0, 255).astype(np.uint8)
        _pil_frames.append(_PILImage.fromarray(_uint8, "L").resize(
            (_uint8.shape[1] * 3, _uint8.shape[0] * 3), _PILImage.NEAREST))

    _buf = _io.BytesIO()
    _pil_frames[0].save(_buf, format="GIF", append_images=_pil_frames[1:],
                        save_all=True, duration=400, loop=0)
    _buf.seek(0)
    mo.vstack([
        mo.md("*Predictions converging to ground truth over EM iterations (top: actual, bottom: predicted):*").center(),
        mo.image(src=_buf, width=500).center(),
    ])
    return


@app.cell
def planning_intro(mo):
    mo.vstack([
        mo.md("""
## Phase 4 — Planning as Inference

Fix the learned model. Ask: **what actions swing the pendulum to θ=π?**

Same factor graph, but now **actions are latent** and only start + goal images are observed. The planner runs the same forward-backward + VMP machinery — this time computing messages to the action variables instead of the dynamics parameters. Same node, different edges.

Receding horizon: plan 8 steps, execute 2, re-observe, repeat.
"""),
        mo.accordion({
            "VMP action update": mo.md("""
```
for each transition k:
    q_yx  = ct_marginal_yx(m_ys[k], m_xs[k], cache, u[k])
    msg_u = ct_message_u(q_yx, cache)        ← message from CT factor to action
    q(u_k) = combine(prior_u, msg_u)          ← posterior = prior × likelihood message
```
Same `ct_marginal_yx` that produces messages for learning A, B, W now produces messages for planning actions. One node, many interfaces.
""")
        }),
    ])
    return


@app.cell
def run_all_plans(
    SimplePendulum, jnp, latent_dim, make_encode_decode,
    model, np, params_em, plan, result,
):
    """Pre-compute planning results for three target angles."""
    _encode_fn, _decode_fn = make_encode_decode(model, params_em)

    # Each target has its own optimal exec_steps (see diagnostics)
    _targets = [
        {"label": "θ = π (swing to top)", "goal": np.pi, "exec_steps": 2, "n_replans": 11},
    ]

    plan_results = {}
    for _tgt in _targets:
        _env = SimplePendulum()
        _env.reset(theta=0.0, theta_dot=0.0)
        _states = [_env.state.copy()]
        _frames = [_env.render()]
        _torques = []

        _env_g = SimplePendulum()
        _env_g.reset(theta=_tgt["goal"], theta_dot=0.0)
        _goal_img = jnp.array(_env_g.render())

        for _cycle in range(_tgt["n_replans"]):
            _current_img = jnp.array(_env.render())
            _obs_plan = [_current_img] + [None] * 6 + [_goal_img]
            _pr = plan(
                observations=_obs_plan,
                encode_fn=_encode_fn, decode_fn=_decode_fn,
                q_a=result.q_a, q_W=result.q_W, q_b=result.q_b,
                latent_dim=latent_dim, action_dim=1,
                n_iterations=200, verbose=False,
            )
            for _i in range(min(_tgt["exec_steps"], len(_pr.actions))):
                _t = float(_pr.actions[_i][0])
                _torques.append(_t)
                _env.step(_t)
                _states.append(_env.state.copy())
                _frames.append(_env.render())

        plan_results[_tgt["label"]] = {
            "goal_theta": _tgt["goal"],
            "sim_states_arr": np.array(_states),
            "sim_frames": _frames,
            "all_torques": _torques,
        }

    return (plan_results,)


@app.cell
def goal_picker(mo, plan_results):
    goal_dropdown = mo.ui.dropdown(
        options=list(plan_results.keys()),
        value="θ = π (swing to top)",
        label="**Target angle**",
    )
    goal_dropdown


@app.cell
def unpack_plan(plan_results, goal_dropdown, np):
    _sel = plan_results[goal_dropdown.value]
    goal_theta = _sel["goal_theta"]
    sim_states_arr = _sel["sim_states_arr"]
    sim_frames = _sel["sim_frames"]
    all_torques = _sel["all_torques"]
    return (goal_theta, sim_states_arr, sim_frames, all_torques)


@app.cell
def plan_slider(mo, sim_states_arr):
    step_slider = mo.ui.slider(
        start=0, stop=len(sim_states_arr) - 1, step=1, value=0,
        label="**Execution Step**", show_value=True, full_width=True,
    )
    step_slider


@app.cell
def plan_display(
    all_torques,
    goal_theta,
    mo,
    np,
    plt,
    sim_frames,
    sim_states_arr,
    step_slider,
    SimplePendulum,
):
    _step = step_slider.value
    _theta_now = sim_states_arr[_step, 0]
    _err = abs((_theta_now - goal_theta + np.pi) % (2 * np.pi) - np.pi)

    # Show START, CURRENT, GOAL side by side
    _env_s = SimplePendulum()
    _env_s.reset(theta=0.0, theta_dot=0.0)
    _start_img = _env_s.render()
    _env_g = SimplePendulum()
    _env_g.reset(theta=goal_theta, theta_dot=0.0)
    _goal_img = _env_g.render()

    _frames_row = mo.hstack([
        mo.vstack([mo.image(src=_start_img, width=140), mo.md("**START** θ=0").center()]),
        mo.md("→"),
        mo.vstack([mo.image(src=sim_frames[_step], width=140), mo.md(f"**NOW** θ={_theta_now:.2f}").center()]),
        mo.md("→"),
        mo.vstack([mo.image(src=_goal_img, width=140), mo.md(f"**GOAL** θ={goal_theta:.2f}").center()]),
    ], justify="center", gap=1.0)

    _fig1, _ax1 = plt.subplots(figsize=(8, 3))
    _ax1.fill_between(range(_step + 1), sim_states_arr[:_step + 1, 0], alpha=0.1, color="#3fb950")
    _ax1.plot(sim_states_arr[:_step + 1, 0], color="#3fb950", linewidth=2.5)
    _ax1.axhline(goal_theta, color="#f85149", ls="--", alpha=0.6, label=f"goal = {goal_theta:.2f}")
    _ax1.axhline(0.0, color="#58a6ff", ls="--", alpha=0.3, label="start = 0")
    _ax1.set_xlim(0, len(sim_states_arr) - 1)
    _ax1.set_ylim(sim_states_arr[:, 0].min() - 0.3, max(sim_states_arr[:, 0].max(), goal_theta) + 0.3)
    _ax1.set_xlabel("step")
    _ax1.set_ylabel("θ (rad)")
    _ax1.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
    _ax1.grid(True)
    plt.tight_layout()

    _fig2, _ax2 = plt.subplots(figsize=(8, 2.5))
    _n_t = min(_step, len(all_torques))
    if _n_t > 0:
        _colors = ["#58a6ff" if t >= 0 else "#f85149" for t in all_torques[:_n_t]]
        _ax2.bar(range(_n_t), all_torques[:_n_t], color=_colors, alpha=0.8, width=0.8)
    _ax2.axhline(0, color="#30363d", linewidth=0.5)
    _ax2.set_xlim(-0.5, max(len(all_torques) - 0.5, 0.5))
    if all_torques:
        _ax2.set_ylim(min(all_torques) * 1.2, max(all_torques) * 1.2)
    _ax2.set_xlabel("step")
    _ax2.set_ylabel("torque (Nm)")
    _ax2.grid(True)
    plt.tight_layout()

    mo.vstack([
        mo.md(f"**Step {_step}/{len(sim_states_arr) - 1}** — θ = {_theta_now:.3f}, error = {_err:.3f} rad"),
        _frames_row,
        _fig1,
        _fig2,
    ])


@app.cell
def plan_gif(mo, np, sim_frames):
    """Auto-playing GIF of the pendulum swing-up."""
    from PIL import Image as _PILImage
    import io as _io

    _pil_frames = []
    for _f in sim_frames:
        _uint8 = np.clip(_f * 255, 0, 255).astype(np.uint8)
        _pil_frames.append(_PILImage.fromarray(_uint8, "L").resize((168, 168), _PILImage.NEAREST))

    _buf = _io.BytesIO()
    _pil_frames[0].save(_buf, format="GIF", append_images=_pil_frames[1:],
                        save_all=True, duration=200, loop=0)
    _buf.seek(0)
    mo.vstack([
        mo.md("*The pendulum swinging up — each frame is a real simulation step driven by inferred actions:*").center(),
        mo.image(src=_buf, width=200).center(),
    ])
    return


@app.cell
def planning_result(goal_theta, mo, np, sim_states_arr):
    _final = sim_states_arr[-1, 0]
    _err = abs((_final - goal_theta + np.pi) % (2 * np.pi) - np.pi)
    mo.md(f"""
    ### Result

    **θ = 0** (hanging down) → **θ = {_final:.2f}** — error **{_err:.3f} rad** from π.

    Learned a dynamics model from random exploration, planned and executed a swing-up in closed loop. All from 28×28 pixels.

    The model can't stabilise at the top — the linear dynamics predict convergence to the resting position, and a single image doesn't encode velocity. Apparently Claude didn't come with built-in stability guarantees. But the swing-up itself is learned dynamics + message-passing planning working together.
    """).callout(kind="success")
    return


@app.cell
def zoom_out(mo):
    mo.md("""
    ## The Trick

    Scroll back up. The sliders in Phase 2, 3, and 4 are all running the **same underlying code** — `ct_forward`, `ct_backward`, and VMP messages on the same factor graph.

    - `ct_message_a` learned the transition matrix $A$
    - `ct_message_b` learned the control matrix $B$
    - `ct_message_u` planned the actions

    **Same node. Same math. Different edges.** That's the factor graph formalism — one set of message-passing rules gives you system identification, state estimation, and planning for free.
    """).callout(kind="warn")
    return


@app.cell
def comparison(mo):
    mo.vstack([
        mo.md("""
    ## Why "Poor Man's"?

    Most world models are neural networks end to end: a ViT encodes, a transformer predicts, CEM samples actions. That works. It also takes thousands of lines of carefully tuned code.

    This notebook does the same job in **~800 lines of math**. The secret: closed-form message passing replaces learned components wherever possible. The only neural network is the VAE — and even that gets refined by messages from the dynamics model.

    What we don't have: epistemic priors (no curiosity-driven exploration), nonlinear dynamics nodes, expected free energy for planning. Real Active Inference (de Vries, [arXiv:2603.20927](https://arxiv.org/abs/2603.20927), 2026) does all of that. We're the budget version — but the same factor graph architecture.
    """),
        mo.accordion({
            "Comparison tables": mo.md("""
    | | Active Inference | This notebook |
    |---|---|---|
    | **Dynamics** | Nonlinear, full message passing | Linear in learned latent space |
    | **Observation model** | Bayesian | VAE + gradient M-step |
    | **Planning** | Expected free energy | Goal-conditioned VMP |
    | **Exploration** | Epistemic (curiosity) | Random |

    | | Message passing (ours) | Neural world models |
    |---|---|---|
    | Encode | VAE → Gaussian message | CNN/ViT → embedding |
    | Predict | Analytic (one matrix multiply) | Learned predictor (RNN/Transformer) |
    | Learn dynamics | Closed-form VMP | Backprop |
    | Plan | VMP on actions | CEM / MPPI / policy gradient |
    """)
        }),
    ])
    return


@app.cell
def outro(mo):
    mo.md("""
    ---

    ## Try It

    ```bash
    git clone https://github.com/lazydynamics/jopa && cd jopa
    uv pip install -e .
    uv run marimo run notebook.py
    ```

    **Things to play with:**
    - Change `goal_theta` in the planning cell — any angle works
    - Reduce `latent_dim` to 2 and watch what breaks
    - Look at `jopa/inference.py` — the planning loop is ~40 lines of VMP
    - Look at `jopa/nodes/transition.py` — every message rule derived from one factor

    *CT node rules: structured mean-field VMP on the continuous transition factor.*
    *Everything else: Claude Code, from a model description.*

    *References: de Vries, B. "Active Inference for Physical AI Agents — An Engineering Perspective", arXiv:2603.20927, 2026. Şenöz, I. et al. "Variational Message Passing and Local Constraint Manipulation in Factor Graphs", Entropy, 2021.*
    """)
    return


if __name__ == "__main__":
    app.run()
