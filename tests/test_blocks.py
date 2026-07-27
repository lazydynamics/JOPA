"""Semantic tests for the JOPA API: distributions, message passing, blocks, planning."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jopa.blocks import (
    Block,
    Frozen,
    JointModel,
    KnownPhysics,
    LearnedAffine,
    LearnedDelayLinear,
    LearnedLinear,
    LearnedVAE,
    LinearCoupling,
    Observation,
)
from jopa.distributions import (
    Gaussian,
    Wishart,
    combine_gaussians,
    gaussian_mean,
    gaussian_mean_cov,
    gaussian_prior,
    vague_gaussian,
)
from jopa.message_passing import compute_marginals, forward_backward
from jopa.nn.vae import VAE
from jopa.nodes.transition import CTCache, CTMeta

# ---- package surface -------------------------------------------------------

def test_package_exports_are_importable():
    """The documented `from jopa import …` surface (README) resolves and is sane."""
    import jopa
    for name in jopa.__all__:
        assert hasattr(jopa, name), f"jopa.{name} is in __all__ but not importable"
    from jopa import Block, Gaussian, LearnedLinear, LearnedVAE
    assert all(callable(x) for x in (Gaussian, Block, LearnedLinear, LearnedVAE))


# ---- helpers ---------------------------------------------------------------

def _msg(mean, prec):
    mean = jnp.asarray(mean, dtype=float)
    lam = prec * jnp.eye(mean.shape[0])
    return Gaussian(eta=lam @ mean, lam=lam)


def _linear_traj(A, T, obs_std, proc_std, rng):
    d = A.shape[0]
    x = rng.randn(d)
    seq = []
    for _ in range(T):
        seq.append(_msg(x + obs_std * rng.randn(d), 1.0 / obs_std ** 2))
        x = A @ x + proc_std * rng.randn(d)
    return seq


# ============================================================================
# Distributions — natural parameter primitives
# ============================================================================

def test_gaussian_information_form_roundtrip():
    mu, cov = jnp.array([1.0, 2.0]), jnp.array([[1.0, 0.3], [0.3, 2.0]])
    lam = jnp.linalg.inv(cov)
    mu_r, cov_r = gaussian_mean_cov(Gaussian(eta=lam @ mu, lam=lam))
    assert jnp.allclose(mu_r, mu) and jnp.allclose(cov_r, cov)


def test_combine_gaussians_is_natural_parameter_sum():
    g1 = Gaussian(eta=jnp.array([1.0, 0.0]), lam=jnp.eye(2))
    g2 = Gaussian(eta=jnp.array([0.0, 2.0]), lam=2 * jnp.eye(2))
    fused = combine_gaussians(g1, g2)
    assert jnp.allclose(fused.eta, g1.eta + g2.eta)
    assert jnp.allclose(fused.lam, g1.lam + g2.lam)


def test_vague_gaussian_is_combine_identity():
    g = Gaussian(eta=jnp.array([3.0]), lam=jnp.array([[4.0]]))
    f = combine_gaussians(g, vague_gaussian(1))
    assert jnp.allclose(f.eta, g.eta) and jnp.allclose(f.lam, g.lam)


# ============================================================================
# Message passing — BP on a linear-Gaussian chain
# ============================================================================

def test_forward_backward_recovers_true_states():
    """With known dynamics + low noise, the smoothed posterior tracks the truth."""
    A = jnp.array([[0.95, 0.10], [-0.10, 0.95]])
    rng = np.random.RandomState(0)
    xs = [rng.randn(2)]
    for _ in range(29):
        xs.append(A @ xs[-1] + 0.01 * rng.randn(2))
    obs = [_msg(x + 0.01 * rng.randn(2), 1e4) for x in xs]

    q_a = gaussian_prior(4, 1e-8, jnp.asarray(A).ravel())
    q_W = Wishart(df=1e6, inv_scale=1e6 * 0.01 * jnp.eye(2))   # E[W] = 100·I
    cache = CTCache(q_a, q_W, CTMeta(lambda a: a.reshape(2, 2)))
    prior_x = Gaussian(eta=jnp.zeros(2), lam=jnp.eye(2) * 1e-6)

    alphas, betas, _, _ = forward_backward(prior_x, obs, cache)
    means = np.array(jax.vmap(gaussian_mean)(compute_marginals(alphas, betas, obs)))
    truth = np.array([np.array(x) for x in xs])
    assert np.max(np.abs(means - truth)) < 0.05


# ============================================================================
# Observations
# ============================================================================

def test_frozen_observation_autowraps_callable():
    blk = Block("x", LearnedLinear(2), observe=lambda d: _msg(d, 1e4))
    assert isinstance(blk.observation, Observation)
    assert blk.observation.learnable is False
    assert blk.observation.message(jnp.array([1.0, 2.0])).eta.shape == (2,)


def test_frozen_decode_passes_through():
    f = Frozen(encode=lambda d: _msg(d, 1.0), decode=lambda z: z * 2)
    assert jnp.allclose(f.decode(jnp.array([1.0, 2.0])), jnp.array([2.0, 4.0]))


# ============================================================================
# LearnedLinear — SSID via CT-factor VMP
# ============================================================================

def test_learned_linear_recovers_transition_joint_mode():
    rng = np.random.RandomState(0)
    A = np.array([[0.95, 0.10], [-0.10, 0.92]])
    seqs = [_linear_traj(A, 80, 0.02, 0.01, rng) for _ in range(10)]
    block = LearnedLinear(dim=2, n_iterations=60)
    block.learn(seqs)
    assert np.allclose(np.array(block.A), A, atol=0.05)


def test_learned_linear_predict_advances_belief_by_A():
    rng = np.random.RandomState(1)
    A = np.array([[0.9, 0.0], [0.0, 0.8]])
    block = LearnedLinear(dim=2, n_iterations=60)
    block.learn([_linear_traj(A, 80, 0.02, 0.01, rng) for _ in range(10)])
    nxt = block.predict(_msg(jnp.array([1.0, 2.0]), 1e4))
    assert np.allclose(np.array(gaussian_mean(nxt)), np.array([0.9, 1.6]), atol=0.08)


def test_per_trajectory_mode_fits_last_trajectory():
    """`per_trajectory` ends fit to the last trajectory's data — distinct
    from the joint posterior when trajectories disagree."""
    rng = np.random.RandomState(0)
    A1 = np.array([[0.9, 0.0], [0.0, 0.9]])
    A2 = np.array([[0.5, 0.0], [0.0, 0.5]])
    seqs = ([_linear_traj(A1, 60, 0.02, 0.01, rng) for _ in range(5)] +
            [_linear_traj(A2, 60, 0.02, 0.01, rng) for _ in range(5)])
    joint = LearnedLinear(dim=2, n_iterations=40, mode="joint"); joint.learn(seqs)
    per = LearnedLinear(dim=2, n_iterations=40, mode="per_trajectory"); per.learn(seqs)
    assert np.linalg.norm(per.A - A2) < np.linalg.norm(joint.A - A2)


def test_learned_linear_rejects_unknown_mode():
    with pytest.raises(ValueError, match=r"mode must be"):
        LearnedLinear(dim=2, mode="bogus")


# ============================================================================
# LearnedAffine — fully-observed regression via the same CT-factor VMP
# ============================================================================

def test_learned_affine_recovers_A_and_B():
    rng = np.random.RandomState(5)
    A_true = np.array([[0.5, -0.3]])
    B_true = np.array([[0.7]])
    n = 400
    x = rng.randn(n, 2)
    u = rng.randn(n, 1)
    y = x @ A_true.T + u @ B_true.T + 0.01 * rng.randn(n, 1)
    reg = LearnedAffine(input_dim=2, output_dim=1, du=1, n_iterations=8).learn(
        [jnp.asarray(x)], [jnp.asarray(y)], [jnp.asarray(u)])
    assert np.allclose(np.array(reg.A), A_true, atol=0.02)
    assert np.allclose(np.array(reg.B), B_true, atol=0.02)


# ============================================================================
# KnownPhysics
# ============================================================================

def test_known_physics_predict_is_affine_step():
    A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.05]])
    offset = jnp.array([0.1, -0.2])
    block = KnownPhysics(2, du=1, linearize=lambda m: (A, B, offset))
    nxt = block.predict(_msg(jnp.array([0.3, -0.1]), 1e4), jnp.array([2.0]))
    expected = A @ jnp.array([0.3, -0.1]) + (B @ jnp.array([2.0])) + offset
    assert np.allclose(np.array(gaussian_mean(nxt)), np.array(expected), atol=1e-3)


# ============================================================================
# LearnedVAE — learnable observation; JointModel.learn becomes Variational EM
# ============================================================================

def test_learned_vae_em_reduces_loss():
    model = VAE(latent_dim=2, ch=8)
    rng = jax.random.PRNGKey(0)
    params = model.init({"params": rng}, jnp.ones((1, 28, 28)), rng)
    npr = np.random.RandomState(0)
    imgs = [(npr.rand(28, 28) < 0.15).astype(np.float32) for _ in range(20)]
    obs = LearnedVAE(model, params, lr=1e-3, n_m_steps=10)
    JointModel([Block("z", LearnedLinear(dim=2, n_iterations=5), observe=obs)]).learn(
        [{"z": imgs}], n_em=3)
    assert obs.learnable
    assert obs.loss_history[-1] < obs.loss_history[0]


# ============================================================================
# JointModel — smooth, plan, filter
# ============================================================================

def test_smooth_tracks_observations_and_predicts_by_A():
    rng = np.random.RandomState(2)
    A = np.array([[0.9, 0.1], [0.0, 0.9]])
    seqs = [_linear_traj(A, 40, 0.02, 0.01, rng) for _ in range(6)]
    raws = [[np.array(gaussian_mean(g)) for g in seq] for seq in seqs]
    block = Block("z", LearnedLinear(dim=2, n_iterations=40),
                  observe=lambda d: _msg(d, 1e4))
    model = JointModel([block])
    model.learn([{"z": r} for r in raws])
    out = model.smooth({"z": raws[0]}, n_predict=5)["z"]
    means, covs = np.array(out["means"]), np.array(out["covs"])
    assert means.shape == (45, 2) and covs.shape == (45, 2, 2)
    # high-precision obs → the smoothed means sit on the observations …
    assert np.max(np.abs(means[:40] - np.array(raws[0]))) < 0.05
    # … the n_predict tail rolls the latent forward by the learned A …
    A_hat = np.array(block.transition.A)
    assert np.allclose(means[41], A_hat @ means[40], atol=0.02)
    # … and predictive uncertainty grows once observations stop.
    assert np.trace(covs[44]) > np.trace(covs[39])


def test_plan_drives_double_integrator_to_goal():
    rng = np.random.RandomState(3)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    trajs = []
    for _ in range(15):
        x = rng.randn(2) * 0.5
        seq, us = [x.copy()], []
        for _ in range(39):
            u = rng.randn(1)
            us.append(u)
            x = A @ x + (B @ u).ravel() + 0.005 * rng.randn(2)
            seq.append(x.copy())
        trajs.append({"z": seq, "control": us})
    block = Block("z", LearnedLinear(dim=2, du=1, n_iterations=40),
                  observe=lambda d: _msg(d, 1e4))
    model = JointModel([block])
    model.learn(trajs)
    actions = model.plan({"z": [np.array([0.0, 0.0])] + [None] * 18 + [np.array([1.0, 0.0])]},
                         n_iterations=300)
    x = np.array([0.0, 0.0])
    for u in np.array(actions):
        x = A @ x + (B @ u).ravel()
    assert abs(x[0] - 1.0) < 0.2


def test_plan_with_known_physics_reaches_goal():
    """KnownPhysics planning (true linear dynamics) drives the state to the goal
    over the horizon — re-linearizing per call, returning (T-1, du) actions."""
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.1]])
    phys = KnownPhysics(2, du=1, linearize=lambda m: (A, B, jnp.zeros(2)))
    block = Block("z", phys, observe=lambda d: _msg(d, 1e3))
    obs = {"z": [np.array([0.0, 0.0])] + [None] * 18 + [np.array([1.0, 0.0])]}
    actions = JointModel([block]).plan(obs, prior_x=_msg([0.0, 0.0], 1e3), n_iterations=200)
    assert actions.shape == (19, 1)
    x = np.array([0.0, 0.0])
    for u in np.array(actions):
        x = np.array(A) @ x + np.array(B) @ u
    assert abs(x[0] - 1.0) < 0.3


def test_linear_coupling_sharpens_noisy_slice_via_clean_one():
    identity = lambda m: (jnp.eye(2), None, jnp.zeros(2))
    s = Block("s", KnownPhysics(2, linearize=identity), observe=lambda d: _msg(d, 1.0))
    z = Block("z", KnownPhysics(2, linearize=identity), observe=lambda d: _msg(d, 1e2))
    coupling = LinearCoupling("s", "z", M=jnp.eye(2), noise_prec=1e2)
    rng = np.random.RandomState(7)
    true_s = np.array([1.0, -0.5])
    stream = [{"s": true_s + rng.randn(2), "z": true_s + 0.1 * rng.randn(2)}
              for _ in range(5)]
    no = float(np.trace(np.linalg.inv(
        np.array(JointModel([s, z]).filter(stream)["s"].lam))))
    yes = float(np.trace(np.linalg.inv(
        np.array(JointModel([s, z], coupling=coupling).filter(stream)["s"].lam))))
    assert yes < 0.3 * no


def test_linear_coupling_fit_keeps_affine_offset():
    rng = np.random.RandomState(11)
    x = rng.randn(100, 2)
    M = np.array([[1.5, -0.25], [0.5, 2.0]])
    offset = np.array([0.75, -1.25])
    z = x @ M.T + offset + 0.01 * rng.randn(100, 2)
    coupling, diagnostics = LinearCoupling.fit("s", "z", x, z, ridge=1e-4)
    assert np.allclose(np.array(coupling.M), M, atol=0.03)
    assert np.allclose(np.array(coupling.offset), offset, atol=0.03)
    assert diagnostics["residual_rmse"] < 0.03


def test_linear_coupling_fit_keeps_ridge_separate_from_noise_floor():
    x = np.zeros((20, 2))
    z = np.broadcast_to(np.array([0.25, -0.5]), (20, 2))
    _, diagnostics = LinearCoupling.fit("s", "z", x, z, ridge=10.0)
    assert diagnostics["noise_var"] <= 1.1e-8
    assert diagnostics["residual_rmse"] == pytest.approx(0.0, abs=1e-8)


def test_linear_coupling_fit_rejects_nonpositive_noise_floor():
    x = np.zeros((2, 1))
    with pytest.raises(ValueError, match="noise_floor must be strictly positive"):
        LinearCoupling.fit("s", "z", x, x, noise_floor=0.0)


def test_linear_coupling_offset_affects_fusion_mean():
    coupling = LinearCoupling("s", "z", M=jnp.eye(2), offset=jnp.array([2.0, -1.0]), noise_prec=1e3)
    s_belief = _msg([0.0, 0.0], 1.0)
    z_belief = _msg([3.0, -2.0], 1e2)
    fused_s, fused_z = coupling.fuse(s_belief, z_belief)
    assert np.allclose(np.array(gaussian_mean(fused_s)), [1.0, -1.0], atol=0.05)
    assert np.allclose(np.array(gaussian_mean(fused_z)), [3.0, -2.0], atol=0.05)


def test_filter_composes_blocks_and_incorporates_observations():
    rng = np.random.RandomState(2)
    A_img = np.array([[0.95, 0.1], [-0.1, 0.95]])
    image = Block("z", LearnedLinear(dim=2, n_iterations=10),
                  observe=lambda d: _msg(d, 1e4))
    image.transition.learn([_linear_traj(A_img, 40, 0.02, 0.01, rng)])
    proprio = Block("p", KnownPhysics(2, linearize=lambda m: (jnp.eye(2), None, jnp.zeros(2))),
                    observe=lambda d: _msg(d, 1e4))
    beliefs = JointModel([proprio, image]).filter(
        [{"p": np.array([0.1, 0.0]), "z": np.array([1.0, 0.5])}])
    assert set(beliefs) == {"p", "z"}
    # high-precision observations → first-step beliefs sit on the observations.
    assert np.allclose(np.array(gaussian_mean(beliefs["z"])), [1.0, 0.5], atol=1e-2)
    assert np.allclose(np.array(gaussian_mean(beliefs["p"])), [0.1, 0.0], atol=1e-2)


def test_vae_supports_64px_frames():
    """img_size=64 encodes/decodes at the right shapes; 28 stays the default."""
    model = VAE(latent_dim=3, ch=8, img_size=64)
    rng = jax.random.PRNGKey(0)
    params = model.init({"params": rng}, jnp.ones((1, 64, 64)), rng)
    mu, log_std = model.apply(params, jnp.ones((2, 64, 64)), method=model.encode)
    recon = model.apply(params, mu, method=model.decode)
    assert mu.shape == (2, 3) and log_std.shape == (2, 3)
    assert recon.shape == (2, 64 * 64)


def test_learned_delay_linear_preserves_shift_and_learns_controlled_top_block():
    rng = np.random.RandomState(9)
    h, delay, du = 2, 3, 1
    B_true = np.array([[0.25], [-0.15]])
    state_seqs, control_seqs = [], []
    for _ in range(8):
        features = [rng.randn(h) * 0.2, rng.randn(h) * 0.2]
        controls = []
        for _ in range(48):
            u = rng.uniform(-1, 1, du)
            nxt = 1.65 * features[-1] - 0.7 * features[-2] + B_true @ u
            nxt += 0.005 * rng.randn(h)
            features.append(nxt); controls.append(u)
        z = [np.concatenate(features[t:t + delay]) for t in range(len(features) - delay + 1)]
        state_seqs.append(np.asarray(z))
        # Two feature states were seeded before controls[0] generated h[2];
        # the first window transition h[0:3] -> h[1:4] therefore uses u[1].
        control_seqs.append(controls[1:])

    d = h * delay
    expected_shift = np.zeros((d - h, d))
    expected_shift[:, h:] = np.eye(d - h)
    tr = LearnedDelayLinear(h, delay=delay, du=du, offset=False,
                            n_iterations=10, shift_cov=1e-7)
    tr.learn_observed(state_seqs, control_seqs, obs_prec=1e4)
    assert np.all(np.isfinite(np.asarray(tr.A)))
    assert np.allclose(np.asarray(tr.A[:d - h]), expected_shift, atol=1e-5)
    assert np.max(np.abs(np.asarray(tr.B[:d - h]))) < 1e-5
    assert np.allclose(np.asarray(tr.B[d - h:]), B_true, atol=0.08)
    tr.attach_replay(state_seqs, control_seqs, neighbors=64,
                     refresh_every=1, obs_prec=1e4, n_iterations=6)
    assert tr.localize(state_seqs[0][10])
    assert np.all(np.isfinite(np.asarray(tr.B)))


def test_exact_plan_matches_lq_in_deterministic_limit():
    """The exact Gaussian action posterior reduces to the LQ solution when the
    transition is near-deterministic (LQG duality sanity check)."""
    from jopa.blocks import _identity_meta
    from jopa.message_passing import (
        infer_actions_exact,
        infer_actions_exact_numpy,
        infer_actions_exact_posterior,
    )

    rng = np.random.RandomState(0)
    d, du, H = 3, 2, 6
    A = np.eye(d) + 0.1 * rng.randn(d, d)
    B = 0.5 * rng.randn(d, du)
    W = 1e6 * np.eye(d)                       # near-deterministic
    x0 = rng.randn(d)
    zg = rng.randn(d)
    reg, pg = 1e-2, 1.0

    q_a = Gaussian(eta=1e9 * jnp.asarray(A.ravel()), lam=1e9 * jnp.eye(d * d))
    q_W = Wishart(df=1e6, inv_scale=1e6 * jnp.linalg.inv(jnp.asarray(W)))
    q_b = Gaussian(eta=1e9 * jnp.asarray(B.ravel()), lam=1e9 * jnp.eye(d * du))
    cache = CTCache(q_a, q_W, _identity_meta(d), q_b)

    prior_x = Gaussian(eta=1e9 * jnp.asarray(x0), lam=1e9 * jnp.eye(d))
    prior_u = Gaussian(eta=jnp.zeros(du), lam=reg * pg * jnp.eye(du))
    vague = Gaussian(eta=jnp.zeros(d), lam=jnp.zeros((d, d)))
    goal = Gaussian(eta=pg * jnp.asarray(zg), lam=pg * jnp.eye(d))
    observations = [vague] * H + [goal]
    acts = np.array(infer_actions_exact(prior_x, observations, cache, prior_u))
    posterior = infer_actions_exact_posterior(
        prior_x, observations, cache, prior_u)
    reference = np.array(
        infer_actions_exact_numpy(prior_x, observations, cache, prior_u))
    # GPU and LAPACK solve the deliberately ill-conditioned (W=1e6) system
    # with slightly different elimination kernels.
    np.testing.assert_allclose(acts, reference, rtol=5e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(posterior.mean), acts, rtol=5e-6, atol=1e-6)
    covariance = np.asarray(posterior.joint_covariance)
    assert covariance.shape == (H * du, H * du)
    assert np.linalg.eigvalsh(covariance).min() > -1e-9

    # closed-form LQ: minimise |x_H - zg|^2 + reg |u|^2 under x' = A x + B u
    M = np.zeros((d, du * H))
    for k in range(H):
        M[:, du * k:du * k + du] = np.linalg.matrix_power(A, H - 1 - k) @ B
    b = zg - np.linalg.matrix_power(A, H) @ x0
    u_lq = np.linalg.solve(M.T @ M + reg * np.eye(du * H), M.T @ b).reshape(H, du)
    assert np.abs(acts - u_lq).max() < 1e-3


def test_plan_method_vmp_still_reaches():
    """The iterative VMP planner stays available via method='vmp'."""
    rng = np.random.RandomState(1)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    trajs = []
    for _ in range(15):
        x = rng.randn(2) * 0.5
        seq, us = [x.copy()], []
        for _ in range(39):
            u = rng.randn(1)
            us.append(u)
            x = A @ x + (B @ u).ravel() + 0.005 * rng.randn(2)
            seq.append(x.copy())
        trajs.append({"z": seq, "control": us})
    block = Block("z", LearnedLinear(dim=2, du=1, n_iterations=40),
                  observe=lambda d: _msg(d, 1e4))
    model = JointModel([block])
    model.learn(trajs)
    obs = {"z": [np.array([0.0, 0.0])] + [None] * 18 + [np.array([1.0, 0.0])]}
    for method in ("exact", "vmp"):
        actions = model.plan(obs, n_iterations=300, method=method)
        x = np.array([0.0, 0.0])
        for u in np.array(actions):
            x = A @ x + (B @ u).ravel()
        assert abs(x[0] - 1.0) < 0.2, method
