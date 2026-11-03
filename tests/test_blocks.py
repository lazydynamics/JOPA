"""Semantic tests for the JOPA API: distributions, message passing, blocks, planning."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jopa import (
    Block,
    Frozen,
    JointModel,
    KnownPhysics,
    LearnedLinear,
    LearnedVAE,
    LinearCoupling,
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
from jopa.nn import VAE
from jopa.nodes.transition import CTCache, CTMeta
from jopa.observations import Observation
from jopa.transitions import _chain

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

def test_chain_sweep_recovers_true_states():
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

    stacked = Gaussian(jnp.stack([g.eta for g in obs]), jnp.stack([g.lam for g in obs]))
    means = np.array(_chain(stacked, cache, None)[0])
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


def test_smooth_rollout_leaves_observed_estimates_unchanged():
    """Forecast slices must not feed back into the observed ones: with a
    loose q(A) on a non-centred latent, VMP over unobserved slices would pull
    the smoothed estimates toward the origin."""
    rng = np.random.RandomState(5)
    A = np.array([[0.98, 0.05], [-0.05, 0.98]])
    seqs = [[_msg(3.0 + 0.1 * rng.randn(2), 25.0) for _ in range(6)]]
    raws = [[np.array(gaussian_mean(g)) for g in seq] for seq in seqs]
    block = Block("z", LearnedLinear(dim=2, n_iterations=10, prior_a_mean=A.ravel(),
                                     prior_a_cov=1e-2),
                  observe=lambda d: _msg(d, 25.0))
    model = JointModel([block])
    model.learn([{"z": r} for r in raws])
    alone = np.array(model.smooth({"z": raws[0]})["z"]["means"])
    rolled = np.array(model.smooth({"z": raws[0]}, n_predict=8)["z"]["means"])
    assert np.allclose(rolled[:6], alone, atol=1e-5)


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
    actions = model.plan({"z": [np.array([0.0, 0.0])] + [None] * 18 + [np.array([1.0, 0.0])]})
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
    actions = JointModel([block]).plan(obs, prior_x=_msg([0.0, 0.0], 1e3))
    assert actions.shape == (19, 1)
    x = np.array([0.0, 0.0])
    for u in np.array(actions):
        x = np.array(A) @ x + np.array(B) @ u
    assert abs(x[0] - 1.0) < 0.3


def test_linear_coupling_sharpens_noisy_slice_via_clean_one():
    def identity(m):
        return (jnp.eye(2), None, jnp.zeros(2))
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


def test_plan_is_the_exact_chain_posterior():
    """`JointModel.plan` equals an independent float64 solve of the [x; u]
    chain, and the LQ solution when the transition is near-deterministic
    (LQG duality)."""
    rng = np.random.RandomState(0)
    d, du, H = 3, 2, 6
    A = np.eye(d) + 0.1 * rng.randn(d, d)
    B = 0.5 * rng.randn(d, du)
    x0, zg = rng.randn(d), rng.randn(d)
    reg, pg = 1e-2, 1.0

    phys = KnownPhysics(d, du=du, linearize=lambda m: (jnp.asarray(A), jnp.asarray(B),
                                                       jnp.zeros(d)), process_std=1e-2)
    block = Block("z", phys, observe=lambda y: _msg(y, pg))
    prior_x = _msg(x0, 1e4)
    prior_u = Gaussian(eta=jnp.zeros(du), lam=reg * pg * jnp.eye(du))
    acts = np.array(JointModel([block]).plan({"z": [None] * H + [zg]},
                                             prior_x=prior_x, prior_u=prior_u))
    vague = Gaussian(eta=jnp.zeros(d), lam=jnp.zeros((d, d)))
    reference = _reference_plan(prior_x, [vague] * H + [_msg(zg, pg)], A, B,
                                np.linalg.inv(np.asarray(phys.Q)), prior_u)
    np.testing.assert_allclose(acts, reference, rtol=1e-5, atol=1e-5)

    # closed-form LQ: minimise |x_H - zg|^2 + reg |u|^2 under x' = A x + B u
    M = np.zeros((d, du * H))
    for k in range(H):
        M[:, du * k:du * k + du] = np.linalg.matrix_power(A, H - 1 - k) @ B
    b = zg - np.linalg.matrix_power(A, H) @ x0
    u_lq = np.linalg.solve(M.T @ M + reg * np.eye(du * H), M.T @ b).reshape(H, du)
    assert np.abs(acts - u_lq).max() < 1e-2


def _reference_plan(prior_x, obs_list, A, B, W, prior_u):
    """Independent NumPy float64 solve of the plan: the block-tridiagonal
    information form of the [x; u] chain, eliminated forward and
    back-substituted."""
    d, du = B.shape
    ds = d + du
    T = len(obs_list)
    def f64(a):
        return np.asarray(a, dtype=np.float64)
    A, B, W = f64(A), f64(B), f64(W)
    AtW, BtW = A.T @ W, B.T @ W

    # per-slice information: D[t] (ds×ds) on s[t], coupling C[t] to s[t+1]
    D = np.zeros((T, ds, ds))
    e = np.zeros((T, ds))
    C = np.zeros((T - 1, ds, ds))
    for t, ob in enumerate(obs_list):
        D[t, :d, :d] += f64(ob.lam)
        e[t, :d] += f64(ob.eta)
        D[t, d:, d:] += f64(prior_u.lam)
        e[t, d:] += f64(prior_u.eta)
    D[0, :d, :d] += f64(prior_x.lam)
    e[0, :d] += f64(prior_x.eta)
    for t in range(T - 1):
        # transition factor: quadratic of  x[t+1] − A x[t] − B u[t]  in W
        D[t, :d, :d] += AtW @ A
        D[t, d:, d:] += BtW @ B
        D[t, :d, d:] += AtW @ B
        D[t, d:, :d] += (AtW @ B).T
        D[t + 1, :d, :d] += W
        C[t, :d, :d] = -AtW          # rows s[t], cols s[t+1] (x-part only)
        C[t, d:, :d] = -BtW

    # α sweep: forward elimination of the block-tridiagonal information form
    Dt, et = D.copy(), e.copy()
    G = np.zeros_like(C)
    for t in range(T - 1):
        G[t] = np.linalg.solve(Dt[t], C[t])
        Dt[t + 1] -= C[t].T @ G[t]
        et[t + 1] -= C[t].T @ np.linalg.solve(Dt[t], et[t])
    # β sweep: back substitution → posterior means of every slice
    mu = np.zeros((T, ds))
    mu[T - 1] = np.linalg.solve(Dt[T - 1], et[T - 1])
    for t in range(T - 2, -1, -1):
        mu[t] = np.linalg.solve(Dt[t], et[t] - C[t] @ mu[t + 1])
    return mu[:-1, d:]


def test_vae_trains_saves_and_loads(tmp_path):
    """`train_vae` lowers the ELBO; a checkpoint round-trips, and one written
    for another latent size is rejected."""
    from jopa.nn import load_params, save_params, train_vae

    rng = np.random.RandomState(0)
    images = (rng.rand(16, 28, 28) < 0.2).astype(np.float32)
    losses = []
    model, params = train_vae(images, latent_dim=2, ch=4, epochs=3, batch_size=8,
                              verbose=False, callback=lambda e, p, loss: losses.append(loss))
    assert losses[-1] < losses[0]
    path = tmp_path / "vae.msgpack"
    save_params(params, path)
    loaded = load_params(model, path)
    assert all(np.array_equal(a, b) for a, b in
               zip(jax.tree.leaves(loaded), jax.tree.leaves(params), strict=True))
    with pytest.raises(ValueError, match="does not match"):
        load_params(VAE(latent_dim=3, ch=4), path)
    with pytest.raises(ValueError, match="28"):
        train_vae(np.zeros((2, 14, 14)), epochs=1, verbose=False)
