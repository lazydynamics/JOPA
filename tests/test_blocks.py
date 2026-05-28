"""Spec for the modular joint-state blocks (TDD — written before the impl).

A `Block` is a slice of the joint state with a `transition` (how it evolves) and
an `observe` map (raw datum → Gaussian message on the slice). `JointModel`
composes blocks; learned blocks learn their dynamics via the CT-node VMP, known
blocks just filter. Designed so adding a modality = appending a `Block`.
"""
import numpy as np
import jax.numpy as jnp

import jax

from jopa.distributions import Gaussian, gaussian_mean
from jopa.blocks import (
    Block, KnownPhysics, LearnedLinear, JointModel, Observation, Frozen, LearnedVAE, LinearCoupling,
)
from jopa.nn.vae import VAE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _msg(mean, prec):
    """Gaussian observation message at `mean` with isotropic precision."""
    mean = jnp.asarray(mean, dtype=float)
    lam = prec * jnp.eye(mean.shape[0])
    return Gaussian(eta=lam @ mean, lam=lam)


def _linear_traj(A, T, obs_std, proc_std, rng):
    """One trajectory of noisy observation messages from x' = A x + noise."""
    d = A.shape[0]
    x = rng.randn(d)
    seq = []
    for _ in range(T):
        seq.append(_msg(x + obs_std * rng.randn(d), 1.0 / obs_std ** 2))
        x = A @ x + proc_std * rng.randn(d)
    return seq


# ---------------------------------------------------------------------------
# Observation — Frozen now; LearnedVAE in B2
# ---------------------------------------------------------------------------

def test_observation_frozen_and_autowrap():
    f = Frozen(lambda d: _msg(d, 1e4))
    assert f.learnable is False
    assert f.message(jnp.array([1.0, 2.0])).eta.shape == (2,)

    # a bare callable passed to a Block is auto-wrapped into an Observation
    blk = Block("x", LearnedLinear(2), observe=lambda d: _msg(d, 1e4))
    assert isinstance(blk.observation, Observation)
    assert blk.observation.learnable is False


# ---------------------------------------------------------------------------
# LearnedLinear — learns its dynamics via the CT node
# ---------------------------------------------------------------------------

def test_learned_linear_recovers_transition():
    rng = np.random.RandomState(0)
    A = np.array([[0.95, 0.10], [-0.10, 0.92]])
    seqs = [_linear_traj(A, 80, 0.02, 0.01, rng) for _ in range(10)]

    block = LearnedLinear(dim=2, n_iterations=60)
    block.learn(seqs)

    assert np.allclose(np.array(block.A), A, atol=0.08), np.array(block.A)


def test_learned_linear_predict_advances_belief():
    rng = np.random.RandomState(1)
    A = np.array([[0.9, 0.0], [0.0, 0.8]])
    seqs = [_linear_traj(A, 80, 0.02, 0.01, rng) for _ in range(10)]
    block = LearnedLinear(dim=2, n_iterations=60)
    block.learn(seqs)

    x0 = jnp.array([1.0, 2.0])
    nxt = block.predict(_msg(x0, 1e4))
    assert np.allclose(np.array(gaussian_mean(nxt)), np.array(A @ x0), atol=0.1)


# ---------------------------------------------------------------------------
# LearnedVAE — learnable observation; JointModel.learn becomes Variational EM
# ---------------------------------------------------------------------------

def test_learned_vae_em_reduces_loss():
    model = VAE(latent_dim=2, ch=8)
    rng = jax.random.PRNGKey(0)
    params = model.init({"params": rng}, jnp.ones((1, 28, 28)), rng)

    npr = np.random.RandomState(0)
    imgs = [(npr.rand(28, 28) < 0.15).astype(np.float32) for _ in range(20)]

    obs = LearnedVAE(model, params, lr=1e-3, n_m_steps=10)
    block = Block("z", LearnedLinear(dim=2, n_iterations=5), observe=obs)
    JointModel([block]).learn([{"z": imgs}], n_em=3)

    assert obs.learnable is True
    assert obs.loss_history[-1] < obs.loss_history[0]      # EM reduced the M-step loss


# ---------------------------------------------------------------------------
# KnownPhysics — no learning, predicts via a supplied linearization
# ---------------------------------------------------------------------------

def test_known_physics_predict_matches_linear_step():
    A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.05]])
    offset = jnp.array([0.1, -0.2])

    block = KnownPhysics(dim=2, du=1, linearize=lambda mean: (A, B, offset))
    x0 = jnp.array([0.3, -0.1])
    u = jnp.array([2.0])
    nxt = block.predict(_msg(x0, 1e4), u)
    assert np.allclose(np.array(gaussian_mean(nxt)), np.array(A @ x0 + B @ u + offset), atol=1e-3)

    assert block.learned is False


# ---------------------------------------------------------------------------
# Coupling — fuses information across blocks (the multimodal point)
# ---------------------------------------------------------------------------

def test_coupling_sharpens_one_slice_with_the_other():
    """A clean observation on slice `z` should sharpen the noisy slice `s`
    when a linear coupling z = M·s + ε ties them."""
    identity = lambda m: (jnp.eye(2), None, jnp.zeros(2))   # static dynamics
    s_block = Block("s", KnownPhysics(2, du=0, linearize=identity),
                    observe=lambda d: _msg(d, 1.0))                 # noisy obs   (std 1.0)
    z_block = Block("z", KnownPhysics(2, du=0, linearize=identity),
                    observe=lambda d: _msg(d, 1e2))                 # clean obs   (std 0.1)
    coupling = LinearCoupling("s", "z", M=jnp.eye(2), noise_prec=1e2)

    rng = np.random.RandomState(7)
    true_s = np.array([1.0, -0.5])
    stream = [
        {"s": true_s + rng.randn(2) * 1.0, "z": true_s + rng.randn(2) * 0.1}
        for _ in range(5)
    ]

    s_var_no = float(np.trace(np.linalg.inv(
        np.array(JointModel([s_block, z_block]).filter(stream)["s"].lam))))
    s_var_yes = float(np.trace(np.linalg.inv(
        np.array(JointModel([s_block, z_block], coupling=coupling).filter(stream)["s"].lam))))

    assert s_var_yes < 0.3 * s_var_no, (s_var_no, s_var_yes)


# ---------------------------------------------------------------------------
# JointModel.plan — planning-as-inference on the composed model (B3)
# ---------------------------------------------------------------------------

def test_joint_model_plans_to_goal():
    rng = np.random.RandomState(3)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])     # position/velocity double integrator
    B = np.array([[0.0], [0.1]])               # control on velocity → controllable

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
    assert np.allclose(np.array(block.transition.A), A, atol=0.1), np.array(block.transition.A)

    start, goal = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    H = 20
    actions = model.plan({"z": [start] + [None] * (H - 2) + [goal]}, n_iterations=300)

    x = start.copy()                            # roll planned actions through TRUE dynamics
    for u in np.array(actions):
        x = A @ x + (B @ u).ravel()
    assert abs(x[0] - goal[0]) < 0.2, x


# ---------------------------------------------------------------------------
# JointModel — compose blocks; learn the learned ones; filter the joint
# ---------------------------------------------------------------------------

def test_joint_model_learns_and_filters():
    rng = np.random.RandomState(2)
    A_img = np.array([[0.95, 0.1], [-0.1, 0.95]])

    # image block: learned linear dynamics, observed via identity "encoder".
    # proprio block: known identity-ish physics, direct observation.
    A_pro = jnp.array([[1.0, 0.05], [0.0, 1.0]])
    B_pro = jnp.zeros((2, 1))
    proprio = Block("proprio", KnownPhysics(2, du=1, linearize=lambda m: (A_pro, B_pro, jnp.zeros(2))),
                    observe=lambda d: _msg(d, 1e4))
    image = Block("image", LearnedLinear(dim=2, n_iterations=60),
                  observe=lambda d: _msg(d, 1.0 / 0.02 ** 2))

    model = JointModel([proprio, image])

    # training trajectories: per block, a sequence of raw observations.
    trajs = []
    for _ in range(10):
        seq = _linear_traj(A_img, 80, 0.02, 0.01, rng)
        img_raw = [np.array(gaussian_mean(g)) for g in seq]            # raw "images"
        pro_raw = [np.array([0.0, 0.0]) for _ in img_raw]              # dummy proprio
        trajs.append({"image": img_raw, "proprio": pro_raw})
    model.learn(trajs)

    assert np.allclose(np.array(model["image"].transition.A), A_img, atol=0.1)

    # filtering a 1-step stream returns a belief per block.
    stream = [{"image": np.array([1.0, 0.5]), "proprio": np.array([0.2, -0.1])}]
    beliefs = model.filter(stream)
    assert set(beliefs) == {"proprio", "image"}
    assert beliefs["image"].eta.shape == (2,)
