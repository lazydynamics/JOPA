"""Factor-graph inference checked against dense references.

Sum-product by elimination must reproduce the dense Gaussian posterior on
an affine graph for any ordering; iterated linearization must reach the
mode of a nonlinear one; the neural-linear node's conjugate messages must
equal the closed-form ridge posterior; and the actuator node's ContinuousTransition
messages must identify a reversed actuator from transitions alone.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jopa import Actuator, Gaussian, MatrixNormal, NeuralLinear, Planner, fg
from jopa.nodes import apply_matrix

REVERSED = jnp.diag(jnp.array([-1.0, 1.0]))

DIMS = {"a": 2, "b": 3, "c": 1, "d": 2}


def _affine_graph(seed=0, order=None):
    """Chain a–b–c–d of random affine Normal nodes, plus a prior on a."""
    rng = np.random.default_rng(seed)
    V = {n: fg.Variable(n, k) for n, k in DIMS.items()}
    nodes = [fg.Normal(V["a"], jnp.asarray(rng.normal(size=2)), 2.0)]
    for x, y in [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]:
        M = jnp.asarray(rng.normal(size=(DIMS[y], DIMS[x])))
        c = jnp.asarray(rng.normal(size=DIMS[y]))
        P = jnp.asarray(rng.uniform(0.5, 2.0, size=DIMS[y]))
        nodes.append(fg.Normal(V[y], fg.Fn(lambda s, M=M, c=c: M @ s + c, V[x]), P))
    return fg.Graph(nodes, order=order)


def _dense(graph):
    """Mean and covariance from the full joint precision."""
    names = list(graph.dims)
    z0 = {n: jnp.zeros(graph.dims[n]) for n in names}
    pots = [n.linearize(z0) for n in graph.nodes]
    eta, lam = fg.graph._scatter(pots, tuple(names), graph.dims)
    cov = np.linalg.inv(np.asarray(lam, np.float64))
    mean = cov @ np.asarray(eta, np.float64)
    out, o = {}, 0
    for n in names:
        k = graph.dims[n]
        out[n] = (mean[o:o + k], cov[o:o + k, o:o + k])
        o += k
    return out


@pytest.mark.parametrize("order", [None, ("d", "c", "b", "a"), ("b", "d", "a", "c")])
def test_elimination_matches_dense_for_any_order(order):
    graph = _affine_graph(order=order)
    ref = _dense(_affine_graph())
    post = graph.infer({n: jnp.zeros(k) for n, k in DIMS.items()}, iterations=1)
    for n in DIMS:
        np.testing.assert_allclose(post.mean[n], ref[n][0], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(post.cov[n], ref[n][1], rtol=1e-3, atol=1e-4)


def test_chain_elimination_equals_general_elimination():
    """The scanned chain sweep is `eliminate` in time order: same means,
    covariances and cross-covariances."""
    rng = np.random.default_rng(0)
    T, d = 7, 3

    def spd(n):
        M = rng.normal(size=(n, n))
        return jnp.asarray(M @ M.T + n * np.eye(n), jnp.float32)

    eta, lam = jnp.asarray(rng.normal(size=(T, d)), jnp.float32), jnp.stack([spd(d) for _ in range(T)])
    pair_eta = jnp.asarray(rng.normal(size=(T - 1, 2 * d)), jnp.float32)
    pair_lam = jnp.stack([spd(2 * d) for _ in range(T - 1)])
    names = tuple(f"x{t}" for t in range(T))
    pots = [fg.Potential((n,), eta[t], lam[t]) for t, n in enumerate(names)]
    pots += [fg.Potential((names[t + 1], names[t]), pair_eta[t], pair_lam[t]) for t in range(T - 1)]
    pairs = tuple(itertools.pairwise(names))
    mean, cov = fg.eliminate(pots, names, dict.fromkeys(names, d), pairs=pairs)
    m, S, C = fg.eliminate_chain(eta, lam, pair_eta, pair_lam)
    for t, n in enumerate(names):
        np.testing.assert_allclose(m[t], mean[n], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(S[t], cov[n], rtol=1e-4, atol=1e-5)
    for t, p in enumerate(pairs):
        np.testing.assert_allclose(C[t], cov[p], rtol=1e-4, atol=1e-5)


def test_expression_arithmetic_builds_delta_nodes():
    a, b = fg.Variable("a", 2), fg.Variable("b", 2)
    values = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 5.0])}
    expr = 2.0 * a + b - 1.0 + (1.0 - a) * b + (-a) - b
    expected = 2 * values["a"] + values["b"] - 1 + (1 - values["a"]) * values["b"] - values["a"] - values["b"]
    np.testing.assert_allclose(fg.evaluate(expr, values), expected)
    assert [v.name for v in fg.variables(expr)] == ["a", "b"]
    with pytest.raises(ValueError, match="dims"):
        fg.variables(a + fg.Variable("a", 3))


def test_iterated_linearization_reaches_nonlinear_mode():
    v = fg.Variable("v", 1)
    graph = fg.Graph([fg.Normal(v, 0.0, 1e-4), fg.Normal(fg.Fn(jnp.tanh, v), 0.5, 1e4)])
    post = graph.infer({"v": jnp.zeros(1)}, iterations=20)
    np.testing.assert_allclose(post.mean["v"], np.arctanh(0.5), rtol=1e-3)
    # Laplace covariance: 1 / (P · tanh'(v)²) at the mode
    np.testing.assert_allclose(post.cov["v"][0, 0], 1 / (1e4 * 0.75 ** 2 + 1e-4), rtol=1e-2)


def test_line_search_never_increases_energy():
    v = fg.Variable("v", 1)
    graph = fg.Graph([fg.Normal(fg.Fn(lambda x: jnp.sin(3 * x), v), 0.9, 1.0)])
    init = {"v": jnp.array([2.0])}
    assert graph.infer(init, iterations=5).energy <= graph.energy(init)


def test_restarts_keep_lowest_energy():
    v = fg.Variable("v", 1)    # two modes: v = ±1, the prior favours +1
    graph = fg.Graph([fg.Normal(fg.Fn(lambda x: x ** 2, v), 1.0, 1e2), fg.Normal(v, 1.0, 1e-2)])
    post = graph.infer_restarts({"v": jnp.array([[-0.8], [0.8]])}, iterations=20)
    np.testing.assert_allclose(post.mean["v"], 1.0, rtol=1e-3)


def _toy_model(seed=0, d=3, du=4, m=6):
    """Random tanh features with a bias; W from a known matrix-normal prior."""
    k1, k2 = jax.random.split(jax.random.key(seed))
    params = {"A": jax.random.normal(k1, (2 * d + du, m - 1)) / 2}
    def features(p, x):
        return jnp.concatenate(
            [jnp.tanh(x @ p["A"]), jnp.ones_like(x[..., :1])], -1)
    W = jax.random.normal(k2, (m, d)) * 0.3
    return NeuralLinear(features, params, MatrixNormal.from_mean(W, 1e3), 1e-4), W


def _transitions(model, n, A=None, seed=1):
    keys = jax.random.split(jax.random.key(seed), 3)
    d = model.weight_mean.shape[1]
    prev, cur = (jax.random.normal(k, (n, d)) for k in keys[:2])
    u = jax.random.uniform(keys[2], (n, 4), minval=-1, maxval=1)
    u_eff = u if A is None else apply_matrix(u, A)
    return (prev, cur), u, model.mean((prev, cur), u_eff)


def test_weight_posterior_is_ridge_solution():
    model, W0 = _toy_model()
    states, u, nxt = _transitions(model, 50)
    post = model.condition(states, u, nxt)
    F, Y = model.phi(states, u), nxt - states[-1]
    lam = 1e3 * np.eye(F.shape[1]) + F.T @ F
    ref = np.linalg.solve(lam, 1e3 * W0 + F.T @ Y)
    np.testing.assert_allclose(post.weight_mean, ref, rtol=1e-3, atol=1e-4)
    # one batch or two halves: the product of messages is the same
    a = model.condition(*(jax.tree.map(lambda x: x[:20], (states, u, nxt))))
    b = a.condition(*(jax.tree.map(lambda x: x[20:], (states, u, nxt))))
    np.testing.assert_allclose(b.weight_mean, post.weight_mean, rtol=1e-3, atol=1e-4)


def test_actuator_identifies_a_reversed_actuator():
    """The CT :a and :W messages recover A = diag(−1, 1); evidence carries across calls."""
    model, _ = _toy_model()
    states, u, nxt = _transitions(model, 40, A=REVERSED)
    act = Actuator.nominal(2, noise_var=model.noise_var)
    for half in (slice(0, 20), slice(20, 40)):
        act = act.update(model, tuple(s[half] for s in states), u[half], nxt[half])
    np.testing.assert_allclose(act.mean, REVERSED, atol=1e-2)


def test_plan_reaches_reachable_goal():
    model, _ = _toy_model()
    history = (jnp.zeros(3), jnp.zeros(3))
    goal = model.mean(history, jnp.array([0.5, -0.3, 0.2, 0.1]))
    u, _ = Planner(horizon=1, action_precision=1e-6)(model, history, goal, jax.random.key(0), 4)
    np.testing.assert_allclose(model.mean(history, u[0]), goal, atol=1e-2)


def test_messages_roundtrip_and_multiply():
    W = jnp.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(MatrixNormal.from_mean(W, 5.0).mean, W, rtol=1e-6)
    a, b = Gaussian.from_mean(jnp.ones(2), 1.0), Gaussian.from_mean(-jnp.ones(2), 3.0)
    np.testing.assert_allclose((a * b).mean, -0.5 * jnp.ones(2), rtol=1e-6)


def test_zero_weight_transitions_send_no_message():
    model, _ = _toy_model()
    states, u, nxt = _transitions(model, 20, A=REVERSED)
    junk = jax.tree.map(lambda x: jnp.concatenate([x, 5 + x[:5]]), (states, u, nxt))
    w = jnp.arange(25) < 20
    ref = Actuator.nominal(2).update(model, states, u, nxt)
    masked = Actuator.nominal(2).update(model, *junk, weight=w)
    np.testing.assert_allclose(masked.mean, ref.mean, atol=1e-4)
    np.testing.assert_allclose(masked.q_w.inv_scale, ref.q_w.inv_scale, rtol=1e-4)


def test_box_keeps_estimate_feasible_at_constrained_mode():
    u = fg.Variable("u", 2)    # unconstrained mode (2, −0.3); the box clips the first
    graph = fg.Graph([fg.Normal(u, jnp.array([2.0, -0.3]), 1.0), fg.Box(u, -1.0, 1.0)])
    post = graph.infer({"u": jnp.zeros(2)}, iterations=5)
    np.testing.assert_allclose(post.mean["u"], [1.0, -0.3], atol=1e-5)


def test_plan_respects_actuator_limits():
    model, _ = _toy_model()
    history = (jnp.zeros(3), jnp.zeros(3))
    goal = model.mean(history, jnp.array([3.0, -3.0, 3.0, 3.0]))     # needs |u| > 1
    u, _ = Planner(horizon=2)(model, history, goal, jax.random.key(0), 4)
    assert jnp.all(jnp.abs(u) <= 1.0)


def test_planning_through_a_reversed_actuator():
    model, _ = _toy_model()
    history = (jnp.zeros(3), jnp.zeros(3))
    target = jnp.array([0.5, -0.3, 0.2, 0.1])
    act = Actuator(Gaussian.from_mean(REVERSED.ravel(), 1e4), Actuator.nominal(2).q_w)
    goal = model.mean(history, apply_matrix(target, REVERSED))
    u, _ = Planner(horizon=1, action_precision=1e-6)(model, history, goal, jax.random.key(0),
                                                     4, actuator=act)
    np.testing.assert_allclose(model.mean(history, apply_matrix(u[0], REVERSED)), goal, atol=1e-2)
    assert not np.allclose(model.mean(history, u[0]), goal, atol=1e-2)


def test_planner_compiles_once_for_any_model_and_actuator():
    model, _ = _toy_model()
    plan = jax.jit(lambda m, g, x, goal, k: Planner()(m, (x, x), goal, k, 4, actuator=g))
    x = jnp.zeros(3)
    plan(model, Actuator.nominal(2), x, x, jax.random.key(0))
    other = model.condition(*_transitions(model, 10))
    plan(other, Actuator.nominal(2, 0.5), x + 1, x, jax.random.key(1))
    assert plan._cache_size() == 1


def test_fit_neural_linear_learns_a_nonlinear_plant():
    from jopa import fit_neural_linear
    from jopa.learn import windows
    rng = np.random.default_rng(0)
    trajs = []
    for _ in range(20):
        x, u = [np.zeros(2), np.zeros(2)], rng.uniform(-1, 1, (40, 2))
        for t in range(1, 40):
            x.append(x[-1] + 0.5 * (x[-1] - x[-2]) + 0.1 * np.tanh(u[t] - x[-1]))
        trajs.append((np.stack(x), u))
    model = fit_neural_linear(trajs, width=32, horizon=3, steps=1500, prior_precision=1e2)
    f, a = windows(trajs, model.order, 3)
    err = model.rollout((f[:, 0], f[:, 1]), a) - f[:, 2:]
    persist = f[:, 1:2] - f[:, 2:]
    assert jnp.mean(err ** 2) < 0.05 * jnp.mean(persist ** 2)


def test_reversed_actuator_example():
    from examples.reversed_actuator import run
    frozen, *_ = run(adapt=False)
    adaptive, act, _ = run(adapt=True)
    assert adaptive < 0.05 and frozen > 1.0
    np.testing.assert_allclose(act.mean, REVERSED, atol=0.1)

