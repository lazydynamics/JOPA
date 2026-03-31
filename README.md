# 🍑 JOPA

**Joint Observation-Planning Architecture** — the same factor graph for learning and planning.

Or: Poor Man's Active Inference via message passing.

---

Closed-loop control from pixels with unknown dynamics and unknown observations. Dynamics learned via message passing, observation model via gradient descent. Takes raw images and control inputs, learns a latent dynamical system via Bayesian inference, and plans actions by running inference on the same factor graph. No reward functions, no policy networks, no replay buffers.

> **How this was built.** This codebase was largely written by [Claude Code](https://claude.ai/code). The only component Claude couldn't derive was the message-passing rules for the Continuous Transition node — those were provided as hand-derived VMP update equations. Everything else — the inference engine, Variational EM loop, planner, VAE integration, and the examples — was assembled by Claude from a description of the generative model and the factor graph structure.
>
> **Note on tests.** The inference diagnostic test suite has been removed from this release as it relies on internal tooling that cannot be open-sourced yet.

## Why "Poor Man's Active Inference"?

This is a very Bayesian version of model-predictive control with unknown transition *and* observation functions. It shares the Active Inference spirit — perception and action as inference on a generative model — but cuts some corners:

- **Variational EM instead of full Bayesian treatment.** The M-step updates VAE parameters via gradient descent rather than maintaining a posterior over neural network weights.
- **VAE pre-training.** The observation model gets a warm start before dynamics learning kicks in.
- **No epistemic priors.** We don't use expected information gain or curiosity-driven exploration. The planner minimises divergence from a goal, not uncertainty.
- **Linear dynamics in latent space.** The transition model is $x_{t+1} = Ax_t + Bu_t + \text{noise}$, which is expressive enough when the VAE learns a good representation, but it's not a universal dynamics model.

What it *does* get right: the entire pipeline — system identification, state estimation, and planning — runs on one unified factor graph via message passing. No separate modules stitched together.

## Generative model

![Factor Graph](docs/model.png)

The model is a linear dynamical system in a VAE's latent space:

**Priors**

$$\mathbf{a} \sim \mathcal{N}(\boldsymbol{\mu}_a,\, \Sigma_a), \quad W \sim \text{Wishart}(\nu_0,\, I), \quad \mathbf{b} \sim \mathcal{N}(\mathbf{0},\, \Sigma_b)$$

where $\mathbf{a} = \text{vec}(A)$ is the vectorised transition matrix and $\mathbf{b} = \text{vec}(B)$ is the vectorised control matrix.

**Latent dynamics**

$$x_1 \sim \mathcal{N}(\mathbf{0},\, I)$$

$$x_t \mid x_{t-1}, u_{t-1} \sim \mathcal{N}\!\left(A\, x_{t-1} + B\, u_{t-1},\; W^{-1}\right) \quad t = 2, \dots, T$$

**Observations**

$$y_t \mid x_t \sim p_\theta(y_t \mid x_t) \quad \text{(VAE decoder)}$$

The VAE encoder $q_\phi(x_t \mid y_t)$ provides Gaussian messages from images into the latent chain. The decoder $p_\theta(y_t \mid x_t)$ reconstructs images from latent states.

## Inference via message passing

All inference operates on the Forney-style factor graph above:

- **Forward-backward messages** on the latent chain (derived from structured mean-field VMP, not pure BP — the messages account for uncertainty in the transition parameters).
- **VMP updates** for the shared parameters $\mathbf{a}$, $W$, $\mathbf{b}$ by accumulating messages from all transition factors.
- **Variational EM** jointly refines the VAE and dynamics: the E-step runs VMP+forward-backward to get posterior marginals $q(x_t)$, the M-step updates VAE parameters via gradient descent — framed as a gradient-based message at the observation nodes ([Şenöz et al.](https://doi.org/10.3390/e23070807)).

## Planning as inference

Once the model is learned, planning becomes inference on the same factor graph:

1. Fix the learned dynamics ($A$, $B$, $W$).
2. Observe start and goal images at $t=0$ and $t=T$.
3. Treat actions $u_t$ as latent variables with priors.
4. Run VMP + forward-backward to infer the action sequence that connects start to goal.

A receding-horizon loop re-observes the actual state after executing a few steps, replans, and repeats — giving closed-loop control from pixels.

![Pendulum swing-up](docs/pendulum_result.png)

## Known limitations

- **VAE pre-training is not message passing.** The observation model is initialised with standard gradient descent. The Variational EM refines it further, but the initial representation is deep learning, not Bayesian inference.
- **Latent space topology.** The VAE doesn't guarantee that the latent space is topologically aligned with the physical state space. Linear interpolation in latent space may not correspond to smooth physical trajectories, which limits the planner's reliability for some goal angles.
- **No velocity from a single image.** A single 28×28 frame doesn't encode angular velocity. The planner can swing the pendulum to a target but cannot stabilise it there — it arrives with non-zero velocity and overshoots.
- **Linear dynamics extrapolation.** The dynamics model is learned from training trajectories with moderate torques. Planning to goals far outside the training distribution requires the linear model to extrapolate, which may fail.
- **Sequential VMP across trajectories.** In the multi-trajectory E-step, each trajectory's VMP accumulation starts from the prior rather than from a global accumulation. This is an approximation that converges across outer EM iterations but may underfit with few iterations.
- **Planning convergence.** The VMP action updates have a contraction factor close to 1 when $B^\top W B \gg \Lambda_{\text{prior}}$, causing slow convergence. Short planning horizons (T=8) with frequent replanning work better than long horizons.
- **Dynamics stability.** The learned transition matrix $A$ can have eigenvalues close to or exceeding 1 in magnitude depending on the VAE initialisation, leading to unstable autonomous dynamics. Training variance means results differ across runs.

## Interactive Notebook

The full story — learning to see, learning physics, planning a swing-up — as an interactive [marimo](https://marimo.io) notebook with animated visualizations:

```bash
uv run marimo run notebook.py
```

Sliders let you scrub through VAE training, EM convergence, and planning execution step by step. Auto-playing GIFs show each phase in motion. A dropdown lets you switch between pre-computed targets (θ=0.5, θ=1.0, θ=π).

## Examples

![Digit rotation](docs/digits_result.png)

| Example | What it does |
|---------|-------------|
| `rotating_digits.py` | Learn rotation dynamics of MNIST digits, predict future frames |
| `controlled_digits.py` | System identification with control inputs, predict under different actions |
| `end_to_end_digits.py` | Variational EM: jointly refine VAE + dynamics |
| `pendulum.py` | Full pipeline: system identification + closed-loop planning on a simulated pendulum |

Run any example with:

```bash
uv run python examples/pendulum.py
```

## Installation

Requires Python 3.10+.

```bash
uv pip install -e .
```

Optional: `pip install matplotlib marimo` for visualisation and the interactive notebook.

## Structure

```
jopa/
  distributions.py     # Gaussian, Wishart in natural parameter form
  message_passing.py   # Forward-backward messages, VMP accumulation
  inference.py         # infer() — learn dynamics, plan() — infer actions
  em.py                # variational_em() — joint VAE + dynamics learning
  nodes/
    transition.py      # Continuous transition node (VMP message rules)
    observation.py     # VAE observation node
  nn/
    vae.py             # Convolutional VAE (Flax)
  data.py              # MNIST loading and rotation utilities
  envs/                # Simulation environments (pendulum)
```

## References

- de Vries, B. "Active Inference for Physical AI Agents — An Engineering Perspective", arXiv:2603.20927, 2026.
- Şenöz, I. et al. "Variational Message Passing and Local Constraint Manipulation in Factor Graphs", Entropy, 2021.

## License

GPL-3.0 — free to use, derivatives must also be open source.
