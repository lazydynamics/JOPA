# 🍑 JOPA

**Joint Observation–Planning Architecture** — one factor graph that learns latent
dynamics, infers latent state, predicts the future, and plans actions. Every
*inference* task is message passing on that graph; the pixel encoder is the one
component trained by gradients.

[![CI](https://github.com/lazydynamics/JOPA/actions/workflows/ci.yml/badge.svg)](https://github.com/lazydynamics/JOPA/actions/workflows/ci.yml)

<p align="center">
  <img src="assets/hero.gif" width="92%" alt="The JOPA factor graph with forward, backward, observation and parameter messages; below, one sweep smoothing and then predicting, and the closed-loop cost of a sensor that overstates its own precision" />
</p>

<p align="center"><em>
  Everything above is computed by the library — conjugate system identification,
  forward/backward message passing, and exact action inference on the same graph.
</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.20927">Active Inference (de Vries, 2026)</a> &nbsp;·&nbsp;
  <a href="https://doi.org/10.3390/e23070807">VMP in Factor Graphs (Şenöz et al., 2021)</a> &nbsp;·&nbsp;
  <a href="https://rxinfer.com">RxInfer</a> &nbsp;·&nbsp;
  <a href="https://lazydynamics.com">Lazy Dynamics</a>
</p>

---

## Design and background

Why message passing, the generative model, the relation to JEPA, and the credit
for the transition-node rules live in the [design notes](docs/index.md).

## The agent loop

```python
from jopa import Agent, Block, JointModel, LearnedLinear

model = JointModel([
    Block("z", LearnedLinear(dim=6, du=2, offset=True), observe=encoder),
])
model.learn(replay)                           # conjugate system identification

agent = Agent(model, horizon=6, forget=0.5)   # filter -> plan -> act -> learn
agent.goal(goal_image)                        # goals are observations too
while True:
    u = agent.step(camera())                  # one message-passing cycle
    act(u)
```

`encoder` is anything mapping an observation to a Gaussian message, so the same
loop drives your environment: wrap a feature extractor in `Frozen`, or use
`VAEObservation` / `PoseMotionObservation` for learned pixel sensors.

## How it fits together

Every example follows the same spine — an `Observation` turns data into a
Gaussian message, a `Block` pairs it with a transition, and `JointModel`
answers one of the four questions. What differs is how the sensor is obtained
and which question gets asked.

**Offline.** Collect observations and actions. Train a sensor by gradient
descent — the only gradients in the system. Encode the data once. Then fit
`q(A,B,W)` by message passing: each encoded observation becomes a message,
forward–backward smooths the chain, and the accumulated messages update the
parameter posteriors. Nothing here is a point estimate.

**Online.** Every step is inference, at three levels of the same graph:
the **state** is filtered (transition message predicts, observation message
updates), the **parameters** may be refit from nearby past experience — still
conjugate VMP, not gradients — and the **actions** are the posterior from one
exact sweep. Apply the first action, discard the rest, repeat.

| stage | digits | pendulum | reacher |
|---|---|---|---|
| sensor objective | `train_vae` (pixel ELBO) | `train_vae` | `train_pose_motion_vae` (action-conditioned latent rollout) |
| sensor after fitting | frozen (`Frozen`) — or refined jointly by variational EM in `digits_end_to_end` | refined in the M-step | frozen |
| dynamics `q(A,B,W)` | `model.learn` | `model.learn` | `model.learn`, then refit locally each step |
| question asked | `smooth` (+ prediction) | `plan`, `Agent` | `Agent` (closed loop) |
| goal | — | a target image | a target image, with decoder-Jacobian precision |

Local refitting, decoder-Jacobian goal geometry and the distance-conditioned
action prior are used only by the Reacher; they exist because a single global
linear model cannot represent an articulated arm across its whole workspace.

## MuJoCo Reacher from pixels

A two-link arm reaching a target it has only ever seen. The agent receives 64×64
grayscale frames and its own torques. No joint angles, no velocities, no reward,
no simulator state. The goal is an image of the arm at the target.

<p align="center">
  <img src="assets/reacher_pixels.gif" width="46%" alt="The arm closing on a goal position, with the fingertip error per step" />
</p>

```bash
python -m examples.reacher.run train    --data outputs/gpu_run   # sensor + q(A,B,W)
python -m examples.reacher.run validate --data outputs/gpu_run   # pixel-only diagnostics
MUJOCO_GL=egl python -m examples.reacher.run evaluate \
    --pose-seed 515151 --poses 20                                # closed loop
```

40 poses, two seeds disjoint from every training and tuning decision, 240
closed-loop steps each, sensor and dynamics frozen:

| | result |
|---|---|
| poses reaching within 1 cm | 40 / 40 |
| poses within 3 cm over the last 20 steps (mean) | 30 / 40 |
| poses whose worst error in the last 20 steps is under 3 cm | 24 / 40 |
| poses under 3 cm for 95% of the last 60 steps | 14 / 40 |
| control step (local refit + exact planning) | 66 ms |

<p align="center">
  <img src="assets/goal_convergence.png" width="88%" alt="Distance to goal against closed-loop step for five poses, and the agent's belief and action posterior spread over the same episode" />
</p>

One control step, in full: the frame is filtered, six future torques are
inferred in a single exact sweep with their posterior spread, the first is
applied, and the rest are discarded and re-inferred next step.

<p align="center">
  <img src="assets/planning.gif" width="88%" alt="The arm, the distance to goal, and the six-step action posterior recomputed every step" />
</p>

Every pose reaches the goal; the terminal error is not uniformly settled — some
poses oscillate around the goal within a few centimetres rather than parking.
`evaluate` writes per-pose traces and both the mean and maximum terminal
criteria.

The sensor (`jopa/nn/pose_motion.py`) is trained on the agent's own replay. The
deployed dynamics are the conjugate posterior, refit each step from the nearest
encoded transitions. Every torque is the mean of the exact action posterior.

Sensor training runs on GPU and is compute-bound. The closed loop is reported on
CPU for two reasons: at a 6-D state the per-step work is 6×6 linear algebra, so
it is dispatch-bound rather than compute-bound (measured GPU utilisation during
stepping: 2–3%), and GPU runs are not bit-reproducible — XLA autotuning makes
otherwise identical episodes diverge, while CPU episodes are identical across
processes. The 66 ms is therefore the reproducible figure, not a ceiling.

## Building blocks

| | |
|---|---|
| `Gaussian`, `Wishart` | Natural-parameter distributions — every message lives here |
| `Block(name, transition, observe)` | One latent-state slice |
| `JointModel.{learn, smooth, filter, plan}` | The four queries as methods; `plan()` returns the exact Gaussian action posterior in one sweep |
| `Agent(model, horizon, forget, ...)` | The closed loop, with surprise- and excitation-gated relearning |
| `LearnedLinear` | `x' ~ N(A·x + B·u [+ c], W⁻¹)` with conjugate `q(A,B,W)`; `offset=True` learns a constant drift |
| `LearnedLinear.attach_replay` / `.localize` | Refit the posterior from the nearest encoded transitions — local dynamics without leaving conjugacy |
| `LearnedLinear.remember(forget)` | Posterior becomes the next prior — continual learning with exponential forgetting |
| `KnownPhysics` | Re-linearized gray-box dynamics |
| `Frozen(encode, decode)` | Any fixed encoder, plus an optional renderer |
| `VAEObservation` | Frozen probabilistic VAE sensor — heteroscedastic messages from pixels |
| `PoseMotionObservation` | Structured `pose[4] + motion[2]` pixel sensor with static image goals and decoder-Jacobian goal geometry |
| `LinearCoupling` | Cross-block Gaussian factor — multimodal fusion |

Library layout: `distributions.py` (messages) · `observations.py` (likelihoods) ·
`transitions.py` (dynamics) · `graph.py` (`Block`, `JointModel`, coupling) ·
`message_passing.py` (α/β sweeps, exact planning) · `agent.py` (the loop) ·
`nn/` (learned sensors).

## A minimal example

System identification then planning on a controlled 2-D system, where the latent
is seen only through `encode`:

```python
import numpy as np
from jopa import Block, Gaussian, JointModel, LearnedLinear

def encode(x):
    lam = 1e4 * np.eye(2)
    return Gaussian(eta=lam @ x, lam=lam)

model = JointModel([
    Block("z", LearnedLinear(dim=2, du=1, n_iterations=40), observe=encode),
])
model.learn(trajectories)                                # [{"z": [...], "control": [...]}, ...]
actions = model.plan({"z": [start, None, None, goal]})   # exact posterior, one sweep
```

## Examples

| script | what it shows |
|---|---|
| [`examples/reacher/`](examples/reacher) | pixel-and-action-only MuJoCo Reacher: `run.py` (CLI), `runtime.py` (the closed loop), `spec.py` (frame and latent geometry), `artifacts.py`, `train.py`, `diagnose.py`, `evaluate.py`, `metrics.py`, `collect.py` |
| [`digits_rotating.py`](examples/digits_rotating.py) | latent linear dynamics behind a frozen VAE; the smoothing sweep extrapolating |
| [`digits_controlled.py`](examples/digits_controlled.py) | add controls, learn `B`, predict under action regimes |
| [`digits_end_to_end.py`](examples/digits_end_to_end.py) | variational EM — dynamics and VAE weights learned together |
| [`pendulum.py`](examples/pendulum.py) | image-goal control on a classic plant |
| [`figures/hero.py`](figures/hero.py) | regenerates `assets/hero.gif` |
| [`figures/planning.py`](figures/planning.py) | regenerates `assets/planning.gif` |
| [`figures/convergence.py`](figures/convergence.py) | regenerates `assets/goal_convergence.png` |

Every image on this page is produced by a command in this repo, except
`assets/model.png`, which is a hand-drawn diagram. `assets/reacher_pixels.gif`
comes out of `evaluate` itself — running it without `--no-video` writes one clip
per pose beside the traces.

## Checkpoint validation

Before planning with a learned checkpoint, gate it on held-out observations:

```bash
uv run jopa-validate \
  --vae checkpoints/vae_d4.npz \
  --sequence data/heldout_sequence.npy \
  --latent-dim 4 \
  --dynamics checkpoints/dynamics.pkl \
  --max-reconstruction-mse 0.02 \
  --min-latent-linearity-r2 0.8 \
  --max-one-step-latent-mse 0.05
```

The report covers reconstruction MSE, latent linearity R², and one-step latent
prediction MSE; the threshold flags make the command exit non-zero, so loops can
gate on the same signal.

## Install and run

```bash
git clone https://github.com/lazydynamics/JOPA.git && cd JOPA
uv sync                            # library + digit/pendulum examples
uv sync --extra reacher            # adds MuJoCo for the Reacher example
uv run pytest tests -q             # run per file: JAX compilation caches add up
```

Everything except sensor training runs on CPU in minutes. **A GPU is needed for
exactly one thing: training a pixel sensor** (`train` below, ~2 h on one A100).
Inference — filtering, planning, closed-loop control — is 6×6 linear algebra
per step and is dispatch-bound, so CPU is the faster and the reproducible
choice.

| Command | Needs | Time |
|---|---|---|
| `python -m examples.digits_rotating` | CPU | ~15 s warm, ~4 min cold (trains a small VAE) |
| `python -m examples.digits_controlled` | CPU | ~1 min |
| `python -m examples.digits_end_to_end` | CPU | ~5 min (variational EM) |
| `python -m examples.pendulum` | CPU | ~30 min first run, ~10 s after (EM result is cached) |
| `python figures/hero.py --out assets/hero.gif` | CPU | ~2 min |
| `MUJOCO_GL=egl python figures/planning.py` | CPU + MuJoCo | ~3 min |
| `python -m examples.reacher.run train --data outputs/gpu_run` | **GPU** | ~2 h |
| `python -m examples.reacher.run validate --data outputs/gpu_run` | CPU | ~10 min |
| `MUJOCO_GL=egl python -m examples.reacher.run evaluate` | CPU + MuJoCo | ~5 min |

The Reacher `train` step needs pixel/action replay in `--data`; `evaluate`
reads the frozen sensor and dynamics from `--outdir` (default
`outputs/reacher_pixels`) and writes traces, per-pose metrics and video there.
`MUJOCO_GL=egl` is required for headless rendering.

## How this was built

Most of this repository was written by Claude Code under human direction. Read
the code before you rely on it, treat the examples as demonstrations rather than
reference implementations, and expect bugs. Known limitations are filed as issues
rather than left implicit, including the ones that bound the results above.

The inference core is the exception. `jopa/nodes/transition.py` implements the
`ContinuousTransition` node, and its structured variational rules were derived
and ported by hand from our earlier [RxInfer](https://rxinfer.com) contribution,
with unit tests in `tests/test_transition_node.py`. That file is the part to
trust. One known gap in it is tracked separately: `ct_forward` predicts with the
posterior mean of `A` rather than its distribution, so the online filter is
certainty-equivalent in the parameters while learning and smoothing are not.

Contributions written with an LLM are welcome, and worth saying so in the pull
request. Keep the diff small enough for a human to review and make sure the
tests pass.

## License

GPL-3.0.
