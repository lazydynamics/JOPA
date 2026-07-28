---
title: JOPA — design notes
---

# JOPA design notes

One factor graph that learns how a system moves, infers where it is, predicts
what comes next, and decides what to do. The same message passing serves all
four; only the set of variables you admit you don't know changes.

This page is the design background. For install, examples and results see the
[README](https://github.com/lazydynamics/JOPA#readme).


## Why message passing

A learned controller is usually four systems: an encoder, a dynamics model, a
state estimator, and a policy — each with its own objective, each discarding
what the next one needs. JOPA is one probabilistic model and four questions
asked of it:

| Question | Inferred | Given |
|---|---|---|
| **System identification** | dynamics `A, B, W` | observations, controls |
| **Variational EM** | dynamics *and* encoder weights, alternating | data |
| **Filtering · smoothing · prediction** | latent state `x_t` | model, observations |
| **Planning** | action sequence `u_t` | model, start + goal |

Variational EM is available when the observation is `learnable` — the E-step
smooths, the M-step takes gradient steps on the encoder against those smoothed
targets (`digits_end_to_end.py`). The Reacher pipeline does **not** use it: the
sensor is trained first on replay, then frozen, and only `q(A,B,W)` keeps
learning. Both routes are supported; they are not the same experiment.

Only which variables are latent changes. Nothing is retrained between
questions, and three properties follow from that.

**Uncertainty is carried end to end.** The sensor emits a Gaussian whose
per-dimension precision it sets itself. The dynamics are a posterior `q(A,B,W)` —
Gaussian over `A` and `B`, Wishart over the process precision — not point
estimates. The planner reads both. Measured on the pixel Reacher: a sensor that overstates
its precision by 12× spends 1.9× the actuation and 2.0× the control chatter for
the same final accuracy, because it acts on its own noise.

**Planning is exact inference, not optimization.** Group each time slice as
`s_t = [x_t; u_t]` and the joint is a Gaussian chain, so the α/β recursions that
smooth the past are the optimal controller (LQG duality). One sweep, no
iteration, no mean-field cut between states and actions, no policy network.
Goals arrive as observations.

**Learning is continual by construction.** Conjugacy makes today's posterior
tomorrow's prior. `remember(forget=0.5)` tempers evidence in natural parameters,
so the model can track a drifting world instead of averaging over its history.
The deployed dynamics are 75 parameters, refit from a few hundred nearby
transitions in milliseconds.

## Relation to JEPA — and what changes for control

The pixel sensor is a VAE, and reconstruction is its largest single loss term —
pixels are what anchor the pose coordinates to something real. What it borrows
from JEPA is the *additional* objective: an action-conditioned latent rollout
that encodes a frame window, rolls the latent forward under the actions actually
taken, and penalises the error in the latent rather than in pixel space. That is
the term which makes the latent linearly predictable, which is what the
conjugate dynamics downstream require. That term carries the same collapse risk
joint-embedding methods face — a latent can satisfy a prediction objective by
becoming constant — so it uses the VICReg remedy: a per-dimension variance anchor
plus an off-diagonal covariance penalty.

The architectures differ in what they are for. I-JEPA and V-JEPA are
self-supervised representation objectives: no goals, no actions, no control.
Goal-directed planning belongs to the wider world-model programme those methods
feed into, not to the JEPA objective itself. JOPA is goal-conditioned by
construction — the goal is an observation, and planning is inference over the
same graph.

Three further differences follow from having to close a loop rather than feed a
downstream head:

- **The embedding is a distribution, not a point.** The encoder emits
  `N(μ(y), Σ(y))` with heteroscedastic precision, calibrated against held-out
  innovations. Fusing vision with dynamics requires knowing how much to trust
  the encoder on this frame; a point embedding cannot express that.
- **The predictor becomes a posterior.** In I-JEPA the predictor serves the
  pretext task and the encoder is the product; V-JEPA 2's action-conditioned
  variant does keep a predictor for planning, but as a learned network. Here the
  predictor used during sensor training is thrown away and replaced by the
  conjugate `q(A,B,W)` — a distribution over dynamics that plans, filters and
  keeps learning online.
- **The latent is structured.** For an articulated arm the state is
  `[pose(4), motion(2)]`: pose from a single frame, motion from differences of
  consecutive pose encodings, and the decoder sees pose only, so velocity cannot
  enter the position channels. A static goal image is `[pose(image), 0, 0]`.

Practical consequences: there is no reward function to design, the deployed
model is small enough to refit inside the control loop, and the controller
reports calibrated uncertainty about both the state and its own dynamics.

## The model

<p align="center">
  <img src="https://raw.githubusercontent.com/lazydynamics/JOPA/main/assets/model.png" width="70%" alt="The JOPA generative model: linear-Gaussian latent transitions with a learned observation likelihood" />
</p>

A `JointModel` is a factor graph; each `Block` contributes a latent slice with a
**transition** factor and an **observation** factor. Transition parameters are
shared across all time steps:

$$
\begin{aligned}
x_t \mid x_{t-1}, u_{t-1} \;&\sim\; \mathcal{N}\!\big(A\,x_{t-1} + B\,u_{t-1},\ W^{-1}\big) && \text{(transition)} \\
y_t \mid x_t \;&\sim\; p_\theta(y_t \mid x_t) && \text{(likelihood)} \\
\mathbf{a} = \mathrm{vec}(A),\ \ \mathbf{b} = \mathrm{vec}(B) \;&\sim\; \mathcal{N}(\cdot), \quad W \;\sim\; \mathcal{W}(\cdot) && \text{(shared priors)}
\end{aligned}
$$

State and parameter inference is variational message passing under the
structured posterior $q(x)\,q(\mathbf{a})\,q(\mathbf{b})\,q(W)$: mean-field
*between* the four groups, but $q(x)$ keeps the full chain structure, so the
latent trajectory is never factorized over time. Encoder weights $\phi$ are not
in this factorization — they are fitted by gradient descent, and the resulting
$\mathcal{N}\big(\mu_\phi(y_t), \Sigma_\phi(y_t)\big)$ enters as an amortized
stand-in for $p_\theta(y_t \mid x_t)$. Planning is the exception to the
mean-field split: states and actions are solved jointly and exactly.

When the future carries no observation, the forward–backward sweep that smooths
the past predicts it — inference and prediction are one operation on different
parts of the chain.

### Where the transition rules come from

Most of the work happens in one node: `ContinuousTransition` in
[`jopa/nodes/transition.py`](https://github.com/lazydynamics/JOPA/blob/main/jopa/nodes/transition.py), the structured VMP rules
for $y \sim \mathcal{N}(Ax + Bu, W^{-1})$ when $x$ and $y$ are *both* uncertain.
That caveat is the entire difficulty. Cut the posterior as $q(y)\,q(x)$ and the
transition matrix you learn is attenuated toward zero — errors-in-variables — so
the rules have to carry $\mathbb{E}[A M A^\top]$ and the parameter-uncertainty
contractions you see in the code. Keeping $q(y,x)$ joint is what makes it
possible to learn dynamics from latents that were themselves inferred.

Everything else here is that node asked a different question. Filtering,
smoothing and prediction are its forward and backward messages; learning
$q(A,B,W)$ is its parameter messages; action inference is either a message from
the same node (`ct_message_u`) or a Gaussian chain assembled from the same
cached posterior expectations.

The rules themselves are not new. They were derived and integrated into
[RxInfer](https://rxinfer.com) some time ago; the write-up remains unpublished.

## Design notes

- **Bayesian inference is the only verb.** Learning, state inference, prediction
  and planning are each `q(·)` over a different subset of the same graph — no
  reward shaping, policy networks, or replay-buffer retraining.
- **Linear-Gaussian latent dynamics**, either global or refit locally around the
  current belief. The encoder is the one gradient-trained component.
- **Planning is exact inference.** The joint over states and actions is Gaussian,
  so the action posterior is one forward–backward sweep in information form. The
  iterative VMP planner remains available as `plan(method="vmp")`.
- **Composability.** Adding a modality is appending a `Block`; information flows
  across slices through `LinearCoupling`. `JointModel` knows only blocks and
  messages — not images, proprioception, or actions.
- **Continual learning is conjugacy.** Forgetting and parameter drift are
  *priors* — exponential forgetting, random-walk diffusion on volatile
  parameters — not optimizer tricks.

## References

- de Vries, B. *Active Inference for Physical AI Agents — An Engineering Perspective*, arXiv:2603.20927, 2026.
- Şenöz, I. et al. *Variational Message Passing and Local Constraint Manipulation in Factor Graphs*, Entropy 23(7), 2021.
- Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023.
- Bardes, A., Ponce, J., LeCun, Y. *VICReg: Variance-Invariance-Covariance Regularization*, ICLR 2022.
- Watter, M. et al. *Embed to Control*, NeurIPS 2015; Zhang, M. et al. *SOLAR*, ICML 2019.


## Credit

The structured variational rules for learning a transition function when both
endpoints are uncertain were derived and integrated into
[RxInfer](https://rxinfer.com) some time before this repository existed; JOPA's
`ContinuousTransition` node implements the same updates. The write-up remains
unpublished.
