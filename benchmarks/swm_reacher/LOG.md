# Result log

Every evaluation of the protocol, in the order it was run. Amendments 1–6
used scripts and an earlier inference engine that were removed when the
library was rebuilt on `jopa.fg` (amendment 7); they are in the git history
at commit `db95185`.

# Gate outcome — 2026-11-01

Protocol: [`PROTOCOL.md`](PROTOCOL.md) at da3e7d6. Both gate conditions were run
before any closed-loop JOPA evaluation.

## 1. Reproduce LeWM — pass

`eval_wm.py --config-name=reacher policy=quentinll/lewm-reacher seed=42`,
upstream config, RTX 5090: **82 %** success (50 episodes). Published: 86 %
(LeWM paper, read from figure), 83.4 ± 5.7 % (7-seed re-run in later work).
Within the ±10-point tolerance. CEM solve 9–17 s per replan over the
50-episode batch; whole evaluation 42 s.

## 2. JOPA open-loop prediction R² ≥ 0.8 at 5 blocks — fail

`gate_predict.py`: LeWM's frozen encoder, one frame per 5-step block,
1000 training / 200 held-out episodes (episode split, seed 3072), 500
held-out starts, R² in the PCA-projected space against the training mean.

| R² at k blocks (d = 8, 15 % of variance) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| JOPA, global q(A, B, W) | 0.67 | 0.45 | 0.33 | 0.24 | **0.19** |
| LeWM's own predictor (yardstick) | 0.99 | 0.99 | 0.99 | 0.98 | 0.98 |

The failure is not an artefact of the projection or the state. Least-squares
linear dynamics on cached embeddings (300 held-out starts):

| d | variance kept | history | R² at k = 1 … 5 |
|---|---|---|---|
| 8 | 0.15 | 2 frames | 0.71 0.51 0.39 0.27 0.20 |
| 8 | 0.15 | 3 frames | 0.71 0.52 0.39 0.27 0.20 |
| 32 | 0.52 | 2 frames | 0.68 0.47 0.35 0.24 0.18 |
| 192 | 1.00 | 3 frames | 0.60 0.37 0.25 0.18 0.13 |

Local least squares on the 500 nearest training transitions reaches
0.85 / 0.71 / 0.55 / 0.38 / 0.10 — better at short range, no better at the
goal offset.

**Reading.** LeWM's latent dynamics are strongly nonlinear: a linear model
in any projection of its 192-d embedding explains about a fifth of the
5-block variance its own predictor explains. LeWM shapes the embedding to be
isotropic Gaussian, not linear in its dynamics; nothing in its objective
asks for the latter.

**Caveat on the yardstick.** LeWM was trained on a random 90 % of *samples*;
our held-out *episodes* were very likely in its training data, so 0.98 is an
in-sample figure. The linear rows are out-of-sample for the linear fit.

## Library defects found by the gate

- `LearnedDelayLinear` local refits under-converge: with the mean-field
  q(A) q(B) factorisation, coordinate ascent crawls along the A-vs-offset
  ridge that a small neighbourhood creates. Against least squares on the same
  500 neighbours (R² 0.85 at k = 1) it reaches 0.31 / 0.44 / 0.61 after 6 / 30
  / 100 iterations.
- `attach_replay`'s default `obs_prec=1e2` makes local refits diverge
  (R² −1.3 at k = 1, −18 at d = 16) through the errors-in-variables
  correction on neighbourhoods narrower than the assumed noise.

Per the protocol, no closed-loop JOPA numbers were produced.

# Amendment 1: linearizing head — fail

`gate_head.py` (PROTOCOL amendment 1): MLP head on `[e_{t-1}, e_t]`, trained
10 000 steps on the 1000 training episodes for 5-block linear rollout, read
out linearly to PCA-32 of LeWM's embedding. JOPA's VMP fit on the head's
latent; same 500 held-out starts.

| latent | read-out R² | R² at k = 1 … 5 (PCA-32) | PCA-8 |
|---|---|---|---|
| 16 (3 000 steps) | 0.47 | 0.38 0.30 0.23 0.19 0.16 | 0.23 at k=5 |
| 32 | 0.92 | 0.63 0.43 0.31 0.23 **0.19** | 0.23 at k=5 |
| 64 | 0.98 | 0.64 0.43 0.31 0.24 **0.19** | 0.23 at k=5 |

The head buys nothing over the PCA state (0.18–0.20). JOPA's VMP fit equals
the head's own gradient-trained linear map to two decimals, so the gap is in
the model class, not in the inference.

**The latent is predictable; it is not linearly predictable.** A residual
MLP `h_{t+1} = h_t + f(h_{t-1}, h_t, u_t)` trained on the same training
episodes (`gate_mlp_ceiling.py`) reaches on the *held-out* starts:

| PCA-32, R² at k = 1 … 5 | |
|---|---|
| residual MLP, held-out | 0.96 0.95 0.95 0.94 **0.94** |
| residual MLP, training episodes | 0.97 0.97 0.97 0.97 0.97 |
| least-squares linear | 0.64 0.41 0.29 0.22 0.17 |
| persistence (`ĥ = h_t`) | 0.58 0.23 −0.02 −0.18 −0.31 |

This removes the in-sample caveat on LeWM's 0.98: a nonlinear predictor
generalises to held-out episodes at 0.94. A global linear model — in PCA
coordinates or in a learned 64-d lift with additive control — barely beats
persistence. The 2-link arm's state enters the image through angles, and the
torque's effect depends on the configuration; a lift with `z' = Kz + Bu`
cannot express that.

**Verdict.** On LeWM's Reacher, a global linear-Gaussian world model is the
wrong model class by a wide margin (0.19 vs 0.94 at the goal offset). Per the
protocol, the comparison stops here; no closed-loop JOPA numbers.

# Amendment 2: plan on LeWM's predictor — no win, stopped

Drop-in solvers through the unmodified `eval_wm.py` (`run_eval.sh`,
`config/`), 50 episodes each.

| planner on LeWM's model | seed 42 | dev seed 100 |
|---|---|---|
| CEM (upstream) | **82 %** | **86 %** |
| Adam (harness `solver=adam`) | 68 % | |
| Gauss–Newton, 1 start, λ = 1 | 66 % | |
| Gauss–Newton, 8 starts, λ = 0.1 | | 76 % |
| Gauss–Newton, 8 starts, λ = 1 | | 80 % |

`diag_solvers.py` (dev starts, not eval): 8-start Gauss–Newton reaches a
lower GoalMSE than CEM on 94 % of starts, yet succeeds less often. Better
optimisation of LeWM's latent goal cost does not buy success; CEM's sampling
is not the bottleneck it looks like.

Dynamics shift, CEM, dev seed 100 (`config/reacher_shift.yaml`, the
harness's own variation space; masses verified to scale):

| arm and finger density | 500 | 1000 | 1500 |
|---|---|---|---|
| LeWM + CEM | 92 % | 86 % | 80 % |

The variation space spans ±50 % density, and frozen LeWM moves by about
±6 points across it — within the binomial noise of 50 episodes (±5). There is
no shift in this harness that degrades LeWM enough for online adaptation to
win.

**Verdict.** On stable-worldmodel Reacher, neither JOPA's own dynamics
(gate, amendment 1) nor exact planning on LeWM's dynamics (amendment 2)
beats LeWM + CEM, and the harness's dynamics shift does not hurt LeWM. The
comparison stops here; no eval-seed numbers beyond seed 42 were produced.

# Where LeWM's misses come from (dev seed 100, CEM, LeWM's model unchanged)

Success is "within 0.05 rad of the goal `qpos` at any step within the budget".

| configuration | success |
|---|---|
| upstream (GoalMSE on the last predicted step, replan every 5 blocks, 50 steps) | 86 % |
| upstream, 100-step budget | 100 % |
| upstream, replan every block | 50 % |
| GoalMSE on every predicted step (`objectives.py`), replan every 5 blocks | 98 % |
| GoalMSE on every predicted step, replan every block | 100 % |

LeWM's misses are a planning-objective artefact: a last-step-only cost lets
the plan arrive at the end of the horizon, so it runs out of budget (and,
when replanning every block, never commits). Scoring every step fixes it for
LeWM's own model and planner. On this benchmark the model is not the limit,
and LeWM with a one-line objective change saturates it — no planner or
dynamics model can beat it on nominal success.

# Amendment 3: low data (100 episodes) — LeWM wins

Dev seed 100 selection (pre-registered). JOPA: 15 configurations tried; the
chosen one (d = 128, 100 neighbours, horizon 3, replan every block) gave 90 %
on dev seed 100 and 80 % on dev seed 101. LeWM retrained on the same 100
episodes, checkpoint at epoch 100 (≈ 12 700 steps):

| LeWM-100, epoch 100, dev seed 100 | replan every 5 blocks | every block |
|---|---|---|
| last-step GoalMSE (upstream) | 48 % | 36 % |
| every-step GoalMSE | 72 % | **98 %** |

Eval seeds, 50 episodes each:

| 100 training episodes | 42 | 1 | 2 | 3 | 4 | mean ± std | median solve, 50 envs |
|---|---|---|---|---|---|---|---|
| LeWM (retrained) + CEM, every-step, every block | 96 | 100 | 98 | 98 | 90 | **96.4 ± 3.8 %** | 3.95 s |
| JOPA, frozen DINOv2 + local conjugate dynamics | 68 | 76 | 78 | 86 | 66 | 74.8 ± 8.1 % | **0.034 s** |

LeWM trained from scratch on 100 episodes, given the every-step objective,
still nearly saturates the task. JOPA plans ~115× faster but succeeds ~22
points less. The 300-episode, DINO-WM and longer-training rows were not run:
the 100-episode result already decides the comparison, and LeWM's later
checkpoints could only raise its number.

**Overall verdict on stable-worldmodel Reacher.** LeWM wins on success in
every regime tested: full data, density shift, and 100 episodes. JOPA's one
measured advantage is planning cost.

# Amendment 4: neural-linear JOPA on LeWM's encoder — near parity, 36× cheaper planning

Held-out 5-block prediction in LeWM's 192-d latent (`train_neural_linear.py`):

| model | R² at k = 1 … 5 |
|---|---|
| persistence | 0.48 0.07 −0.20 −0.36 −0.47 |
| JOPA linear (gate) | ≈ 0.17 at k = 5 |
| neural-linear, gradient-trained head | 0.94 0.91 0.89 0.87 0.86 |
| neural-linear, exact Bayesian last layer | 0.94 0.90 0.85 0.81 0.78 |

Closed loop, full data, both with the every-step cost and replanning every
block. The JOPA configuration was fixed before its first run and not tuned.

| | dev 100 | dev 101 | 42 | 1 | 2 | 3 | 4 | eval mean ± std | median solve |
|---|---|---|---|---|---|---|---|---|---|
| LeWM + CEM | 100 | 100 | 98 | 100 | 100 | 100 | 100 | **99.6 ± 0.9 %** | 2.0 s |
| JOPA neural-linear (Bayesian W) | 94 | 92 | 92 | 90 | 96 | 96 | 92 | 93.2 ± 2.7 % | **0.055 s** |
| ablation: gradient-trained head | 96 | | | | | | | | |

Solve times are per replan for the batch of still-running episodes (CEM's
batch shrinks as episodes succeed, which favours CEM). Going from linear to
neural-linear closes most of the gap: JOPA is 6 points behind LeWM with 36×
cheaper planning.

# Amendment 5: tuned neural-linear JOPA — within 2.4 points of LeWM

Prior mean = gradient-trained head (λ = 1e4), 8 starts, 20 Gauss–Newton
iterations; chosen on dev seeds 100 / 101 (100 % / 96 %, LeWM 100 % / 100 %).

| full data, every-step cost, replan every block | 42 | 1 | 2 | 3 | 4 | mean ± std | median solve |
|---|---|---|---|---|---|---|---|
| LeWM + CEM | 98 | 100 | 100 | 100 | 100 | 99.6 ± 0.9 % | 2.0 s |
| JOPA neural-linear, amendment 5 | 94 | 96 | 100 | 100 | 96 | **97.2 ± 2.7 %** | **0.14 s** |
| JOPA neural-linear, amendment 4 (untuned) | 92 | 90 | 96 | 96 | 92 | 93.2 ± 2.7 % | 0.055 s |

LeWM is ahead or tied on every seed (mean paired gap 2.4 points); JOPA plans
14× faster.

# Amendment 6: actuator fault (shoulder wired backwards) — JOPA wins with the gain prior

Eval seeds, 50 episodes each, `gain = [-1, 1]` from the first step:

| method | 42 | 1 | 2 | 3 | 4 | mean ± std |
|---|---|---|---|---|---|---|
| LeWM frozen | 2 | 8 | 0 | 4 | 4 | 3.6 ± 3.0 % |
| JOPA frozen | 4 | 6 | 0 | 4 | 6 | 4.0 ± 2.4 % |
| *generic online adaptation* | | | | | | |
| LeWM, online fine-tuning | 32 | 34 | 40 | 50 | 42 | **39.6 ± 7.1 %** |
| JOPA, conjugate last-layer update | 14 | 16 | 16 | 10 | 18 | 14.8 ± 3.0 % |
| *per-actuator gain prior* | | | | | | |
| LeWM + gain, Adam | 66 | 70 | 80 | 64 | 74 | 70.8 ± 6.4 % |
| LeWM + gain, Laplace | 16 | 26 | 30 | 28 | 38 | 27.6 ± 7.9 % |
| JOPA, gain latent, Laplace | 74 | 76 | 84 | 80 | 86 | **80.0 ± 5.1 %** |

With the declared structural prior, JOPA beats the best LeWM variant on every
seed (mean paired gap +9.2 points), and identifies the fault exactly
(posterior g = (−1.00, 1.00) ± 0.002 after the first replan with data). The
same Laplace inference through LeWM's network converges to a biased gain
(≈ (−0.63, 0.73)) and plans worse than Adam: JOPA's model is identifiable in
its action input, LeWM's is not. Without the prior, LeWM's fine-tuning beats
JOPA's last-layer update by 25 points: generic adaptation of a 192 × 513
layer from a few hundred transitions does not work.

# Amendment 7: JOPA as factor-graph message passing — the win holds

Same model and tasks as amendments 5–6, now library code only: `jopa.fg`
(Gaussian sum-product with iterated linearization, box nodes),
`jopa.nodes.neural_linear` (matrix-normal last layer, exact conjugate
messages), `jopa.control` (reaching and actuator-gain graphs); φ and W refit
by `jopa.fit_neural_linear` (held-out R² 0.94 → 0.82 over
1–5 blocks, σ² 0.050). Eval seeds, 50 episodes each:

| | 42 | 1 | 2 | 3 | 4 | mean ± std | median solve |
|---|---|---|---|---|---|---|---|
| *shoulder wired backwards* | | | | | | | |
| JOPA, gain graph (amendment 7) | 72 | 74 | 80 | 80 | 80 | **77.2 ± 3.9 %** | 0.27 s |
| LeWM + gain, Adam (best LeWM, amendment 6) | 66 | 70 | 80 | 64 | 74 | 70.8 ± 6.4 % | |
| JOPA prototype (amendment 6) | 74 | 76 | 84 | 80 | 86 | 80.0 ± 5.1 % | |
| *nominal* | | | | | | | |
| JOPA (amendment 7) | 96 | 96 | 98 | 98 | 96 | 96.8 ± 1.1 % | 0.19 s |
| LeWM + CEM | 98 | 100 | 100 | 100 | 100 | 99.6 ± 0.9 % | 2.0 s |
| JOPA prototype (amendment 5) | 94 | 96 | 100 | 100 | 96 | 97.2 ± 2.7 % | 0.14 s |

Under the fault JOPA beats the best LeWM variant on four seeds and ties one
(mean paired gap +6.4 points); the gain posterior reaches (−1.02, 1.00) ±
0.002. Nominal is unchanged: 2.8 points behind LeWM at about a tenth of the
planning time. The rebuild is within noise of the prototype: on dev seed
100 the same configuration spans 76–86 % across solver seeds, the
prototype 82–88 %.

Two findings from the rebuild, both on dev seeds:

- *Lower model cost is not better control.* A tanh-squashed planner reaches
  a 3.7× lower model goal cost than the box-constrained one on 200 held-out
  problems, yet succeeds less under the fault (54 / 62 vs 78 / 74 %), and
  more iterations make it worse (66 %): it exploits model error. The box's
  saturated first moves are robust. W's epistemic variance along the plans
  (σ²φᵀΛ⁻¹φ ≈ 6e-4 per dimension) is far too small to flag this.
- *The harness shrinks the batch as episodes finish*, which made every
  replan recompile (9 s). Padding to a fixed batch, with zero-weight
  transitions in the gain plate, gives one compile per run.

# Library cleanup after amendment 7 — reruns

The library was briefly reduced to `jopa.fg` plus model nodes (`jopa.nodes`,
`jopa.planning`, `jopa.learn`), deleting the ContinuousTransition engine and
its demos. That deletion was reverted: the CT engine is JOPA's core again and
the neural-linear stack sits beside it. No configuration changed. Checks:

- On fixed inputs (CPU) the planner, its energies and the gain posterior
  are bit-identical to the amendment-7 code.
- `encode.py` + `train.py` from scratch reproduce the registered model:
  held-out R² 0.94 0.90 0.87 0.85 0.82, σ² 5.01e-2.
- The amendment-7 code is deterministic on the GPU: dev rerun of fault seed 1
  gave 74 % twice with identical gain traces.
- Final code, eval seeds 42 / 1 / 2 / 3 / 4: nominal 96 / 96 / 98 / 98 / 96
  (identical to amendment 7); fault 74 / 74 / 84 / 84 / 76 = 78.4 ± 5.0 %
  (amendment 7: 77.2 ± 3.9 %). The fault path compiles to a different GPU
  program and the closed loop amplifies rounding, so seeds move.
- New control row, JOPA without adaptation under the fault:
  6 / 10 / 0 / 4 / 8 = 5.6 ± 3.8 %.

# Amendment 8: the actuator learned by the ContinuousTransition node — the win holds

Not pre-registered: written after the runs, which are reported in full.

Question: can the result rest on the CT node's VMP rules rather than on a
separate gain node? The transition stays neural-linear (amendment 7, same
`jopa_nl.pkl`); the actuator becomes a CT node.

Two placements failed first, on held-out data:

- *CT for the whole transition*, `x' − x − W₀ᵀφ(s) ~ N(a + B(u ⊗ ψ(s)), W⁻¹)`
  with q(a) q(B) q(W) by the CT rules: the fault would be
  B → B·diag(g ⊗ 1). Affine in u, the latent cannot be predicted: 5k-step
  fits reach held-out R² 0.58 / 0.62 / 0.77 at k = 1 with 4 / 16 / 64
  modulating features, against 0.92 for neural-linear; k = 1 was 0.56 after
  20k steps. On LeWM's latent the command's effect is strongly nonlinear.
- *CT on the executed actuation*, `v ~ N(A u, W⁻¹)` with the transition's
  message to v and mean-field q(v) q(A) q(W). Per transition, v is not
  identifiable (MAP v̂ correlates 0.11 with the true command, posterior sd
  ≈ 1), so mean-field VMP carries almost no information to A: on dataset
  transitions with the shoulder flipped, A shrank toward 0 (−0.27 / −0.54)
  where the gain node gives −0.98. A first version of this reached eval
  seed 42 (4 %) and started seed 1 before it was stopped; both seeds were
  rerun with the final code below.

What works: linearized at E[A], the transition is linear in A,
`x' − f(s, v₀) + J v₀ = Σ A_pq X_pq + ε`, so every latent coordinate of every
observed transition is a CT observation `y ~ N(aᵀx, w⁻¹)` with known
regressor x, a = vec A (2 × 2: reversal, swaps and cross-talk) and the
transition noise precision w. q(a) and q(w) are the CT :a and :W messages
from the summed moments (`jopa.ActuatorCT`, `solver=jopa_ct`); prior
vec A ~ N(vec I, I), E[w] = 1; five sweeps per replan, each relinearized.
On the flipped dataset transitions it tracks the gain node replan by replan
(−0.88 / −0.93 / −0.96 / −0.99 vs −0.88 / −0.93 / −0.96 / −0.98).

| fault, eval seeds | 42 | 1 | 2 | 3 | 4 | mean ± std | median solve |
|---|---|---|---|---|---|---|---|
| JOPA, CT actuator node | 74 | 72 | 86 | 88 | 84 | **80.8 ± 7.3 %** | 0.24 s |
| JOPA, gain node (amendment 7) | 72 | 74 | 80 | 80 | 80 | 77.2 ± 3.9 % | 0.27 s |
| LeWM + gain, Adam | 66 | 70 | 80 | 64 | 74 | 70.8 ± 6.4 % | 10.7 s |

Ahead of the best LeWM variant on all five seeds (mean paired gap +10.0
points). The first update, from 50 transitions, reads
A = [[−1.03, 0.04], [0.00, 1.04]] ± 0.005 on seed 42 and ends at
[[−1.01, 0.04], [−0.00, 1.03]] ± 0.003. Against the gain node the CT node
is +3.6 points, within seed noise.

On the healthy arm the same adaptive solver (`jopa_ct_nominal`) gets
96 / 96 / 100 / 98 / 94 = 96.8 ± 2.3 % at 0.23 s, equal to JOPA without
adaptation (96.8 ± 1.1 %): q(A) stays at the identity and costs nothing.

# Final code — reruns

`ActuatorCT` became `jopa.Actuator`, the only actuator node (the gain node
of amendments 6–7 was removed), and `solver=jopa_adaptive` uses it. Every
JOPA row was rerun with this code; LeWM rows are the registered logs.

| eval seeds | 42 | 1 | 2 | 3 | 4 | mean ± std |
|---|---|---|---|---|---|---|
| fault, JOPA (`jopa_fault`) | 72 | 72 | 80 | 86 | 84 | 78.8 ± 6.6 % |
| healthy, JOPA (`jopa`) | 96 | 96 | 100 | 98 | 94 | 96.8 ± 2.3 % |
| fault, no adaptation (`jopa_frozen`) | 6 | 12 | 0 | 6 | 6 | 6.0 ± 4.2 % |

The amendment-8 run of the same model gave 80.8 %; the rename compiles a
different GPU program and the closed loop amplifies rounding, as in the
cleanup reruns above. Against LeWM + gain (Adam) the fault row is ahead on
four seeds and tied on one (+8.0 points mean paired gap).

# Review fixes — reruns

The solver's timer now waits for the plan (`jax.block_until_ready`); before,
it stopped at dispatch and missed ~20 ms per replan. Every JOPA row was rerun
with the PR code; LeWM rows are the registered logs.

| eval seeds | 42 | 1 | 2 | 3 | 4 | mean ± std | median solve |
|---|---|---|---|---|---|---|---|
| fault, JOPA (`jopa_fault`) | 78 | 72 | 86 | 82 | 84 | 80.4 ± 5.5 % | 0.25 s |
| healthy, JOPA (`jopa`) | 96 | 96 | 100 | 98 | 96 | 97.2 ± 1.8 % | 0.25 s |
| fault, no adaptation (`jopa_frozen`) | 2 | 12 | 0 | 6 | 6 | 5.2 ± 4.6 % | 0.07 s |

A second run of fault seed 42 on the same code gave 74. Against LeWM + gain
(Adam) the fault row is ahead on all five seeds (+9.6 points mean paired gap).
