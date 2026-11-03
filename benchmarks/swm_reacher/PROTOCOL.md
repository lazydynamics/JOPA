# JOPA vs LeWM on stable-worldmodel Reacher — protocol

Written before any closed-loop run. Changes after the first run are listed at
the bottom with the date and the reason; nothing above that list is edited.

## Question

With the encoder, data, environment, goals and success test held fixed, does
replacing LeWM's learned predictor and CEM planner with JOPA's conjugate
linear-Gaussian dynamics and exact planning change goal-reaching success — at
nominal dynamics, under dynamics shift, with less data — and at what cost per
decision?

## Held fixed (taken from upstream, unmodified)

| | source |
|---|---|
| harness | `galilai-group/stable-worldmodel`, commit pinned in `setup.sh` |
| model, eval config | `lucas-maes/le-wm`, commit pinned in `setup.sh`; `config/eval/reacher.yaml` |
| environment | `swm/ReacherDMControl-v0`, task `qpos_match` |
| success | every joint within 0.05 rad of the goal `qpos` within the budget |
| episodes | start state from `dmc/reacher_random`, goal 25 env steps later in the same trajectory, budget 50 env steps |
| episode sampling | `eval.py`'s own sampler; seeds 42 (upstream) and 1, 2, 3, 4; 50 episodes per seed |
| encoder | LeWM's released Reacher checkpoint (`quentinll/lewm-reacher`), frozen, CLS token + projector, 192-d |
| action | LeWM's block: 5 env steps (`action_block: 5`), so one decision is a 10-d action |
| hardware | one RTX 5090; every method timed on the same machine |

`eval_jopa.py` is `le-wm/eval.py` with only the policy construction changed;
`diff` against upstream is part of the release.

## JOPA's side

- **Sensor.** LeWM's embedding of the current frame, projected to `d` dims by
  PCA fit on the training episodes. Observation precision per dimension is
  the inverse residual variance of the fitted dynamics on training episodes.
  No gradient step anywhere.
- **State.** The last two projected embeddings at block boundaries
  (`LearnedDelayLinear`, delay 2), so velocity is in the state.
- **Dynamics.** `q(A, B, W)` learned by VMP from the training episodes.
  Online: `Agent` with local refits from the encoded training replay.
- **Planning.** `Agent`, exact inference, goal = projected embedding of the
  goal frame.
- **Free choices,** fixed by the gate on held-out episodes only, never on eval
  episodes: `d` ∈ {8, 16, 32}, local refit on/off, horizon ∈ {3, 5} blocks.

## Comparisons

1. **Nominal.** All methods with upstream settings. Reference rows: LeWM,
   PLDM, DINO-WM-noprop, re-run here from released checkpoints. Published
   numbers are quoted beside ours, never in place of them.
2. **Replanning cadence.** Upstream replans every 5 blocks. Every method is
   also run replanning every block, so "JOPA replans more often" is not
   confounded with "JOPA's model is better".
3. **Dynamics shift.** The harness's own Reacher variation space:
   `agent.arm_density` and `agent.finger_density` at 500 and 1500 (each alone,
   then both), applied from episode start. No method sees shifted data
   beforehand. LeWM is frozen (as published); JOPA adapts online as its agent
   always does — that difference is the claim being tested.
4. **Data.** Training episodes subsampled to 100, 300, 1000 and all, nested,
   seed 0. LeWM is retrained per subset with its own `train.py` and config;
   JOPA is refit on the same subsets with the same frozen encoder (encoder
   trained on all data — reported as such; a fully low-data row would need
   the encoder retrained too).
5. **Latency.** Median and 95th-percentile wall-clock per decision at batch 1,
   and total eval time at upstream batch size.

## Gate (before any closed-loop run)

On held-out training episodes (episode split 90/10, seed 3072, as upstream):

- reproduce LeWM's nominal success at seed 42 within ±10 points of the paper;
- JOPA's open-loop prediction R² in the projected space ≥ 0.8 at 5 blocks
  (= the 25-step goal offset).

Fail either → stop and report; no closed-loop JOPA numbers are produced.

## Reporting

Success mean ± std over the 5 seeds, every cell, including losses. One command
regenerates each table. Raw per-episode results are kept.

## Changes after the first run

- **2026-11-01, amendment 1 (after the gate failed, see `GATE.md`).** One
  further state for JOPA is allowed: a small MLP head φ on the last two
  frozen LeWM embeddings, trained once offline by gradient descent so that
  the block dynamics in its latent are linear over 5 blocks, with a linear
  read-out to the gate's metric space. JOPA discards the head's own dynamics
  and learns q(A, B, c, W) by VMP on the encoded training episodes. The
  gate metric, held-out starts and 0.8 threshold are unchanged. Free
  choice: latent size ∈ {16, 32, 64}. Reason: the PCA state fails the gate
  and a learned linearizing lift is the standard remedy; "no gradient step
  anywhere" no longer holds for this variant.
- **2026-11-01, amendment 2 (after amendment 1 failed).** JOPA's side may
  instead plan on LeWM's own predictor, linearized around the current plan
  (Gauss–Newton with exact Gaussian steps, `solvers.py`), entered through the
  harness's `solver=` slot; online adaptation under the variation-space shift
  is the claim to test. Free choices (prior precision, number of starts) are
  set on dev seed 100 only, never on the eval seeds.
- **2026-11-02, amendment 3: low data (comparison 4), before any eval seed.**
  Written after dev-seed tuning and before any eval-seed run.
  - *Data.* Nested subsets of 100 ⊂ 300 ⊂ 1000 episodes (`make_subset.py`,
    permutation seed 0) of the released 10 000.
  - *LeWM.* Retrained from scratch per subset with the harness's
    `scripts/train/lewm.py`, `data=dmc`, unchanged recipe except a matched
    budget of about 50 000 steps (100 episodes: 400 epochs; 300: 130). It is
    evaluated with its best dev-seed-100 configuration over checkpoints
    (every 100 epochs; every 30 at 300), objective (last step / every step)
    and cadence (5 blocks / 1 block).
  - *DINO-WM.* The harness's `prejepa.py` (frozen DINOv2-small, learned
    predictor), pixels only, same subsets, same selection rule. It shares
    JOPA's encoder, so it separates "JOPA wins" from "DINOv2 features win".
  - *JOPA.* `solver=dino_local` (`solvers.py`): the same frozen DINOv2-small,
    no task-data gradient step; local conjugate affine dynamics from the
    nearest replay transitions; exact least-squares planning on the
    every-step cost; replan every block. Free choices (state size,
    neighbours, horizon, priors) are set on dev seed 100 and must hold on dev
    seed 101 before any eval seed.
  - *Eval.* Seeds 42, 1, 2, 3, 4, 50 episodes each, mean ± std; median
    solve time per replan. No eval-seed number is used for any choice.
- **2026-11-02, amendment 4: neural-linear JOPA (written before any eval-seed
  result of it).** Dynamics `e' − e = W φ(e_prev, e, u)` on LeWM's frozen
  encoder: φ trained once offline on the 1000 gate training episodes
  (5-block rollout loss, `train_neural_linear.py`); W given its exact
  conjugate (matrix-normal) posterior on those features. Planning:
  Gauss–Newton on the every-step goal cost through the posterior mean,
  horizon 3, replan every block, 4 starts, 10 iterations, action precision
  0.01 (`solver=neural_linear`). These values were fixed before the first
  run and not tuned. Reference: LeWM + CEM at its best dev configuration
  (every-step cost, replan every block), full data. Same eval seeds.
- **2026-11-02, amendment 5: tuned neural-linear JOPA (after amendment 4's
  eval, before this variant's eval).** Bayesian last layer with the
  gradient-trained head as prior mean, prior precision λ = 1e4
  (`refit_neural_linear.py`); planner 8 starts, 20 Gauss–Newton iterations.
  Chosen among four variants on dev seeds 100 and 101 only. Amendment 4's
  eval numbers stay in the report; this is a second, separately registered
  configuration.
- **2026-11-02, amendment 6: actuator fault, online adaptation (before any
  eval seed of it).** `shift.py`'s `ReacherShift` with `gain = [-1, 1]`
  (shoulder wired backwards), present from the first step of every episode;
  no method sees shifted data beforehand. Transitions observed during the run
  (pooled over its parallel episodes, `history_len = 3`) are available to
  every adaptive method identically. Rows, each set on dev seed 100 only:
  - LeWM frozen (every-step cost, replan every block);
  - LeWM online fine-tuning of predictor + action encoder (100 Adam steps,
    lr 1e-3 per replan; best of 6);
  - LeWM + per-actuator gain on its action input, Adam (100 steps, lr 1e-1;
    best of 2);
  - LeWM + per-actuator gain, Laplace / Gauss–Newton (JOPA's inference on
    LeWM's network);
  - JOPA neural-linear frozen (amendment 5);
  - JOPA, conjugate last-layer update with forgetting 1e-4 (best of 3);
  - JOPA, per-actuator gain latent, Laplace (prior N(1, 1)).
  The per-actuator gain is a declared structural prior, offered to both
  models; the generic rows are the prior-free comparison.
- **2026-11-02, amendment 7: JOPA rebuilt as factor-graph message passing
  (before any eval seed of it).** Same model and tasks as amendments 5–6, now
  entirely library code (`jopa.fg`, `jopa.control`, `jopa.nodes.neural_linear`)
  behind a thin adapter (`jopa_solver.py`): planning is iterated-linearization
  sum-product on the reaching graph with box-constrained commands; the gain
  posterior is updated by the same inference on the actuator-gain graph and
  carried from one replan to the next (streaming, instead of re-fitting the
  pooled transitions). φ and W are refit with the library fitter
  (`train_jopa.py`, JAX) on the same 1000 gate episodes and recipe. Choices on
  dev seeds 100 / 101 only: box over tanh bounds (fault 78 / 74 vs 54 / 62;
  nominal 92 / 96 vs 96 / 100). Dev-seed spread over solver seeds, same
  configuration: 76–86 %. Rows: nominal (`solver=jopa`) and actuator fault
  (`solver=jopa_adaptive`), eval seeds 42, 1, 2, 3, 4; LeWM rows are those
  already reported.
