# Joint-State Model — Proprioception + Image Latent (corrected)

## The error in the previous version

The previous model collapsed the image latent $z$ onto the angle $\theta$ and made
vision a noisy observation of $\theta$ (with a "linkage" $z \leftrightarrow \theta$).
**Wrong.** The image latent stays in the state as its own component. We do *not* try
to map $z$ to $\theta$.

## Joint state

$$
x_t = \big[\, \underbrace{\theta_t,\ \omega_t}_{\text{proprio}},\ \ \underbrace{z_t}_{\text{image latent }(\mathbb R^{d_z})} \,\big]
$$

- $(\theta_t,\omega_t)$ — proprioceptive part; we (can) know its physics.
- $z_t$ — the VAE-encoded latent of the image; we do **not** know its dynamics.

## Likelihood — two parts, on their own slices of the state

$$
\underbrace{m_t \mid \theta_t,\omega_t \sim \mathcal N\big((\theta_t,\omega_t),\,R\big)}_{\text{proprioception}}
\qquad\qquad
\underbrace{y_t \mid z_t \sim p_{\text{dec}}(y_t \mid z_t)}_{\text{vision (VAE)}}
$$

The VAE encoder supplies the message on $z_t$; proprioception supplies the message on
$(\theta_t,\omega_t)$. Each factor touches only its own block of $x_t$.

## Transition — the decision: how to combine, and what the CT node learns

The CT node exists to **learn the dynamics we don't know**. Two ways to combine:

### (a) One fused vector — learn *all* the dynamics
One CT factor over the whole joint state:
$$
x_{t+1} \sim \mathcal N\!\big(A\,x_t + B\,\tau_t,\ W^{-1}\big),\qquad
x = [\theta,\omega,z]
$$
$A$ is $(2{+}d_z)\times(2{+}d_z)$, learned by the CT `:a/:b/:W` rules — it captures proprio
physics, image-latent dynamics, **and their coupling**, all from data.

- *Pros:* uniform, learns the cross-coupling between physics and appearance for free.
- *Cons:* re-learns dynamics we already know (proprio), higher-dimensional, no use of the
  known physics structure.

### (b) Split — learn image dynamics, *filter* proprio
Block-structured transition:
$$
\underbrace{(\theta_{t+1},\omega_{t+1}) = \text{known physics}(\theta_t,\omega_t,\tau_t)}_{\text{filter only — no learning}}
\qquad
\underbrace{z_{t+1} \sim \mathcal N(A_z\,[z_t,\ \cdot\,] + B_z\tau_t,\ W_z^{-1})}_{\text{CT node learns this}}
$$

- Proprio block: known gray-box physics → just **filtered** (Kalman/BP), nothing learned.
- Image block: unknown → the **CT node learns** $A_z, B_z, W_z$.
- (Open sub-question: does $z_{t+1}$ depend only on $z_t$, or also on the proprio state /
  control — i.e. is there cross-coupling from $(\theta,\omega)$ into the $z$-dynamics?)

- *Pros:* uses the known physics where we have it; only learns the genuinely-unknown part;
  matches "use the CT node for what we don't know."
- *Cons:* must specify the block structure / coupling explicitly.

## Three tasks on this one graph (unchanged in spirit)

| Task | Inferred | Fixed / observed |
|---|---|---|
| Learn dynamics | (a) $A,B,W$ over joint $x$ &nbsp;/&nbsp; (b) $A_z,B_z,W_z$ for the image block | $x_t$ (from proprio + VAE), $\tau_t$ |
| Filter / estimate | $x_t = [\theta,\omega,z]$ | params, $m_t$, $y_t$, $\tau_t$ |
| Control | $\tau_t$ | params, current $x$, setpoint |

## Modular implementation of (b)

Each modality is a **`Block`** — a slice of the joint state with its own *transition*
and *observation*. The joint model is just a list of blocks. Each block reuses an existing
JOPA component as its engine, so there is almost no new machinery.

```
Block:
  name, dim
  transition : Transition        # how this slice evolves
  observe(data) -> Gaussian      # likelihood message onto this slice
```

Two transition kinds (the only distinction (b) needs):

```
KnownPhysics(Transition)         # proprio: (θ,ω)
  linearize(state) -> (A,B,c)    # gray-box physics, re-linearized per step
  learned = False                # FILTER only — nothing to learn

LearnedLinear(Transition)        # image latent: z
  q_a, q_W, q_b                  # conjugate posteriors
  learn(seq, ctrl)               # accumulate_vmp_messages  (== infer/rotating_digits)
  predict(belief, u)             # ct_forward
  learned = True
```

```
JointModel(blocks, coupling=None):
  learn(data)    : for each LearnedLinear block, run its CT-node VMP on that
                   block's observed sequence (proprio block: nothing).
  filter(stream) : predict + update each block; block-diagonal transition unless
                   `coupling` links slices (start independent, add coupling later).
  control(setpt) : plan torques on the controllable (proprio) block via plan().
```

For the pendulum (b):

| Block | dim | transition | observation | engine reused |
|---|---|---|---|---|
| `proprio` | 2 | `KnownPhysics(k)` | proprio reading | `physics.linearize` + filter |
| `image`   | $d_z$ | `LearnedLinear` | frozen VAE encoder | `accumulate_vmp_messages` / `infer` |

The `coupling` slot is the single extension point: `None` → two independent chains
(vision is parallel perception, doesn't yet inform $\theta$); later a cross-term
$z_{t+1}\leftarrow(\theta_t,\omega_t)$ (or a shared observation) makes vision actually
sharpen the physical-state estimate, and the same abstraction adds a 3rd/4th modality
by appending a `Block`.

## The decision to make

1. Confirm the **`Block` / `Transition` / `JointModel`** abstraction above (modular, reuses
   existing engines, no DSL).
2. Start with `coupling=None` (independent proprio + image chains), or wire the
   $z\leftrightarrow(\theta,\omega)$ coupling now so vision immediately helps estimate $\theta$?
