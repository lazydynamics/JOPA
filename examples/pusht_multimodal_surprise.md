# PushT visual-world latent surprise demo

This example is the first real-data follow-up to the `JointModel` block API.
It uses a single visual/world latent block, matching the existing JOPA pattern:

```text
observation.environment_state -> q(z_t)
z[t+1] ~ A z[t] + B u[t]
```

The default dataset is `lerobot/pusht_keypoints`. Its
`observation.environment_state` field is a 16-D keypoint/world state derived
from the PushT scene. The script projects that feature with PCA, learns
controlled latent dynamics, then scores holdout episodes with online
predictive-vs-corrected belief gaps.

## Setup

Install the benchmark extras:

```bash
uv pip install -e ".[benchmarks,test]"
```

Run a small smoke experiment:

```bash
uv run python examples/pusht_multimodal_surprise.py \
  --download \
  --episodes 40 \
  --stride 3 \
  --latent-dim 6 \
  --vmp-iters 10 \
  --top-k 5
```

Run the fuller default experiment:

```bash
uv run python examples/pusht_multimodal_surprise.py \
  --download \
  --episodes 80 \
  --stride 2 \
  --latent-dim 8 \
  --vmp-iters 20 \
  --seed 0
```

## Local files

The script downloads PushT data into:

```text
data/pusht_keypoints/
```

It writes run outputs into:

```text
outputs/pusht_surprise/
  summary.json
  train_metrics.csv
  test_metrics.csv
  surprise_vs_reward.png
  top_surprise_episodes.png
  episode_diagnostics.png
```

Both `data/` and nested `outputs/` are ignored by git. Do not commit dataset
files or run outputs unless the project explicitly changes that policy.

## Metrics

Per episode, the script reports:

- one-step transition `R2`, `MAE`, `RMSE`;
- normalized innovation squared, `NIS/d`;
- Gaussian negative log likelihood;
- mean, p95, and max `KL(corrected || predicted)`;
- alert rate against the train p95 KL threshold;
- terminal belief distance;
- correlation between surprise/terminal distance and PushT reward.

The point is not to beat a policy benchmark yet. The point is to test whether
JOPA's existing message-passing API can expose useful uncertainty and
distribution-shift diagnostics on a real robotics dataset.

The default split is shuffled deterministically with `--seed 0`. Use
`--no-shuffle` only when debugging episode-order effects.

`next.success` is not always populated usefully in the keypoint parquet, so the
script also treats `max_reward >= --success-reward-threshold` as success. The
default threshold is `0.95`.
