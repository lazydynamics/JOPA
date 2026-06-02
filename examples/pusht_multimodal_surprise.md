# PushT visual-world latent surprise demo

This example is a hypothesis test for real-data multi-modal JOPA diagnostics.
It compares a visual/world latent baseline:

```text
observation.environment_state -> q(z_t)
z[t+1] ~ A z[t] + B u[t]
```

against a coupled proprio + visual/world model:

```text
observation.state -> q(p_t)
observation.environment_state -> q(z_t)
p[t+1] ~ A_p p[t] + B_p u[t]
z[t+1] ~ A_z z[t] + B_z u[t]
z_t ~= M p_t + noise
```

The default dataset is `lerobot/pusht_keypoints`. Its
`observation.environment_state` field is a 16-D keypoint/world state derived
from the PushT scene. The script projects that feature with PCA, learns
controlled latent dynamics, then scores holdout episodes with online
predictive-vs-corrected belief gaps and coupling inconsistency.

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

Run the fuller comparison experiment used for the checked-in metrics:

```bash
uv run python examples/pusht_multimodal_surprise.py \
  --download \
  --episodes 206 \
  --stride 3 \
  --latent-dim 8 \
  --vmp-iters 12 \
  --seed 0 \
  --success-reward-threshold 0.92
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
  one_block_train_metrics.csv
  one_block_test_metrics.csv
  two_block_train_metrics.csv
  two_block_test_metrics.csv
  surprise_vs_reward.png
  top_surprise_episodes.png
  episode_diagnostics.png
  pca_explained_variance.png
```

Dataset files stay ignored. The current branch intentionally includes the small
CSV/PNG/JSON run artifacts under `outputs/pusht_surprise/` so reviewers can see
the exact comparison result without re-running the example.

## Metrics

Per episode, the script reports:

- one-step transition `R2`, `MAE`, `RMSE`;
- normalized innovation squared, `NIS/d`;
- Gaussian negative log likelihood;
- mean, p95, and max `KL(corrected || predicted)`;
- alert rate against the train p95 KL threshold;
- terminal belief distance;
- two-block coupling KL;
- correlation between surprise/terminal distance and PushT reward.

The point is not to beat a policy benchmark yet. The point is to test whether
JOPA's existing message-passing API can expose useful uncertainty and
distribution-shift diagnostics on a real robotics dataset.

The default split is shuffled deterministically with `--seed 0`. Use
`--no-shuffle` only when debugging episode-order effects.

`next.success` is false for every episode in the keypoint parquet tested here,
so the comparison treats `max_reward >= --success-reward-threshold` as a success
proxy. The checked-in run uses `0.92`, which yields 5 high-reward holdout
episodes on the deterministic split.
