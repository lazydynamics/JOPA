# JOPA vs LeWM on stable-worldmodel Reacher

The harness (`eval_wm.py` from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel))
and LeWM's released Reacher checkpoint run unmodified; JOPA is a drop-in
`solver=`. 50 episodes per seed, eval seeds 42, 1, 2, 3, 4, replan every
action block.

| | 42 | 1 | 2 | 3 | 4 | mean ± std | median solve |
|---|---|---|---|---|---|---|---|
| **shoulder wired backwards** | | | | | | | |
| JOPA | 78 | 72 | 86 | 82 | 84 | **80.4 ± 5.5 %** | 0.25 s |
| LeWM + gain, Adam | 66 | 70 | 80 | 64 | 74 | 70.8 ± 6.4 % | 10.75 s |
| LeWM, online fine-tuning | 32 | 34 | 40 | 50 | 42 | 39.6 ± 7.1 % | 14.86 s |
| LeWM + gain, Laplace | 16 | 26 | 30 | 28 | 38 | 27.6 ± 7.9 % | 16.19 s |
| JOPA, no adaptation | 2 | 12 | 0 | 6 | 6 | 5.2 ± 4.6 % | 0.07 s |
| LeWM, no adaptation | 2 | 8 | 0 | 4 | 4 | 3.6 ± 3.0 % | 16.52 s |
| **healthy arm** | | | | | | | |
| JOPA | 96 | 96 | 100 | 98 | 96 | 97.2 ± 1.8 % | 0.25 s |
| LeWM + CEM | 98 | 100 | 100 | 100 | 100 | 99.6 ± 0.9 % | 1.99 s |

Median solve is per replan, for the batch of still-running episodes, with the
plan synchronized off the GPU. JOPA's time covers encoding, the `q(A)` update
and the plan; LeWM's is the harness's CEM timer (encoding and the plan), so the
adaptive LeWM rows exclude their fine-tuning or gain step and are lower bounds. Reruns of identical code differ by a few
episodes per seed (GPU rounding, amplified by the closed loop): seed 42 of the
fault row gave 78 and 74, and three full runs of it averaged 77.2–80.4 %. JOPA
runs the same solver in both settings: it adapts `q(A)` online with the
ContinuousTransition rules (`jopa.Actuator`) and plans through `E[A]`.

```bash
bash setup.sh                       # once, on a CUDA machine: harness, checkpoint, dataset
bash reproduce.sh train             # encode the dataset, fit the dynamics
bash reproduce.sh                   # every row; one log per row and seed
python report.py --runs ~/jopa-bench/runs --figure ../../assets
```

| file | |
|---|---|
| `jopa_solver.py` | the adapter: LeWM's encoder, then `jopa.Actuator` and `jopa.Planner` |
| `encode.py`, `train.py` | LeWM embeddings of 1000 / 200 episodes; `jopa.fit_neural_linear` on them |
| `lewm_baselines.py` | LeWM with online fine-tuning, and with a gain on its action input (Adam, Laplace) |
| `shift.py`, `config/` | the fault (shoulder torque × −1 before the simulator) and solver configs |
| `objectives.py` | LeWM's goal cost on every predicted step (its best dev configuration) |
| `reproduce.sh`, `report.py`, `montage.py` | runs → `results.json` → figure and episode montage |
| `PROTOCOL.md`, `LOG.md` | the pre-registered protocol and every run, failed variants included |
