"""PushT visual-world latent surprise diagnostics.

This example uses the public `lerobot/pusht_keypoints` parquet dataset. The
16-D `observation.environment_state` keypoint/world feature is treated as a
vision-derived observation, projected to a latent Gaussian message, and modeled
as one controlled JOPA block:

    observation.environment_state -> q(z_t)
    z[t+1] ~ A z[t] + B u[t]

The script reports transition quality and online predictive-vs-corrected belief
surprise, using Rich for progress and terminal tables. Dataset files and run
outputs stay outside version control.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from jopa import Block, Frozen, Gaussian, JointModel, LearnedLinear, near_identity_prior
from jopa.diagnostics import (
    block_precision,
    filter_with_diagnostics,
    transition_metrics,
    transition_residuals,
)


REPO_ID = "lerobot/pusht_keypoints"
DATA_FILE = "data/chunk-000/file-000.parquet"
META_INFO = "meta/info.json"
DEFAULT_FEATURE = "observation.environment_state"


@dataclass
class Episode:
    index: int
    z: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    success: bool


def _require_optional_deps():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('Install benchmark deps first: uv pip install -e ".[benchmarks]"') from exc
    return pq


def _download(url: str, path: Path, console: Console):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": "jopa-pusht-example"})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"download {path.name}", total=total or None)
            with path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))


def ensure_dataset(root: Path, repo_id: str, download: bool, console: Console):
    data_path = root / DATA_FILE
    info_path = root / META_INFO
    if download:
        base = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
        _download(f"{base}/{DATA_FILE}", data_path, console)
        _download(f"{base}/{META_INFO}", info_path, console)
    if not data_path.exists():
        raise SystemExit(
            f"Missing {data_path}. Re-run with --download or place the PushT parquet file there."
        )
    return data_path


def _as_2d(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == object:
        arr = np.asarray([np.asarray(v, dtype=np.float32) for v in values], dtype=np.float32)
    return arr.astype(np.float32)


def load_episodes(
    path: Path,
    feature: str,
    max_episodes: int | None,
    stride: int,
    success_reward_threshold: float,
    console: Console,
) -> list[Episode]:
    pq = _require_optional_deps()
    columns = [feature, "action", "episode_index", "frame_index", "next.reward", "next.success"]
    table = pq.read_table(path, columns=columns)
    rows = table.to_pydict()
    episode_ids = np.asarray(rows["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(rows["frame_index"], dtype=np.int64)
    feature_values = _as_2d(rows[feature])
    actions = _as_2d(rows["action"])
    rewards = np.asarray(rows["next.reward"], dtype=np.float32)
    successes = np.asarray(rows["next.success"], dtype=bool)

    episodes: list[Episode] = []
    for episode in sorted(np.unique(episode_ids).tolist()):
        mask = episode_ids == episode
        order = np.argsort(frame_ids[mask])
        idx = np.where(mask)[0][order]
        if stride > 1:
            idx = idx[::stride]
        z = feature_values[idx]
        action = actions[idx]
        reward = rewards[idx]
        success = bool(np.any(successes[idx]) or np.max(reward) >= success_reward_threshold)
        if len(z) < 4:
            continue
        episodes.append(
            Episode(
                index=int(episode),
                z=z,
                action=action[: len(z) - 1],
                reward=reward,
                success=success,
            )
        )
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    console.log(f"loaded {len(episodes)} episodes from {path}")
    return episodes


def split_episodes(episodes: list[Episode], train_frac: float, seed: int, shuffle: bool):
    episodes = list(episodes)
    if shuffle:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(episodes))
        episodes = [episodes[i] for i in order]
    n_train = int(round(len(episodes) * train_frac))
    n_train = min(max(n_train, 1), max(len(episodes) - 1, 1))
    return episodes[:n_train], episodes[n_train:]


def fit_normalizer(seqs):
    x = np.concatenate(seqs, axis=0)
    return x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-6


def fit_pca(seqs, dim: int):
    x = np.concatenate(seqs, axis=0)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dim].T
    z = centered @ components
    scale = z.std(axis=0, keepdims=True) + 1e-6
    explained = np.var(z, axis=0).sum() / max(np.var(centered, axis=0).sum(), 1e-12)
    return mean, components, scale, float(explained)


def project(x, mean, components, scale):
    return ((x - mean) @ components / scale).astype(np.float32)


def transform_episodes(train_eps, test_eps, latent_dim: int):
    z_mean, z_components, z_scale, explained = fit_pca([ep.z for ep in train_eps], latent_dim)
    a_mean, a_std = fit_normalizer([ep.action for ep in train_eps])
    for ep in train_eps + test_eps:
        ep.z = project(ep.z, z_mean, z_components, z_scale)
        ep.action = ((ep.action - a_mean) / a_std).astype(np.float32)
    return explained


def msg(x, precision: float):
    x = jnp.asarray(x)
    lam = precision * jnp.eye(x.shape[0])
    return Gaussian(eta=lam @ x, lam=lam)


def learn_model(train_eps, latent_dim: int, action_dim: int, precision: float, vmp_iters: int, console: Console):
    block = Block(
        "z",
        LearnedLinear(
            dim=latent_dim,
            du=action_dim,
            n_iterations=vmp_iters,
            **near_identity_prior(latent_dim, cov=0.5),
        ),
        observe=Frozen(lambda x: msg(x, precision)),
    )
    model = JointModel([block])
    trajectories = [
        {"z": [jnp.asarray(x) for x in ep.z], "control": [jnp.asarray(u) for u in ep.action]}
        for ep in train_eps
    ]
    with console.status("learning controlled latent dynamics with VMP"):
        model.learn(trajectories)
    return model, block


def terminal_stats(episodes):
    finals = np.stack([ep.z[-1] for ep in episodes])
    return finals.mean(axis=0), finals.var(axis=0) + 1e-4


def terminal_distance(z, terminal):
    mean, var = terminal
    return float(np.sqrt(np.mean((z - mean) ** 2 / np.maximum(var, 1e-6))))


def episode_rows(model, block, episodes, precision, terminal, console: Console, label: str):
    W = np.asarray(block_precision(block))
    train_kl = []
    rows = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task(label, total=len(episodes))
        for ep in episodes:
            row = score_episode(model, block, ep, W, terminal)
            rows.append(row)
            train_kl.extend(row["_kls"])
            progress.update(task, advance=1)
    threshold = float(np.quantile(np.asarray(train_kl), 0.95)) if train_kl else math.inf
    for row in rows:
        kls = np.asarray(row.pop("_kls"), dtype=np.float64)
        row["alert_rate"] = float(np.mean(kls > threshold)) if len(kls) else 0.0
    return rows, threshold


def score_episode(model, block, ep: Episode, W, terminal):
    truth, _, residual = transition_residuals(block.transition.A, block.transition.B, ep.z, ep.action)
    tm = transition_metrics(residual, truth, W)
    stream = []
    zero_u = np.zeros(ep.action.shape[-1], dtype=np.float32)
    for t, z in enumerate(ep.z):
        stream.append({"z": z, "control": zero_u if t == 0 else ep.action[t - 1]})
    _, diagnostics = filter_with_diagnostics(model, stream)
    gaps = [d.predictive_corrected for d in diagnostics if d.block == "z" and d.predictive_corrected is not None]
    kls = np.asarray([g.kl for g in gaps], dtype=np.float64)
    mahals = np.asarray([g.mahalanobis for g in gaps], dtype=np.float64)
    traces = np.asarray([g.trace_after for g in gaps], dtype=np.float64)
    row = {
        "episode": ep.index,
        "success": ep.success,
        "frames": len(ep.z),
        "max_reward": float(np.max(ep.reward)),
        "final_reward": float(ep.reward[-1]),
        "r2": tm.r2,
        "mae": tm.mae,
        "rmse": tm.rmse,
        "mean_nis_per_dim": tm.mean_nis_per_dim,
        "mean_nll": tm.mean_nll,
        "mean_kl": float(np.mean(kls)) if len(kls) else 0.0,
        "p95_kl": float(np.quantile(kls, 0.95)) if len(kls) else 0.0,
        "max_kl": float(np.max(kls)) if len(kls) else 0.0,
        "mean_mahalanobis": float(np.mean(mahals)) if len(mahals) else 0.0,
        "mean_trace": float(np.mean(traces)) if len(traces) else 0.0,
        "terminal_dist": terminal_distance(ep.z[-1], terminal),
    }
    row["surprise_score"] = row["mean_kl"] + 0.25 * row["p95_kl"] + row["mean_nis_per_dim"]
    row["_kls"] = kls.tolist()
    return row


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _corr(xs, ys):
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if len(xs) < 2 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def render_plots(out_dir: Path, rows, console: Console):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rewards = [r["max_reward"] for r in rows]
    surprise = [r["surprise_score"] for r in rows]
    terminal = [r["terminal_dist"] for r in rows]
    episodes = [r["episode"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(rewards, surprise, c=terminal, cmap="viridis", s=45)
    ax.set_xlabel("max reward")
    ax.set_ylabel("JOPA surprise score")
    ax.set_title("PushT holdout surprise vs reward")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("terminal belief distance")
    fig.tight_layout()
    fig.savefig(out_dir / "surprise_vs_reward.png", dpi=160)
    plt.close(fig)

    top = sorted(rows, key=lambda r: r["surprise_score"], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(r["episode"]) for r in top], [r["surprise_score"] for r in top])
    ax.set_xlabel("episode")
    ax.set_ylabel("surprise score")
    ax.set_title("Top PushT surprise episodes")
    fig.tight_layout()
    fig.savefig(out_dir / "top_surprise_episodes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, surprise, label="surprise")
    ax.plot(episodes, terminal, label="terminal distance")
    ax.set_xlabel("episode")
    ax.set_title("Holdout episode diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "episode_diagnostics.png", dpi=160)
    plt.close(fig)
    console.log(f"wrote plots to {out_dir}")


def print_table(console: Console, title: str, rows, sort_key: str, top_k: int):
    table = Table(title=title)
    for col in ["rank", "episode", "reward", "success", "surprise", "r2", "nis/d", "KL", "alerts", "terminal"]:
        table.add_column(col)
    for rank, row in enumerate(sorted(rows, key=lambda r: r[sort_key], reverse=True)[:top_k], start=1):
        table.add_row(
            str(rank),
            str(row["episode"]),
            f"{row['max_reward']:.3f}",
            str(row["success"]),
            f"{row['surprise_score']:.3f}",
            f"{row['r2']:.3f}",
            f"{row['mean_nis_per_dim']:.3f}",
            f"{row['mean_kl']:.3f}",
            f"{row['alert_rate']:.1%}",
            f"{row['terminal_dist']:.3f}",
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/pusht_keypoints"))
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--feature", default=DEFAULT_FEATURE)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--precision", type=float, default=100.0)
    parser.add_argument("--vmp-iters", type=int, default=20)
    parser.add_argument("--success-reward-threshold", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pusht_surprise"))
    args = parser.parse_args()

    console = Console()
    console.print(Panel.fit("PushT visual-world latent JOPA diagnostics", style="bold cyan"))
    data_path = ensure_dataset(args.data_root, args.repo_id, args.download, console)
    episodes = load_episodes(
        data_path,
        args.feature,
        args.episodes,
        args.stride,
        args.success_reward_threshold,
        console,
    )
    if len(episodes) < 4:
        raise SystemExit("Need at least four episodes for a train/test split")
    train_eps, test_eps = split_episodes(episodes, args.train_frac, args.seed, not args.no_shuffle)
    explained = transform_episodes(train_eps, test_eps, args.latent_dim)
    model, block = learn_model(
        train_eps,
        args.latent_dim,
        train_eps[0].action.shape[-1],
        args.precision,
        args.vmp_iters,
        console,
    )
    terminal = terminal_stats(train_eps)
    train_rows, kl_threshold = episode_rows(model, block, train_eps, args.precision, terminal, console, "score train episodes")
    test_rows, _ = episode_rows(model, block, test_eps, args.precision, terminal, console, "score holdout episodes")

    summary = {
        "repo_id": args.repo_id,
        "feature": args.feature,
        "episodes": len(episodes),
        "train": len(train_eps),
        "test": len(test_eps),
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "stride": args.stride,
        "latent_dim": args.latent_dim,
        "pca_explained_variance": explained,
        "train_kl_p95": kl_threshold,
        "success_reward_threshold": args.success_reward_threshold,
        "test_successes": int(sum(1 for r in test_rows if r["success"])),
        "test_mean_surprise": float(np.mean([r["surprise_score"] for r in test_rows])),
        "test_mean_r2": float(np.mean([r["r2"] for r in test_rows])),
        "corr_surprise_reward": _corr([r["surprise_score"] for r in test_rows], [r["max_reward"] for r in test_rows]),
        "corr_terminal_reward": _corr([r["terminal_dist"] for r in test_rows], [r["max_reward"] for r in test_rows]),
    }

    console.print(
        Panel(
            "\n".join(
                [
                    f"train/test: {summary['train']}/{summary['test']}",
                    f"latent_dim: {args.latent_dim}",
                    f"PCA explained variance: {explained:.3f}",
                    f"mean test surprise: {summary['test_mean_surprise']:.3f}",
                    f"mean test R2: {summary['test_mean_r2']:.3f}",
                    f"corr(surprise, reward): {summary['corr_surprise_reward']:.3f}",
                ]
            ),
            title="run summary",
        )
    )
    print_table(console, "Top surprise holdout episodes", test_rows, "surprise_score", args.top_k)
    print_table(console, "Largest terminal belief misses", test_rows, "terminal_dist", args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "train_metrics.csv", train_rows)
    _write_csv(args.output_dir / "test_metrics.csv", test_rows)
    _write_json(args.output_dir / "summary.json", summary)
    render_plots(args.output_dir, test_rows, console)
    console.print(f"[green]wrote metrics to {args.output_dir}[/green]")


if __name__ == "__main__":
    main()
