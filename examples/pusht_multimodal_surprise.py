"""PushT visual-world latent surprise diagnostics.

This example uses the public `lerobot/pusht_keypoints` parquet dataset. The
16-D `observation.environment_state` keypoint/world feature is treated as a
vision-derived observation, projected to a latent Gaussian message, and modeled
as one controlled JOPA block:

    observation.environment_state -> q(z_t)
    z[t+1] ~ A z[t] + B u[t]

It also compares an optional two-block model:

    observation.state -> q(p_t)
    observation.environment_state -> q(z_t)
    z_t ~= M p_t + noise

The script reports transition quality and online predictive-vs-corrected belief
surprise, using Rich for progress and terminal tables.
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

from jopa import Block, Frozen, Gaussian, JointModel, LearnedLinear, LinearCoupling, near_identity_prior
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
    p: np.ndarray
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
    columns = ["observation.state", feature, "action", "episode_index", "frame_index", "next.reward", "next.success"]
    table = pq.read_table(path, columns=columns)
    rows = table.to_pydict()
    episode_ids = np.asarray(rows["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(rows["frame_index"], dtype=np.int64)
    feature_values = _as_2d(rows[feature])
    states = _as_2d(rows["observation.state"])
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
        p = states[idx]
        action = actions[idx]
        reward = rewards[idx]
        success = bool(np.any(successes[idx]) or np.max(reward) >= success_reward_threshold)
        if len(z) < 4:
            continue
        episodes.append(
            Episode(
                index=int(episode),
                p=p,
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
    all_z = centered @ vt.T
    total_var = max(float(np.var(centered, axis=0).sum()), 1e-12)
    component_var = np.var(all_z, axis=0)
    cumulative = np.cumsum(component_var) / total_var
    components = vt[:dim].T
    z = all_z[:, :dim]
    scale = z.std(axis=0, keepdims=True) + 1e-6
    explained = float(cumulative[min(dim, len(cumulative)) - 1])
    return mean, components, scale, explained, cumulative.astype(float).tolist()


def project(x, mean, components, scale):
    return ((x - mean) @ components / scale).astype(np.float32)


def transform_episodes(train_eps, test_eps, latent_dim: int):
    z_mean, z_components, z_scale, explained, cumulative = fit_pca([ep.z for ep in train_eps], latent_dim)
    p_mean, p_std = fit_normalizer([ep.p for ep in train_eps])
    a_mean, a_std = fit_normalizer([ep.action for ep in train_eps])
    for ep in train_eps + test_eps:
        ep.p = ((ep.p - p_mean) / p_std).astype(np.float32)
        ep.z = project(ep.z, z_mean, z_components, z_scale)
        ep.action = ((ep.action - a_mean) / a_std).astype(np.float32)
    return explained, cumulative


def msg(x, precision: float):
    x = jnp.asarray(x)
    lam = precision * jnp.eye(x.shape[0])
    return Gaussian(eta=lam @ x, lam=lam)


def learn_block_model(name, train_eps, seq_getter, dim: int, action_dim: int, precision: float, vmp_iters: int, console: Console):
    block = Block(
        name,
        LearnedLinear(
            dim=dim,
            du=action_dim,
            n_iterations=vmp_iters,
            **near_identity_prior(dim, cov=0.5),
        ),
        observe=Frozen(lambda x: msg(x, precision)),
    )
    model = JointModel([block])
    trajectories = [
        {name: [jnp.asarray(x) for x in seq_getter(ep)], "control": [jnp.asarray(u) for u in ep.action]}
        for ep in train_eps
    ]
    with console.status(f"learning {name} controlled dynamics with VMP"):
        model.learn(trajectories)
    return model, block


def fit_coupling(train_eps, ridge: float = 1e-3):
    p = np.concatenate([ep.p for ep in train_eps], axis=0)
    z = np.concatenate([ep.z for ep in train_eps], axis=0)
    return LinearCoupling.fit("p", "z", p, z, ridge=ridge)


def learn_two_block_model(train_eps, p_dim: int, z_dim: int, action_dim: int, precision: float, vmp_iters: int, console: Console):
    _, p_block = learn_block_model("p", train_eps, lambda ep: ep.p, p_dim, action_dim, precision, vmp_iters, console)
    _, z_block = learn_block_model("z", train_eps, lambda ep: ep.z, z_dim, action_dim, precision, vmp_iters, console)
    coupling, coupling_diagnostics = fit_coupling(train_eps)
    return JointModel([p_block, z_block], coupling=coupling), p_block, z_block, coupling_diagnostics


def terminal_stats(episodes):
    finals = np.stack([ep.z[-1] for ep in episodes])
    return finals.mean(axis=0), finals.var(axis=0) + 1e-4


def terminal_distance(z, terminal):
    mean, var = terminal
    return float(np.sqrt(np.mean((z - mean) ** 2 / np.maximum(var, 1e-6))))


def episode_rows(
    model,
    block,
    episodes,
    terminal,
    console: Console,
    label: str,
    model_name: str = "one_block",
    alert_threshold: float | None = None,
):
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
            row = score_episode(model, block, ep, W, terminal, model_name)
            rows.append(row)
            train_kl.extend(row["_kls"])
            progress.update(task, advance=1)
    threshold = (
        float(alert_threshold)
        if alert_threshold is not None
        else float(np.quantile(np.asarray(train_kl), 0.95))
        if train_kl
        else math.inf
    )
    for row in rows:
        kls = np.asarray(row.pop("_kls"), dtype=np.float64)
        row["alert_rate"] = float(np.mean(kls > threshold)) if len(kls) else 0.0
    return rows, threshold


def score_episode(model, block, ep: Episode, W, terminal, model_name: str):
    truth, _, residual = transition_residuals(block.transition.A, block.transition.B, ep.z, ep.action)
    tm = transition_metrics(residual, truth, W)
    stream = []
    zero_u = np.zeros(ep.action.shape[-1], dtype=np.float32)
    for t, z in enumerate(ep.z):
        step = {"z": z, "control": zero_u if t == 0 else ep.action[t - 1]}
        if "p" in model._by_name:
            step["p"] = ep.p[t]
        stream.append(step)
    _, diagnostics = filter_with_diagnostics(model, stream)
    z_gaps = [d.predictive_corrected for d in diagnostics if d.block == "z" and d.predictive_corrected is not None]
    p_gaps = [d.predictive_corrected for d in diagnostics if d.block == "p" and d.predictive_corrected is not None]
    coupling_gaps = [d.corrected_fused for d in diagnostics if d.block == "z" and d.corrected_fused is not None]
    gaps = z_gaps
    kls = np.asarray([g.kl for g in gaps], dtype=np.float64)
    mahals = np.asarray([g.mahalanobis for g in gaps], dtype=np.float64)
    traces = np.asarray([g.trace_after for g in gaps], dtype=np.float64)
    p_kls = np.asarray([g.kl for g in p_gaps], dtype=np.float64)
    coupling_kls = np.asarray([g.kl for g in coupling_gaps], dtype=np.float64)
    row = {
        "model": model_name,
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
        "mean_p_kl": float(np.mean(p_kls)) if len(p_kls) else 0.0,
        "mean_coupling_kl": float(np.mean(coupling_kls)) if len(coupling_kls) else 0.0,
        "terminal_dist": terminal_distance(ep.z[-1], terminal),
    }
    row["surprise_score"] = (
        row["mean_kl"]
        + 0.25 * row["p95_kl"]
        + row["mean_nis_per_dim"]
        + 0.25 * row["mean_coupling_kl"]
    )
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


def render_plots(out_dir: Path, rows, pca_curve, latent_dim: int, console: Console):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rewards = [r["max_reward"] for r in rows]
    surprise = [r["surprise_score"] for r in rows]
    terminal = [r["terminal_dist"] for r in rows]
    episodes = [r["episode"] for r in rows]
    models = sorted(set(r["model"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name in models:
        idx = [i for i, r in enumerate(rows) if r["model"] == model_name]
        ax.scatter(
            [rewards[i] for i in idx],
            [surprise[i] for i in idx],
            c=[terminal[i] for i in idx],
            cmap="viridis",
            s=45,
            label=model_name,
        )
    ax.set_xlabel("max reward")
    ax.set_ylabel("JOPA surprise score")
    ax.set_title("PushT holdout surprise vs reward")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("terminal belief distance")
    ax.legend()
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
    for model_name in models:
        model_rows = [r for r in rows if r["model"] == model_name]
        ax.plot([r["episode"] for r in model_rows], [r["surprise_score"] for r in model_rows], label=f"{model_name} surprise")
    ax.plot(episodes[: len(rows) // max(len(models), 1)], terminal[: len(rows) // max(len(models), 1)], label="terminal distance", linestyle="--")
    ax.set_xlabel("episode")
    ax.set_title("Holdout episode diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "episode_diagnostics.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(1, len(pca_curve) + 1)
    ax.plot(xs, pca_curve, marker="o")
    ax.axhline(0.95, color="tab:orange", linestyle="--", linewidth=1, label="95%")
    ax.axhline(0.99, color="tab:green", linestyle="--", linewidth=1, label="99%")
    ax.axvline(latent_dim, color="tab:red", linestyle=":", linewidth=1.5, label=f"chosen dim={latent_dim}")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("PushT world-state PCA elbow")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_explained_variance.png", dpi=160)
    plt.close(fig)
    console.log(f"wrote plots to {out_dir}")


def summarize_rows(rows, model_name):
    subset = [r for r in rows if r["model"] == model_name]
    return {
        "n": len(subset),
        "mean_surprise": float(np.mean([r["surprise_score"] for r in subset])),
        "mean_r2": float(np.mean([r["r2"] for r in subset])),
        "mean_kl": float(np.mean([r["mean_kl"] for r in subset])),
        "mean_coupling_kl": float(np.mean([r["mean_coupling_kl"] for r in subset])),
        "mean_alert_rate": float(np.mean([r["alert_rate"] for r in subset])),
        "corr_surprise_reward": _corr([r["surprise_score"] for r in subset], [r["max_reward"] for r in subset]),
        "corr_terminal_reward": _corr([r["terminal_dist"] for r in subset], [r["max_reward"] for r in subset]),
    }


def print_comparison(console: Console, summary):
    table = Table(title="Hypothesis comparison")
    for col in ["model", "mean surprise", "mean R2", "mean KL", "coupling KL", "alert rate", "corr surprise/reward"]:
        table.add_column(col)
    for name, row in summary.items():
        table.add_row(
            name,
            f"{row['mean_surprise']:.3f}",
            f"{row['mean_r2']:.3f}",
            f"{row['mean_kl']:.3f}",
            f"{row['mean_coupling_kl']:.3f}",
            f"{row['mean_alert_rate']:.1%}",
            f"{row['corr_surprise_reward']:.3f}",
        )
    console.print(table)


def print_table(console: Console, title: str, rows, sort_key: str, top_k: int):
    table = Table(title=title)
    for col in ["rank", "model", "episode", "reward", "success", "surprise", "r2", "nis/d", "KL", "coupling", "alerts", "terminal"]:
        table.add_column(col)
    for rank, row in enumerate(sorted(rows, key=lambda r: r[sort_key], reverse=True)[:top_k], start=1):
        table.add_row(
            str(rank),
            row["model"],
            str(row["episode"]),
            f"{row['max_reward']:.3f}",
            str(row["success"]),
            f"{row['surprise_score']:.3f}",
            f"{row['r2']:.3f}",
            f"{row['mean_nis_per_dim']:.3f}",
            f"{row['mean_kl']:.3f}",
            f"{row['mean_coupling_kl']:.3f}",
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
    explained, pca_curve = transform_episodes(train_eps, test_eps, args.latent_dim)
    action_dim = train_eps[0].action.shape[-1]
    terminal = terminal_stats(train_eps)

    one_model, one_block = learn_block_model(
        "z",
        train_eps,
        lambda ep: ep.z,
        args.latent_dim,
        action_dim,
        args.precision,
        args.vmp_iters,
        console,
    )
    one_train_rows, one_kl_threshold = episode_rows(
        one_model,
        one_block,
        train_eps,
        terminal,
        console,
        "score one-block train episodes",
        "one_block_visual_world",
    )
    one_test_rows, _ = episode_rows(
        one_model,
        one_block,
        test_eps,
        terminal,
        console,
        "score one-block holdout episodes",
        "one_block_visual_world",
        alert_threshold=one_kl_threshold,
    )

    two_model, _, two_z_block, coupling_diagnostics = learn_two_block_model(
        train_eps,
        train_eps[0].p.shape[-1],
        args.latent_dim,
        action_dim,
        args.precision,
        args.vmp_iters,
        console,
    )
    two_train_rows, two_kl_threshold = episode_rows(
        two_model,
        two_z_block,
        train_eps,
        terminal,
        console,
        "score two-block train episodes",
        "two_block_proprio_visual",
    )
    two_test_rows, _ = episode_rows(
        two_model,
        two_z_block,
        test_eps,
        terminal,
        console,
        "score two-block holdout episodes",
        "two_block_proprio_visual",
        alert_threshold=two_kl_threshold,
    )

    train_rows = one_train_rows + two_train_rows
    test_rows = one_test_rows + two_test_rows
    comparison = {
        "one_block_visual_world": summarize_rows(test_rows, "one_block_visual_world"),
        "two_block_proprio_visual": summarize_rows(test_rows, "two_block_proprio_visual"),
    }

    summary = {
        "hypothesis": (
            "A coupled proprio + visual-world JOPA should improve holdout world-state prediction, "
            "belief consistency, or reward/surprise separation compared with a visual-world block alone."
        ),
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
        "pca_cumulative_explained_variance": pca_curve,
        "pca_components_for_95": int(np.searchsorted(pca_curve, 0.95) + 1),
        "pca_components_for_99": int(np.searchsorted(pca_curve, 0.99) + 1),
        "one_block_train_kl_p95": one_kl_threshold,
        "two_block_train_kl_p95": two_kl_threshold,
        "coupling": coupling_diagnostics,
        "success_reward_threshold": args.success_reward_threshold,
        "test_successes": int(sum(1 for r in one_test_rows if r["success"])),
        "comparison": comparison,
    }

    console.print(
        Panel(
            "\n".join(
                [
                    f"train/test: {summary['train']}/{summary['test']}",
                    f"latent_dim: {args.latent_dim}",
                    f"PCA explained variance: {explained:.3f}",
                    f"PCA components for 95%/99%: {summary['pca_components_for_95']}/{summary['pca_components_for_99']}",
                    f"one-block mean surprise/R2: {comparison['one_block_visual_world']['mean_surprise']:.3f}/{comparison['one_block_visual_world']['mean_r2']:.3f}",
                    f"two-block mean surprise/R2: {comparison['two_block_proprio_visual']['mean_surprise']:.3f}/{comparison['two_block_proprio_visual']['mean_r2']:.3f}",
                    f"two-block coupling KL: {comparison['two_block_proprio_visual']['mean_coupling_kl']:.3f}",
                ]
            ),
            title="run summary",
        )
    )
    print_comparison(console, comparison)
    print_table(console, "Top surprise holdout episodes", test_rows, "surprise_score", args.top_k)
    print_table(console, "Largest terminal belief misses", test_rows, "terminal_dist", args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "train_metrics.csv", train_rows)
    _write_csv(args.output_dir / "test_metrics.csv", test_rows)
    _write_csv(args.output_dir / "one_block_train_metrics.csv", one_train_rows)
    _write_csv(args.output_dir / "one_block_test_metrics.csv", one_test_rows)
    _write_csv(args.output_dir / "two_block_train_metrics.csv", two_train_rows)
    _write_csv(args.output_dir / "two_block_test_metrics.csv", two_test_rows)
    _write_json(args.output_dir / "summary.json", summary)
    render_plots(args.output_dir, test_rows, pca_curve, args.latent_dim, console)
    console.print(f"[green]wrote metrics to {args.output_dir}[/green]")


if __name__ == "__main__":
    main()
