"""Encode training and held-out Reacher episodes with LeWM's frozen encoder.

One frame per action block (5 env steps) and the block's 10-d raw action,
for 1000 training and 200 held-out episodes of the released dataset
(episode-level split, seed 3072 — the protocol's gate split).

    python encode.py --out runs/episodes.npz
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch

BLOCK = 5            # env steps per action block (upstream action_block)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def split_episodes(n_episodes, seed, n_train, n_test):
    """Episode-level 90/10 split; the first n of each side are used."""
    order = np.random.default_rng(seed).permutation(n_episodes)
    cut = int(0.9 * n_episodes)
    return np.sort(order[:cut][:n_train]), np.sort(order[cut:][:n_test])


def read_blocks(h5, episodes):
    """Frames at block boundaries and the block actions, per episode."""
    offsets, lengths = h5["ep_offset"][:], h5["ep_len"][:]
    frames, actions = [], []
    for e in episodes:
        start, length = int(offsets[e]), int(lengths[e])
        n_blocks = (length - 1) // BLOCK
        rows = start + BLOCK * np.arange(n_blocks + 1)
        frames.append(h5["pixels"][rows[0]:rows[-1] + 1][::BLOCK])
        steps = h5["action"][start:start + BLOCK * n_blocks]
        actions.append(steps.reshape(n_blocks, BLOCK * steps.shape[1]))
    return frames, actions


@torch.no_grad()
def encode(model, frames, batch=512):
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i + batch]).cuda().permute(0, 3, 1, 2).float() / 255.0
        x = (x - IMAGENET_MEAN.cuda()) / IMAGENET_STD.cuda()
        out.append(model.encode({"pixels": x[:, None]})["emb"][:, 0].float().cpu())
    return torch.cat(out).numpy()


def load(path):
    """``(train, test)`` lists of ``(embeddings, actions)`` per episode."""
    z = np.load(path)
    def get(p):
        return [(z[f"{p}_emb_{i}"], z[f"{p}_act_{i}"])
                         for i in range(sum(k.startswith(f"{p}_emb_") for k in z.files))]
    return get("tr"), get("te")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-episodes", type=int, default=1000)
    p.add_argument("--test-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    t0 = time.time()
    model = swm.wm.utils.load_pretrained("quentinll/lewm-reacher").cuda().eval()
    path = Path(swm.data.utils.get_cache_dir(sub_folder="datasets"), "dmc/reacher_random.h5")
    with h5py.File(path, "r") as h5:
        train_ep, test_ep = split_episodes(len(h5["ep_len"]), args.seed,
                                           args.train_episodes, args.test_episodes)
        tr_frames, tr_actions = read_blocks(h5, train_ep)
        te_frames, te_actions = read_blocks(h5, test_ep)
    tr_emb = [encode(model, f) for f in tr_frames]
    te_emb = [encode(model, f) for f in te_frames]
    print(f"encoded {len(train_ep)}+{len(test_ep)} episodes in {time.time() - t0:.0f}s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **{f"tr_emb_{i}": e for i, e in enumerate(tr_emb)},
        **{f"tr_act_{i}": a for i, a in enumerate(tr_actions)},
        **{f"te_emb_{i}": e for i, e in enumerate(te_emb)},
        **{f"te_act_{i}": a for i, a in enumerate(te_actions)})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
