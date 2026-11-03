"""Side-by-side video of two solvers on the same episodes.

The harness writes ``env_<i>.mp4`` per episode (agent | dataset | goal) and
logs ``episode_successes``. This tiles episodes as goal | A | B, each
labelled with its outcome: the first ``--episodes`` in index order, or with
``--disagree`` the episodes where exactly one solver reaches the goal, in
proportion to how often each does (first ones in index order), with the
full counts in the title.

    python montage.py --a runs/videos/lewm_gain_adam --a-log runs/videos/logs/lewm_gain_adam_s42.log \\
                      --b runs/videos/jopa_fault --b-log runs/videos/logs/jopa_fault_s42.log \\
                      --disagree --episodes 8 --out ../../assets/reacher_fault.gif
"""
import argparse
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL = 112           # px per panel
PAD = 6
HEAD = 30
TITLE = 34


def successes(log):
    body = re.search(r"episode_successes.: array\(\[(.*?)\]", Path(log).read_text(), re.S)
    return [w == "True" for w in re.findall(r"True|False", body.group(1))]


def panels(video):
    """Agent and goal views of every frame, cropped from the harness's triptych."""
    v = iio.imread(video)                                  # (T, 288, 736, 3)
    return v[:, 16:240, 16:240], v[:, 16:240, 492:716]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in "ab":
        p.add_argument(f"--{k}", type=Path, required=True)
        p.add_argument(f"--{k}-log", type=Path, required=True)
    p.add_argument("--a-name", default="LeWM + gain (Adam)")
    p.add_argument("--b-name", default="JOPA")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--columns", type=int, default=2)
    p.add_argument("--disagree", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    ok = {"a": successes(args.a_log), "b": successes(args.b_log)}
    n = len(ok["a"])
    ids, title = list(range(args.episodes)), f"first {args.episodes} of {n} episodes"
    if args.disagree:
        b_only = [i for i in range(n) if ok["b"][i] and not ok["a"][i]]
        a_only = [i for i in range(n) if ok["a"][i] and not ok["b"][i]]
        k = round(args.episodes * len(b_only) / (len(a_only) + len(b_only)))
        ids = b_only[:k] + a_only[:args.episodes - k]
        title = (f"{len(a_only) + len(b_only)} of {n} episodes end differently: {args.b_name} "
                 f"alone reaches {len(b_only)}, {args.a_name} alone {len(a_only)} "
                 f"(shown {k} : {args.episodes - k})")
    font = ImageFont.load_default(size=12)
    bold = ImageFont.load_default(size=14)
    rows = -(-args.episodes // args.columns)
    block_w = 3 * CELL + 2 * PAD
    W = args.columns * block_w + (args.columns - 1) * 4 * PAD
    H = TITLE + HEAD + rows * (CELL + HEAD)
    clips = []
    for i in ids:
        a, goal = panels(args.a / f"env_{i}.mp4")
        b, _ = panels(args.b / f"env_{i}.mp4")
        clips.append((goal, a, b))
    T = min(min(len(c) for c in clip) for clip in clips)

    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, H), "#fcfcfb")
        draw = ImageDraw.Draw(img)
        draw.text((W // 2, TITLE // 2), title, fill="#0b0b0b", font=bold, anchor="mm")
        for c in range(args.columns):
            x0 = c * (block_w + 4 * PAD)
            for j, name in enumerate(("goal", args.a_name, args.b_name)):
                draw.text((x0 + j * (CELL + PAD) + CELL // 2, TITLE + HEAD // 2), name,
                          fill="#0b0b0b",
                          font=font, anchor="mm")
        for slot, (ep, (goal, a, b)) in enumerate(zip(ids, clips, strict=True)):
            r, c = divmod(slot, args.columns)
            x0, y0 = c * (block_w + 4 * PAD), TITLE + HEAD + r * (CELL + HEAD)
            for j, (clip, key) in enumerate(((goal, None), (a, "a"), (b, "b"))):
                tile = Image.fromarray(clip[t]).resize((CELL, CELL))
                img.paste(tile, (x0 + j * (CELL + PAD), y0))
                if key and t > 0.8 * T:
                    good = ok[key][ep]
                    draw.text((x0 + j * (CELL + PAD) + CELL // 2, y0 + CELL + 12),
                              "reached" if good else "missed", anchor="mm", font=font,
                              fill="#008300" if good else "#e34948")
            draw.text((x0 + CELL // 2, y0 + CELL + 12), f"episode {ep}", anchor="mm",
                      font=font, fill="#52514e")
        frames.append(np.asarray(img))
    frames += [frames[-1]] * 12                              # hold the outcome
    iio.imwrite(args.out, np.stack(frames), duration=120, loop=0)
    print("wrote", args.out, f"{len(frames)} frames")


if __name__ == "__main__":
    main()
