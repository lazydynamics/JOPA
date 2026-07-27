"""Render one clip per held-out pose and rank them by how well each settles.

    MUJOCO_GL=egl python figures/reacher_clips.py --poses 10 --out previews
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import BRICK, INK, INK_LIGHT, PAPER, RULE

from examples.reacher.runtime import ReacherLoop


def _chrome(image, step, error, start, size):
    from PIL import Image, ImageDraw

    pad_top, pad_bottom = 30, 44
    canvas = Image.new("RGB", (size, size + pad_top + pad_bottom), PAPER)
    canvas.paste(Image.fromarray(image).convert("RGB"), (0, pad_top))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), "goal given as an image", fill=INK_LIGHT)
    draw.text((10, size + pad_top + 8), f"step {step:3d}", fill=INK_LIGHT)
    draw.text((78, size + pad_top + 8), f"{error:5.2f} cm to goal", fill=INK)
    track_x, track_w = 10, size - 20
    y = size + pad_top + 28
    draw.rectangle((track_x, y, track_x + track_w, y + 5), fill=RULE)
    filled = int(track_w * min(error / max(start, 1e-6), 1.0))
    draw.rectangle((track_x, y, track_x + max(filled, 2), y + 5), fill=BRICK)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=10)
    parser.add_argument("--seed", type=int, default=909090)
    parser.add_argument("--out", default="previews")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args(argv)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    loop = ReacherLoop(record=True)
    rows = []
    for index in range(args.poses):
        run = loop.episode(args.seed + index, steps=args.steps)
        error = run["error"]
        tail = error[-20:]
        row = {
            "pose": index,
            "start_cm": run["start_cm"],
            "min_cm": float(error.min()),
            "terminal_mean_cm": float(tail.mean()),
            "terminal_max_cm": float(tail.max()),
            "settled_frac60": float((error[-60:] < 3.0).mean()),
            "p2p60_cm": float(error[-60:].max() - error[-60:].min()),
        }
        frames = [
            _chrome(image, step * args.stride, error[step * args.stride],
                    run["start_cm"], run["frame_size"])
            for step, image in enumerate(run["frames"][::args.stride])
        ]
        path = out / f"reacher_{index:02d}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=60, loop=0, optimize=True)
        row["clip"] = path.name
        rows.append(row)
        print(f"{path.name}  start {row['start_cm']:5.1f}  min {row['min_cm']:5.2f}"
              f"  terminal max {row['terminal_max_cm']:5.2f}"
              f"  settled {row['settled_frac60']:.2f}", flush=True)

    rank = sorted(rows, key=lambda r: (-r["settled_frac60"],
                                       r["terminal_max_cm"]))
    (out / "ranking.json").write_text(json.dumps(rank, indent=2) + "\n")
    print("\nbest clips (settled fraction, then worst terminal error):")
    for row in rank[:5]:
        print(f"  {row['clip']}  settled {row['settled_frac60']:.2f}  "
              f"terminal max {row['terminal_max_cm']:.2f} cm  "
              f"start {row['start_cm']:.0f} cm")
    loop.close()


if __name__ == "__main__":
    main()
