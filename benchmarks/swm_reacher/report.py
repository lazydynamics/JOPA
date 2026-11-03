"""Result table and figure from `reproduce.sh`'s logs.

    python report.py --runs ~/jopa-bench/runs        # logs -> results.json
    python report.py --figure ../../assets           # results.json -> figures

Success is the harness's ``success_rate`` per seed; solve time is the median
over every replan of every seed (wall clock for the batch of unfinished
episodes, first-call compilation included in the median's tail only).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SEEDS = (42, 1, 2, 3, 4)
ROWS = {   # name: (setting, label)
    "jopa": ("nominal", "JOPA"),
    "lewm": ("nominal", "LeWM + CEM"),
    "jopa_fault": ("fault", "JOPA"),
    "lewm_gain_adam": ("fault", "LeWM + gain, Adam"),
    "lewm_finetune": ("fault", "LeWM, online fine-tuning"),
    "lewm_gain_laplace": ("fault", "LeWM + gain, Laplace"),
    "jopa_frozen": ("fault", "JOPA, no adaptation"),
    "lewm_frozen": ("fault", "LeWM, no adaptation"),
}


def parse(runs: Path):
    out = {}
    for name, (setting, label) in ROWS.items():
        success, solve = [], []
        for s in SEEDS:
            log = (runs / f"{name}_s{s}.log").read_text()
            success.append(float(re.findall(r"success_rate.: ([0-9.]+)", log)[0]))
            solve += [float(t) for t in re.findall(r"solve time: ([0-9.]+)", log)]
        out[name] = {"setting": setting, "label": label, "success": success,
                     "median_solve_s": float(np.median(solve))}
        trace = re.findall(r"actuator posterior mean (\[\[.*?\]\]) sd (\[\[.*?\]\])",
                           (runs / f"{name}_s{SEEDS[0]}.log").read_text())
        if trace:   # q(A) after each replan, first seed, row-major
            def num(t):
                return [float(v) for v in re.findall(r"-?[0-9.]+(?:e-?[0-9]+)?", t)]
            out[name]["actuator"] = [num(m) for m, _ in trace]
            out[name]["actuator_sd"] = [num(sd) for _, sd in trace]
    return out


def table(results):
    lines = ["| | " + " | ".join(map(str, SEEDS)) + " | mean ± std | median solve |",
             "|---|" + "---|" * (len(SEEDS) + 2)]
    for r in results.values():
        s = np.asarray(r["success"])
        lines.append(f"| {r['label']} | " + " | ".join(f"{x:.0f}" for x in s)
                     + f" | {s.mean():.1f} ± {s.std(ddof=1):.1f} % | {r['median_solve_s']:.2f} s |")
    return "\n".join(lines)


THEMES = {   # reference palette (dataviz skill): slot 1 blue = JOPA, slot 2 orange = LeWM
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  jopa="#2a78d6", lewm="#eb6834"),
    "dark": dict(surface="#0d1117", text="#f0efec", muted="#c3c2b7", grid="#30302e",
                 jopa="#3987e5", lewm="#d95926"),
}


def figure(results, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fault = [k for k, r in results.items() if r["setting"] == "fault"][::-1]
    for theme, c in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "text.color": c["text"], "axes.labelcolor": c["muted"],
                             "xtick.color": c["muted"], "ytick.color": c["text"]})
        fig, (a, b, t) = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=c["surface"],
                                      gridspec_kw={"width_ratios": [2.6, 1.1, 0.9]})
        for ax in (a, b, t):
            ax.set_facecolor(c["surface"])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(c["grid"])
            ax.tick_params(length=0)
            ax.grid(axis="x", color=c["grid"], linewidth=0.8)
            ax.set_axisbelow(True)

        def bars(ax, keys, value, fmt, xmax, seeds=False, c=c):
            ax.set_xlim(0, xmax * 1.16)
            ax.set_ylim(-0.6, len(keys) - 0.4) if len(keys) > 2 else ax.set_ylim(-2.5, 3.5)
            for i, k in enumerate(keys):
                r, jopa = results[k], k.startswith("jopa")
                col = c["jopa"] if jopa else c["lewm"]
                v = value(r)
                ax.barh(i, v, height=0.6, color=col)
                end = v
                if seeds:
                    ax.scatter(r["success"], np.full(len(SEEDS), i), s=16, color=c["surface"],
                               edgecolors=c["text"], linewidths=0.9, zorder=3)
                    end = max(end, max(r["success"]))
                ax.text(end + xmax * 0.03, i, fmt(v), va="center", color=c["text"],
                        fontsize=10, fontweight="bold" if jopa else None)
            ax.set_yticks(range(len(keys)), [results[k]["label"] for k in keys])

        title = dict(loc="left", color=c["text"], fontsize=11, fontweight="bold", pad=10)
        bars(a, fault, _mean, lambda v: f"{v:.1f}%", 100, seeds=True)
        a.set_xticks(range(0, 101, 20))
        a.set_title("Shoulder actuator wired backwards", **title)
        a.set_xlabel("success (%) — bar: mean of 5 seeds, dots: seeds")

        nominal = ["lewm", "jopa"]
        bars(b, nominal, _mean, lambda v: f"{v:.1f}%", 100)
        b.set_xticks(range(0, 101, 50))
        b.set_title("Healthy arm", **title)
        b.set_xlabel("success (%), mean of 5 seeds")

        bars(t, nominal, lambda r: r["median_solve_s"], lambda v: f"{v:.2f} s", 2.0)
        t.set_xticks([0, 1, 2])
        t.set_yticks(range(len(nominal)), [""] * len(nominal))
        t.set_title("Planning time", **title)
        t.set_xlabel("median s per replan")

        fig.tight_layout(w_pad=1.5)
        fig.savefig(out / f"lewm_reacher_{theme}.png", dpi=200, facecolor=c["surface"])
        plt.close(fig)


def _mean(r):
    return float(np.mean(r["success"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, help="parse logs into results.json")
    p.add_argument("--figure", type=Path, help="draw figures from results.json into this dir")
    args = p.parse_args(argv)
    path = HERE / "results.json"
    if args.runs:
        path.write_text(json.dumps(parse(args.runs), indent=1))
    results = json.loads(path.read_text())
    print(table(results))
    if args.figure:
        args.figure.mkdir(parents=True, exist_ok=True)
        figure(results, args.figure)


if __name__ == "__main__":
    main()
