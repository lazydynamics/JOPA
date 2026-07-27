"""Rendering for the JOPA figures. Data comes from
`message_passing_animation.build`; this module only draws it.

Art direction: paper ground, flat matte fills, desaturated technical-manual
palette, one saturated accent reserved for the message in motion. No glow,
no gradient, no shadow. Hierarchy comes from scale, weight and space.
"""
from __future__ import annotations

import numpy as np

PAPER = "#f2efe6"
INK = "#2f2c26"
INK_LIGHT = "#7a746a"
RULE = "#d5cfc0"
SLATE = "#78848c"
OCHRE = "#b09a6b"
OLIVE = "#87886c"
BRICK = "#9d6b58"
ACCENT = "#d9401c"

N_STATES = 5
HOP = 5
HOLD = 6


def _ease_out(t):
    return 1.0 - (1.0 - np.clip(t, 0.0, 1.0)) ** 3


def _layout():
    xs = np.linspace(0.10, 0.90, N_STATES)
    states = [(x, 0.52) for x in xs]
    factors = [((xs[i] + xs[i + 1]) / 2, 0.52) for i in range(N_STATES - 1)]
    controls = [(f[0], 0.78) for f in factors]
    likelihoods = [(x, 0.30) for x in xs]
    observations = [(x, 0.09) for x in xs]
    return {
        "states": states, "factors": factors, "controls": controls,
        "likelihoods": likelihoods, "observations": observations,
        "parameters": (0.50, 0.95),
    }


def _schedule(layout):
    """One message per step: (start, end, caption).

    Order follows the inference: evidence enters from the observations, is
    propagated along the chain in both directions, and the transitions then
    report what they learned to the shared parameters.
    """
    hops = []
    for index in range(N_STATES - 1):
        hops.append((layout["states"][index], layout["factors"][index]))
        hops.append((layout["factors"][index], layout["states"][index + 1]))

    caption = "observation messages — the sensor states its own precision"
    steps = [(layout["observations"][i], layout["states"][i], caption)
             for i in range(N_STATES)]
    caption = "forward pass — each message carries the past one factor onward"
    steps += [(a, b, caption) for a, b in hops]
    caption = "backward pass — each message brings the future back"
    steps += [(b, a, caption) for a, b in reversed(hops)]
    caption = "parameter messages — every transition updates q(A, B, W)"
    steps += [(f, layout["parameters"], caption) for f in layout["factors"]]
    return steps


def _draw_graph(ax, layout, active=None, caption="", annotate=True):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for (sx, sy), (lx, ly), (ox, oy) in zip(
            layout["states"], layout["likelihoods"], layout["observations"]):
        ax.plot([sx, lx], [sy, ly], color=RULE, lw=0.9, zorder=1)
        ax.plot([lx, ox], [ly, oy], color=RULE, lw=0.9, zorder=1)
    px, py = layout["parameters"]
    for index, (fx, fy) in enumerate(layout["factors"]):
        left, right = layout["states"][index], layout["states"][index + 1]
        ax.plot([left[0], fx], [left[1], fy], color=RULE, lw=0.9, zorder=1)
        ax.plot([fx, right[0]], [fy, right[1]], color=RULE, lw=0.9, zorder=1)
        cx, cy = layout["controls"][index]
        ax.plot([cx, fx], [cy, fy], color=RULE, lw=0.9, zorder=1)
        ax.plot([px, fx], [py, fy], color=RULE, lw=0.6, zorder=1)

    for fx, fy in layout["factors"]:
        ax.scatter([fx], [fy], s=86, marker="s", facecolor=PAPER,
                   edgecolor=INK, lw=1.0, zorder=4)
    for lx, ly in layout["likelihoods"]:
        ax.scatter([lx], [ly], s=74, marker="s", facecolor=PAPER,
                   edgecolor=INK_LIGHT, lw=1.0, zorder=4)
    for index, (sx, sy) in enumerate(layout["states"]):
        ax.scatter([sx], [sy], s=560, facecolor=PAPER, edgecolor=INK, lw=1.3,
                   zorder=4)
        ax.text(sx, sy, f"$x_{index}$", color=INK, ha="center", va="center",
                fontsize=11.5, zorder=6)
    for index, (ox, oy) in enumerate(layout["observations"]):
        ax.scatter([ox], [oy], s=380, marker="s", facecolor=SLATE,
                   edgecolor=SLATE, lw=1.0, zorder=4)
        ax.text(ox, oy, f"$y_{index}$", color=PAPER, ha="center", va="center",
                fontsize=10, zorder=6)
    for index, (cx, cy) in enumerate(layout["controls"]):
        ax.scatter([cx], [cy], s=250, marker="D", facecolor=PAPER,
                   edgecolor=INK_LIGHT, lw=1.0, zorder=4)
        ax.text(cx, cy, f"$u_{index}$", color=INK_LIGHT, ha="center",
                va="center", fontsize=9, zorder=6)
    ax.scatter([px], [py], s=700, facecolor=OCHRE, edgecolor=OCHRE, lw=1.0,
               zorder=4)
    ax.text(px, py, "θ", color=PAPER, ha="center", va="center", fontsize=12,
            zorder=6)
    if annotate:
        ax.text(px + 0.035, py, "q(A, B, W)", color=INK, ha="left",
                va="center", fontsize=10.5, zorder=6)

    if active is not None:
        start, end, progress = active
        eased = _ease_out(progress)
        x = start[0] + (end[0] - start[0]) * eased
        y = start[1] + (end[1] - start[1]) * eased
        ax.scatter([x], [y], s=110, facecolor=ACCENT, edgecolor=ACCENT,
                   lw=0, zorder=8)
    if caption:
        ax.text(0.5, -0.04, caption, color=INK_LIGHT, ha="center", va="top",
                fontsize=10.5, transform=ax.transAxes)


def _bare_axes(ax):
    ax.clear()
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_LIGHT)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK_LIGHT, labelsize=8.5, length=3, width=0.9,
                   direction="out")
    ax.grid(False)


def _draw_belief(ax, estimation, fraction=1.0):
    _bare_axes(ax)
    steps = np.arange(len(estimation.mean))
    mean = estimation.mean[:, 0]
    std = estimation.std[:, 0]
    truth = estimation.truth[:len(steps), 0]
    observed = estimation.n_observed
    shown = max(int(np.clip(fraction, 0.0, 1.0) * len(steps)), 1)

    ax.fill_between(steps[:shown], (mean - 2 * std)[:shown],
                    (mean + 2 * std)[:shown], color=SLATE, alpha=0.22,
                    linewidth=0)
    ax.plot(steps[:shown], truth[:shown], color=INK_LIGHT, lw=1.0)
    ax.plot(steps[:shown], mean[:shown], color=INK, lw=1.6)
    marks = min(shown, observed)
    ax.scatter(steps[:marks], estimation.observations[:marks, 0], s=7,
               color=SLATE, linewidths=0)
    if shown > observed:
        ax.axvline(observed - 0.5, color=INK_LIGHT, lw=0.8)

    low = min(truth.min(), (mean - 2 * std).min())
    high = max(truth.max(), (mean + 2 * std).max())
    span = high - low
    ax.set_xlim(0, len(steps) - 1)
    ax.set_ylim(low - 0.06 * span, high + 0.06 * span)
    ax.set_xlabel("step", color=INK_LIGHT, fontsize=9)
    ax.set_ylabel("position", color=INK_LIGHT, fontsize=9)

    near, far = len(steps) // 6, len(steps) // 3
    if shown > near:
        ax.text(near, mean[near] + 0.30 * span, "posterior mean", color=INK,
                fontsize=8.5, ha="center", va="bottom")
    if shown > far:
        ax.text(far, truth[far] + 0.14 * span, "true state", color=INK_LIGHT,
                fontsize=8.5, ha="center", va="bottom")
    tail = len(steps) - 4
    if shown >= len(steps):
        ax.text(tail, mean[tail] + 2 * std[tail] + 0.03 * span, "± 2σ",
                color=SLATE, fontsize=8.5, ha="right", va="bottom")
    if shown > observed:
        ax.text(observed - 1.5, high + 0.02 * span, "observations stop",
                color=INK_LIGHT, fontsize=8.5, ha="right", va="top")


def _draw_control(ax, runs, fraction=1.0):
    _bare_axes(ax)
    length = len(runs[0].control)
    steps = np.arange(length)
    labels = ("calibrated sensor", "sensor claims certainty")
    colors = (INK, BRICK)
    shown = max(int(np.clip(fraction, 0.0, 1.0) * length), 1)
    curves = []
    for run, color, label in zip(runs, colors, labels):
        effort = np.cumsum(np.abs(run.control))
        ax.plot(steps[:shown], effort[:shown], color=color, lw=1.7)
        curves.append((effort, color, label))
    ceiling = max(c[0][-1] for c in curves)
    ax.set_xlim(0, length - 1)
    ax.set_ylim(0, ceiling * 1.08)
    ax.set_xlabel("closed-loop step", color=INK_LIGHT, fontsize=9)
    ax.set_ylabel("cumulative action", color=INK_LIGHT, fontsize=9)
    at = int(length * 0.62)
    if shown > at:
        for effort, color, label in curves:
            above = color is BRICK
            ax.text(at, effort[at] + (0.05 if above else -0.05) * ceiling,
                    label, color=color, fontsize=8.5, ha="center",
                    va="bottom" if above else "top")
    if shown >= length:
        honest, rival = runs
        ratio = np.abs(rival.control).sum() / np.abs(honest.control).sum()
        ax.text(0.02, 0.96, f"{ratio:.1f}× the action for the same accuracy",
                transform=ax.transAxes, color=INK_LIGHT, fontsize=9, va="top")


def render_model(path, dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = _layout()
    figure = plt.figure(figsize=(9.4, 5.2), facecolor=PAPER)
    ax = figure.add_axes([0.03, 0.24, 0.94, 0.68])
    _draw_graph(ax, layout, annotate=True)

    figure.text(0.5, 0.965, "One latent chain. Parameters shared across time.",
                color=INK, fontsize=13, ha="center", va="top")

    rows = [
        (0.155, "transition",
         r"$x_t \sim \mathcal{N}(A x_{t-1} + B u_{t-1},\; W^{-1})$"),
        (0.095, "likelihood",
         r"$q_\phi(x_t \mid y_t) = \mathcal{N}(\mu_\phi(y_t),\; \Sigma_\phi(y_t))$"),
        (0.035, "parameters",
         r"$q(\mathbf{a}),\, q(\mathbf{b})$ Gaussian;  $q(W)$ Wishart"),
    ]
    for y, label, equation in rows:
        figure.text(0.085, y, label, color=INK_LIGHT, fontsize=9.5,
                    ha="left", va="center")
        figure.text(0.30, y, equation, color=INK, fontsize=10.5, ha="left",
                    va="center")
    figure.text(0.93, 0.155,
                "controls given while learning,\ninferred while planning",
                color=INK_LIGHT, fontsize=9, ha="right", va="center",
                linespacing=1.6)
    figure.savefig(path, dpi=dpi, facecolor=PAPER)
    plt.close(figure)
    return path


def render(data, path, fps=20, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    layout = _layout()
    schedule = _schedule(layout)
    graph_frames = len(schedule) * HOP
    panel_frames = 26
    total = graph_frames + 2 * panel_frames + HOLD

    figure = plt.figure(figsize=(11.0, 7.3), facecolor=PAPER)
    ax_graph = figure.add_axes([0.02, 0.45, 0.96, 0.47])
    ax_belief = figure.add_axes([0.075, 0.09, 0.36, 0.27])
    ax_control = figure.add_axes([0.585, 0.09, 0.34, 0.27])

    figure.text(0.025, 0.972, "JOPA", color=INK, fontsize=16,
                fontweight="bold", va="top")
    figure.text(0.025, 0.930,
                "learning, filtering, prediction and planning are the same "
                "messages",
                color=INK_LIGHT, fontsize=10.5, va="top")

    def update(frame):
        if frame < graph_frames:
            index = frame // HOP
            start, end, caption = schedule[index]
            progress = (frame % HOP) / (HOP - 1)
            _draw_graph(ax_graph, layout, active=(start, end, progress),
                        caption=caption)
            _draw_belief(ax_belief, data["estimation"], 0.0)
            _draw_control(ax_control, data["runs"], 0.0)
            return []
        after = frame - graph_frames
        if after < panel_frames:
            belief = _ease_out(after / (panel_frames - 1))
            control = 0.0
            caption = "the same sweep, read out: smoothing, then prediction"
        else:
            belief = 1.0
            control = _ease_out(
                min((after - panel_frames) / (panel_frames - 1), 1.0))
            caption = "and what the sensor's own precision costs in control"
        _draw_graph(ax_graph, layout, active=None, caption=caption)
        _draw_belief(ax_belief, data["estimation"], belief)
        _draw_control(ax_control, data["runs"], control)
        return []

    animation = FuncAnimation(figure, update, frames=total, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi,
                   savefig_kwargs={"facecolor": PAPER})
    plt.close(figure)
    return path
