"""Presentation layer for the boat demo — summary plot, demo animation, and
the plan/act/adapt loop animation. Pure matplotlib over the logs that
examples/boat.py records; no inference happens here.

Conventions: research-figure style — neutral labels, data marks only, light
surface. The craft is drawn as what it is: an omnidirectional thruster
platform (no heading — thrust sets velocity in any direction). Palette
validated for CVD separation; blue is the adaptive model in every figure,
gray the frozen one; aqua/yellow carry the current components (always
directly labeled); violet is surprise.
"""
import numpy as np

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
WATER = "#f0f4f8"
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"

_OCT = np.stack([np.cos(np.pi / 8 + np.arange(8) * np.pi / 4),
                 np.sin(np.pi / 8 + np.arange(8) * np.pi / 4)], axis=1)


class _Craft:
    """Top-down omnidirectional thruster platform: octagonal body, four
    corner pods. Fixed orientation — the craft has no bow."""

    def __init__(self, ax, color, size, z):
        from matplotlib.patches import Circle, Polygon
        self.body_t = _OCT * size
        self.body = Polygon(self.body_t, closed=True, facecolor=color,
                            edgecolor=SURFACE, lw=0.6, zorder=z)
        ax.add_patch(self.body)
        self.off = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * size * 1.02
        self.pods = []
        for o in self.off:
            pod = Circle(tuple(o), size * 0.30, facecolor=SURFACE,
                         edgecolor=color, lw=0.9, zorder=z + 1)
            ax.add_patch(pod)
            self.pods.append(pod)
        self.track, = ax.plot([], [], "-", color=color, lw=1.2, alpha=0.4,
                              zorder=z - 1, solid_capstyle="round")

    def move(self, p, track):
        self.body.set_xy(self.body_t + p)
        for pod, o in zip(self.pods, self.off):
            pod.center = tuple(p + o)
        self.track.set_data(track[:, 0], track[:, 1])


class Stage:
    """Top-down view: banks, goal marker, current vector, two crafts with
    their recent tracks. Every mark is data; nothing is decorative."""

    def __init__(self, ax):
        from matplotlib.patches import Polygon, Rectangle
        self.ax = ax
        ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BASELINE)
        ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor=WATER, zorder=0))
        ax.add_patch(Rectangle((-0.75, -0.75), 1.5, 1.5, fill=False,
                               edgecolor=BASELINE, ls=(0, (4, 3)), lw=0.9))
        self.caption = ax.annotate("", (0.03, 0.97), xycoords="axes fraction",
                                   fontsize=9, color=INK2, va="top")

        # Goal: a target marker (ring + dot).
        self.goal_ring = ax.scatter([], [], s=170, facecolors="none",
                                    edgecolors=INK, linewidths=1.2, zorder=4)
        self.goal_dot = ax.scatter([], [], s=10, color=INK, zorder=4)

        # True current: one labeled vector, lower left.
        self.cur_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=3,
                                     arrowprops=dict(arrowstyle="-|>", color=INK2,
                                                     lw=1.4, mutation_scale=11))
        self.cur_label = ax.annotate("current", (-0.97, -0.885), fontsize=7.5,
                                     color=MUTED, va="center")

        self.frozen = _Craft(ax, MUTED, 0.038, 5)
        self.adaptive = _Craft(ax, BLUE, 0.046, 7)

        for y, color, lab in ((0.955, BLUE, "adaptive model"),
                              (0.885, MUTED, "frozen model")):
            ax.add_patch(Polygon(_OCT * 0.028 + [0.53, y * 2.04 - 1.02],
                                 closed=True, facecolor=color, lw=0, zorder=10))
            ax.annotate(lab, (0.78, y), xycoords="axes fraction", fontsize=8,
                        color=INK2, va="center", ha="left")

    def update(self, pos_i, info_i, log_on, log_off, acts):
        name, cur = acts[int(log_on["act"][info_i])]
        cur = np.asarray(cur, dtype=float)
        self.caption.set_text(f"t = {info_i} · {name}")

        base = np.array([-0.90, -0.885])
        self.cur_arrow.set_position(tuple(base))
        self.cur_arrow.xy = tuple(base + 0.45 * cur)
        self.cur_arrow.arrow_patch.set_alpha(1.0 if np.linalg.norm(cur) > 0 else 0.0)
        self.cur_label.set_text("current" if np.linalg.norm(cur) > 0
                                else "current: none")

        self.goal_ring.set_offsets([log_on["goal"][info_i]])
        self.goal_dot.set_offsets([log_on["goal"][info_i]])
        lo = max(0, pos_i - 30)
        self.adaptive.move(log_on["p"][pos_i], log_on["p"][lo:pos_i + 1])
        self.frozen.move(log_off["p"][pos_i], log_off["p"][lo:pos_i + 1])


def _style_axis(ax, keep=("left", "bottom")):
    ax.set_facecolor(SURFACE)
    for side, spine in ax.spines.items():
        spine.set_color(BASELINE)
        spine.set_visible(side in keep)
    ax.tick_params(colors=INK2, labelsize=7, length=3)
    ax.grid(True, color=GRID, lw=0.6)


def _panel(ax, label):
    ax.set_title(label, fontsize=8.5, color=INK2, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BASELINE)


def _belief_panel(ax):
    from matplotlib.patches import Ellipse
    _panel(ax, "current estimate q(c), ±2σ")
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), aspect="equal")
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    true_pt = ax.scatter([], [], marker="x", s=70, color=INK, lw=1.4, zorder=4)
    post_pt = ax.scatter([], [], s=22, color=AQUA, zorder=5)
    ell = Ellipse((0, 0), 0, 0, fill=True, alpha=0.15, color=AQUA)
    ax.add_patch(ell)
    ax.annotate("× true", (0.05, 0.05), xycoords="axes fraction",
                fontsize=7, color=MUTED)

    def update(cur, c_mu, c_cov):
        true_pt.set_offsets([np.asarray(cur)])
        post_pt.set_offsets([c_mu])
        evals, evecs = np.linalg.eigh(c_cov)
        ell.set_center(c_mu)
        ell.width, ell.height = 4 * np.sqrt(np.maximum(evals, 1e-12))
        ell.angle = float(np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1])))

    return update


def save_money_plot(path, log_on, log_off, err_on, err_off, acts, act_steps):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib for the plots:  uv pip install -e '.[viz]')")
        return False
    t = np.arange(len(err_on))
    bounds = [act_steps * i for i in range(1, len(acts))]
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.8), sharex=True,
                             gridspec_kw={"hspace": 0.38})
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    _style_axis(ax)
    ax.semilogy(t, log_on["nis"], color=VIOLET, lw=0.9)
    ax.axhline(4.0, color=BASELINE, ls="--", lw=0.9)
    ax.annotate("E[χ²] under the model", (0.995, 0.11), xycoords="axes fraction",
                ha="right", fontsize=7, color=MUTED)
    ax.set_ylabel("surprise (NIS)", fontsize=9, color=INK)

    ax = axes[1]
    _style_axis(ax)
    true_c = np.array([acts[a][1] for a in log_on["act"]])
    for j, (color, lbl) in enumerate(((AQUA, "$c_x$"), (YELLOW, "$c_y$"))):
        mu = log_on["cur_mu"][:, j]
        sd = np.sqrt(log_on["cur_cov"][:, j, j])
        ax.plot(t, true_c[:, j], color=color, ls=(0, (4, 3)), lw=1.0)
        ax.plot(t, mu, color=color, lw=1.6, label=f"{lbl} posterior mean")
        ax.fill_between(t, mu - 2 * sd, mu + 2 * sd, color=color, alpha=0.12, lw=0)
    ax.set_ylim(-0.8, 0.8)
    ax.annotate("dashed: truth · band: ±2σ", (0.995, 0.05),
                xycoords="axes fraction", ha="right", fontsize=7, color=MUTED)
    ax.set_ylabel("current estimate", fontsize=9, color=INK)
    leg = ax.legend(fontsize=8, loc="upper left", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(INK2)

    ax = axes[2]
    _style_axis(ax)
    ax.plot(t, err_off, color=MUTED, lw=1.2, label="frozen model")
    ax.plot(t, err_on, color=BLUE, lw=1.5, label="adaptive model")
    ax.set_ylabel("distance to goal", fontsize=9, color=INK)
    ax.set_xlabel("step", fontsize=9, color=INK)
    leg = ax.legend(fontsize=8, loc="upper left", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(INK2)

    for ax in axes:
        for b in bounds:
            ax.axvline(b, color=BASELINE, ls=":", lw=0.9)
    for b, (name, _) in zip([0] + bounds, acts):
        axes[0].annotate(name, (b + 4, 1.04), xycoords=("data", "axes fraction"),
                         fontsize=8, color=INK2)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {path}")
    return True


def save_hero_gif(path, log_on, log_off, acts, act_steps):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        return
    n = len(log_on["p"])
    fig = plt.figure(figsize=(11, 5.0))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.15, 1, 1],
                          left=0.03, right=0.98, top=0.90, bottom=0.08,
                          wspace=0.28, hspace=0.5)
    stage = Stage(fig.add_subplot(gs[:, :2]))
    axf = fig.add_subplot(gs[0, 2])
    axi = fig.add_subplot(gs[0, 3])
    axw = fig.add_subplot(gs[1, 2])
    axs = fig.add_subplot(gs[1, 3])

    fig.text(0.03, 0.945, "Image-only agent, adaptive vs frozen world model · "
             "the current changes at t = 100 and t = 200, unannounced",
             fontsize=9, color=INK2)

    _panel(axf, "observation (28×28)")
    im = axf.imshow(log_on["frame"][0], cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest")
    _panel(axi, "prediction (decoded plan endpoint)")
    imi = axi.imshow(log_on["imag"][0], cmap="gray", vmin=0, vmax=1,
                     interpolation="nearest")
    upd_belief = _belief_panel(axw)

    _panel(axs, "surprise (NIS)")
    axs.set_facecolor(SURFACE)
    nis = np.asarray(log_on["nis"])
    axs.semilogy(np.arange(n), nis, color=VIOLET, lw=0.8)
    axs.tick_params(colors=INK2, labelsize=6, length=2)
    axs.grid(True, color=GRID, lw=0.5)
    for b in [act_steps * i for i in range(1, len(acts))]:
        axs.axvline(b, color=BASELINE, ls=":", lw=0.9)
    cursor, = axs.plot([], [], "o", color=VIOLET, ms=3.5)

    def upd(i):
        i = min(i, n - 1)
        stage.update(i, i, log_on, log_off, acts)
        im.set_data(log_on["frame"][i])
        imi.set_data(log_on["imag"][i])
        upd_belief(acts[int(log_on["act"][i])][1],
                   log_on["cur_mu"][i], log_on["cur_cov"][i])
        cursor.set_data([i], [nis[i]])
        return ()

    anim = FuncAnimation(fig, upd, frames=range(0, n, 2), interval=70, blit=False)
    anim.save(path, writer=PillowWriter(fps=15), dpi=130)
    print(f"  saved {path}")
    try:                                       # MP4 companion, when ffmpeg exists
        from matplotlib.animation import FFMpegWriter
        mp4 = path.replace(".gif", ".mp4")
        anim.save(mp4, writer=FFMpegWriter(fps=15, bitrate=3200), dpi=160)
        print(f"  saved {mp4}")
    except Exception as e:
        print(f"  (no MP4 — {type(e).__name__}: install ffmpeg for a video)")
    plt.close(fig)


def save_loop_gif(path, log_on, log_off, acts, act_steps, chunk, exec_steps,
                  window=(96, 122)):
    """The agent cycle around the first regime change, one *phase event* per
    frame in true order — adapt (absorb chunk, remember, diffuse) happens at
    chunk boundaries, plan (VMP on the action sequence) at replan steps, act
    every step. A frame shows the state the phase operates on."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        return
    t0, t1 = window
    fig = plt.figure(figsize=(10, 5.0))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1.1],
                          left=0.03, right=0.97, top=0.90, bottom=0.08,
                          wspace=0.25, hspace=0.45)
    stage = Stage(fig.add_subplot(gs[:, :2]))
    axc = fig.add_subplot(gs[0, 2])
    axw = fig.add_subplot(gs[1, 2])

    fig.text(0.03, 0.945, "The agent cycle, one phase per frame — dotted: "
             "planned trajectory · arrow: applied thrust",
             fontsize=9, color=INK2)

    plan_line, = stage.ax.plot([], [], "o", color=BLUE, ms=2.2, alpha=0.8,
                               zorder=8)
    thrust = stage.ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=8,
                               arrowprops=dict(arrowstyle="-|>", color=INK,
                                               lw=1.5, mutation_scale=12))

    # Static phase list; only the marker and the type weight change.
    axc.set(xlim=(0, 1), ylim=(0, 1))
    axc.axis("off")
    ROWS = (("plan", "infer actions (VMP)", 0.86),
            ("act", "apply the first thrust", 0.68),
            ("adapt", "absorb chunk, update prior", 0.50))
    words = []
    for name, desc, y in ROWS:
        words.append(axc.annotate(name, (0.13, y), fontsize=12, color=MUTED,
                                  va="center"))
        axc.annotate(desc, (0.34, y), fontsize=7.5, color=MUTED, va="center")
    marker = axc.annotate("▸", (0.05, ROWS[0][2]), fontsize=12, color=INK,
                          va="center")

    upd_belief = _belief_panel(axw)

    # One frame per phase event, in the order they happen inside a step
    # (the chunk is absorbed first, then the plan, then the action).
    schedule = []
    for t in range(t0, t1):
        if t % chunk == 0:
            schedule += [("adapt", t)] * 2                 # hold the update
        if t % exec_steps == 0:
            schedule.append(("plan", t))
        schedule.append(("act", t))

    def upd(k):
        phase, i = schedule[k]
        pos_i = i if phase == "act" else i - 1             # pre-action state
        stage.update(pos_i, i, log_on, log_off, acts)
        path_i = log_on["plan_path"][i]
        plan_line.set_data(path_i[:, 0], path_i[:, 1])
        plan_line.set_alpha(0.9 if phase == "plan" else 0.35)
        p, u = log_on["p"][pos_i], log_on["u"][i]
        thrust.set_position(tuple(p))
        thrust.xy = tuple(p + 0.26 * u)
        thrust.arrow_patch.set_alpha(1.0 if phase == "act" else 0.0)
        row = {"plan": 0, "act": 1, "adapt": 2}[phase]
        marker.set_position((0.05, ROWS[row][2]))
        for j, w in enumerate(words):
            w.set_color(INK if j == row else MUTED)
            w.set_fontweight("bold" if j == row else "normal")
        upd_belief(acts[int(log_on["act"][i])][1],
                   log_on["cur_mu"][i if phase == "adapt" else pos_i],
                   log_on["cur_cov"][i if phase == "adapt" else pos_i])
        return ()

    anim = FuncAnimation(fig, upd, frames=len(schedule), interval=240, blit=False)
    anim.save(path, writer=PillowWriter(fps=4), dpi=130)
    plt.close(fig)
    print(f"  saved {path}")
