"""Presentation layer for the boat demo — summary plot, demo animation, and
the plan/act/adapt loop animation. Pure matplotlib over the logs that
examples/boat.py records; no inference happens here.

Design rules (research-figure conventions): the subject — two crafts and
their divergence — is the largest thing on screen; time series are drawn
progressively with a playhead, never pre-completed; world events (the
current changing) are visible in the arena the moment they happen; axes
never rescale mid-animation. Palette validated for CVD separation; blue is
the adaptive model everywhere, dark gray the frozen one, aqua/yellow the
current components, violet surprise.
"""
import numpy as np

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, AQUA, YELLOW, VIOLET, ALERT = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#d03b3b"
WATERS = ("#eef3f8", "#e9f4f0", "#f4f1e6")     # per-regime water tint
TINTS = (None, AQUA, YELLOW)                   # per-regime span tint (plots)

_OCT = np.stack([np.cos(np.pi / 8 + np.arange(8) * np.pi / 4),
                 np.sin(np.pi / 8 + np.arange(8) * np.pi / 4)], axis=1)


def _fade_track(ax, color, lw=2.5, n=80):
    """A path whose recent segments are opaque and old ones fade out."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    lc = LineCollection([], linewidths=lw, capstyle="round", zorder=3)
    ax.add_collection(lc)

    def update(track):
        track = track[-n:]
        if len(track) < 3:
            lc.set_segments([])
            return
        segs = np.stack([track[:-1], track[1:]], axis=1)
        k = len(segs)
        lc.set_segments(segs)
        lc.set_colors([to_rgba(color, a) for a in np.linspace(0.05, 0.9, k)])

    return update


class _Craft:
    """Top-down omnidirectional thruster platform: octagonal body, four
    corner pods. Fixed orientation — the craft has no bow."""

    def __init__(self, ax, color, size, z):
        from matplotlib.patches import Circle, Polygon
        self.body_t = _OCT * size
        self.body = Polygon(self.body_t, closed=True, facecolor=color,
                            edgecolor=SURFACE, lw=0.8, zorder=z)
        ax.add_patch(self.body)
        self.off = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * size * 1.02
        self.pods = []
        for o in self.off:
            pod = Circle(tuple(o), size * 0.30, facecolor=SURFACE,
                         edgecolor=color, lw=1.1, zorder=z + 1)
            ax.add_patch(pod)
            self.pods.append(pod)
        self.track = _fade_track(ax, color, lw=2.5)

    def move(self, p, track):
        self.body.set_xy(self.body_t + p)
        for pod, o in zip(self.pods, self.off):
            pod.center = tuple(p + o)
        self.track(track)


class Stage:
    """Top-down view, cropped to the patrol area. The current is a field of
    arrows over the whole water — it appears and turns when the world does."""

    LIM = 0.88

    def __init__(self, ax):
        from matplotlib.patches import Polygon, Rectangle
        self.ax = ax
        ax.set(xlim=(-self.LIM, self.LIM), ylim=(-self.LIM, self.LIM),
               aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BASELINE)
        self.water = Rectangle((-1, -1), 2, 2, facecolor=WATERS[0], zorder=0)
        ax.add_patch(self.water)
        ax.add_patch(Rectangle((-0.75, -0.75), 1.5, 1.5, fill=False,
                               edgecolor=BASELINE, ls=(0, (4, 3)), lw=0.9))
        self.caption = ax.annotate("", (0.03, 0.975), xycoords="axes fraction",
                                   fontsize=9.5, color=INK2, va="top")
        self.event = ax.annotate("", (0.5, 0.80), xycoords="axes fraction",
                                 fontsize=12, color=INK, ha="center",
                                 fontweight="bold")

        gx, gy = np.meshgrid(np.linspace(-0.72, 0.72, 7),
                             np.linspace(-0.72, 0.72, 7))
        self.qx, self.qy = gx, gy
        self.flow = ax.quiver(gx, gy, np.zeros_like(gx), np.zeros_like(gy),
                              color=INK2, scale=5.5, width=0.0045,
                              alpha=0.15, zorder=1)

        self.goal_ring = ax.scatter([], [], s=420, facecolors="none",
                                    edgecolors=INK, linewidths=1.6, zorder=4)
        self.goal_dot = ax.scatter([], [], s=16, color=INK, zorder=4)

        self.frozen = _Craft(ax, INK2, 0.050, 5)
        self.adaptive = _Craft(ax, BLUE, 0.058, 7)

        # Legend outside the axes, above the arena.
        for x, color, lab in ((0.02, BLUE, "adaptive model"),
                              (0.40, INK2, "frozen model")):
            ax.add_patch(Polygon(_OCT * 0.023 + [(x + 0.025) * 2 * self.LIM - self.LIM,
                                                 self.LIM + 0.10],
                                 closed=True, facecolor=color, lw=0,
                                 clip_on=False, zorder=10))
            ax.annotate(lab, (x + 0.06, 1.045), xycoords="axes fraction",
                        fontsize=9, color=INK2, va="center", clip_on=False)

    def update(self, pos_i, info_i, log_on, log_off, acts, act_steps):
        act_i = int(log_on["act"][info_i])
        name, cur = acts[act_i]
        cur = np.asarray(cur, dtype=float)
        self.caption.set_text(f"t = {info_i} · {name}")
        self.water.set_facecolor(WATERS[act_i])

        since = info_i - act_i * act_steps if act_i > 0 else 99
        self.flow.set_UVC(np.full_like(self.qx, cur[0]),
                          np.full_like(self.qy, cur[1]))
        self.flow.set_alpha(0.45 if since < 8 else 0.15)
        self.event.set_text("the current changes"
                            if act_i > 0 and since < 15 else "")
        self.event.set_alpha(max(0.0, 1.0 - since / 15) if act_i > 0 else 0.0)

        self.goal_ring.set_offsets([log_on["goal"][info_i]])
        self.goal_dot.set_offsets([log_on["goal"][info_i]])
        lo = max(0, pos_i - 80)
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


def _regime_spans(ax, acts, act_steps, upto=None, alpha=0.06, label=False):
    """Light per-regime tints; with `upto`, returns spans to reveal later."""
    spans = []
    for k in range(1, len(acts)):
        s = ax.axvspan(k * act_steps, (k + 1) * act_steps, lw=0,
                       color=TINTS[k], alpha=alpha)
        if upto is not None:
            s.set_visible(False)
        spans.append((k * act_steps, s))
        if label:
            ax.annotate(acts[k][0], (k * act_steps + 4, 1.03),
                        xycoords=("data", "axes fraction"), fontsize=7.5,
                        color=INK2)
    return spans


def _belief_panel(ax):
    """q(c) on fixed axes: the ellipse sliding and tightening IS the story."""
    from matplotlib.patches import Ellipse
    _panel(ax, "current estimate q(c), ±2σ")
    ax.set(xlim=(-0.75, 0.75), ylim=(-0.75, 0.75), aspect="equal")
    ax.set_facecolor(SURFACE)
    ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xticklabels(["−.5", "0", ".5"]); ax.set_yticklabels(["−.5", "0", ".5"])
    ax.tick_params(colors=MUTED, labelsize=6, length=2)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    true_pt = ax.scatter([], [], marker="x", s=80, color=INK, lw=1.6, zorder=5)
    trail = ax.scatter([], [], s=6, color=AQUA, zorder=4)
    post_pt = ax.scatter([], [], s=26, color=AQUA, zorder=6)
    ell = Ellipse((0, 0), 0, 0, fill=True, alpha=0.15, color=AQUA)
    ax.add_patch(ell)
    ell.set_clip_box(ax.bbox)
    ax.annotate("× true", (0.05, 0.05), xycoords="axes fraction",
                fontsize=7, color=MUTED)

    def update(cur, c_mu, c_cov, mean_trail=None):
        true_pt.set_offsets([np.asarray(cur)])
        post_pt.set_offsets([c_mu])
        if mean_trail is not None and len(mean_trail):
            trail.set_offsets(mean_trail)
            trail.set_alpha(0.25)
        evals, evecs = np.linalg.eigh(c_cov)
        ell.set_center(c_mu)
        ell.width, ell.height = 4 * np.sqrt(np.maximum(evals, 1e-12))
        ell.angle = float(np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1])))
        return ell

    return update


def _progressive_strip(ax, ylabel, n):
    """A time series drawn up to the playhead, never pre-completed."""
    _style_axis(ax)
    ax.set_xlim(0, n)
    ax.set_ylabel(ylabel, fontsize=8, color=INK)
    playhead = ax.axvline(0, color=INK2, lw=0.9)
    return playhead


def save_money_plot(path, log_on, log_off, err_on, err_off, acts, act_steps):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib for the plots:  uv pip install -e '.[viz]')")
        return False
    t = np.arange(len(err_on))
    goal_switch = np.flatnonzero(np.any(np.diff(log_on["goal"], axis=0) != 0,
                                        axis=1)) + 1
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.6), sharex=True,
                             gridspec_kw={"hspace": 0.45,
                                          "height_ratios": [1.3, 0.8, 0.8, 1.0]})
    fig.patch.set_facecolor(SURFACE)

    # 1 — the outcome first: tracking error, gap tinted.
    ax = axes[0]
    _style_axis(ax)
    ax.plot(t, err_off, color=INK2, lw=1.3, label="frozen model")
    ax.plot(t, err_on, color=BLUE, lw=1.6, label="adaptive model")
    ax.fill_between(t, err_on, err_off, where=err_off > err_on,
                    color=ALERT, alpha=0.10, lw=0)
    ax.scatter(goal_switch, np.zeros_like(goal_switch) - 0.06, marker="^",
               s=18, color=MUTED, clip_on=False)
    ax.annotate("▲ goal moves", (0.995, -0.24), xycoords="axes fraction",
                ha="right", fontsize=7, color=MUTED)
    ax.set_ylabel("distance to goal", fontsize=9, color=INK)
    _regime_spans(ax, acts, act_steps, label=True)
    leg = ax.legend(fontsize=8, loc="upper left", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(INK2)

    # 2/3 — the current posterior, one thin panel per component (no overlap).
    true_c = np.array([acts[a][1] for a in log_on["act"]])
    for ax, j, color, lbl in ((axes[1], 0, AQUA, "$c_x$"),
                              (axes[2], 1, YELLOW, "$c_y$")):
        _style_axis(ax)
        mu = log_on["cur_mu"][:, j]
        sd = np.sqrt(log_on["cur_cov"][:, j, j])
        ax.plot(t, true_c[:, j], color=color, ls=(0, (4, 3)), lw=1.0)
        ax.plot(t, mu, color=color, lw=1.6)
        ax.fill_between(t, mu - 2 * sd, mu + 2 * sd, color=color, alpha=0.15, lw=0)
        ax.set_ylim(-0.75, 0.75)
        ax.set_ylabel(lbl, fontsize=9, color=INK)
        _regime_spans(ax, acts, act_steps)
    axes[1].annotate("current estimate — solid: posterior mean · dashed: truth "
                     "· band: ±2σ", (0.995, 1.08), xycoords="axes fraction",
                     ha="right", fontsize=7, color=MUTED)

    # 4 — surprise: raw trace faint, rolling median solid.
    ax = axes[3]
    _style_axis(ax)
    nis = np.asarray(log_on["nis"], dtype=float)
    ax.semilogy(t, nis, color=VIOLET, lw=0.7, alpha=0.25)
    k = 9
    med = np.array([np.nanmedian(nis[max(0, i - k):i + 1]) for i in t])
    ax.semilogy(t, med, color=VIOLET, lw=1.5)
    ax.axhline(4.0, color=BASELINE, ls="--", lw=0.9)
    ax.annotate("E[χ²] under the model", (0.995, 0.10), xycoords="axes fraction",
                ha="right", fontsize=7, color=MUTED)
    ax.set_ylabel("surprise (NIS)", fontsize=9, color=INK)
    ax.set_xlabel("step", fontsize=9, color=INK)
    _regime_spans(ax, acts, act_steps)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {path}")
    return True


def _hero_schedule(n, acts_at, slow_pad=(5, 50), hold=12, tail=22):
    """Event-weighted frames: fast-forward calm stretches, single-step through
    the windows around each regime change, hold on the change, rest at the end."""
    windows = [(a - slow_pad[0], a + slow_pad[1]) for a in acts_at]
    frames = []
    for t in range(n):
        in_window = any(lo <= t < hi for lo, hi in windows)
        if t in acts_at:
            frames.extend([t] * hold)
        elif in_window:
            frames.append(t)
        elif t % 4 == 0:
            frames.append(t)
    frames.extend([n - 1] * tail)
    return frames


def save_hero_gif(path, log_on, log_off, err_on, err_off, acts, act_steps):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        return
    n = len(log_on["p"])
    fig = plt.figure(figsize=(11, 5.2))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 3, width_ratios=[2.15, 1, 1],
                          height_ratios=[1.5, 0.72, 0.72],
                          left=0.035, right=0.975, top=0.865, bottom=0.09,
                          wspace=0.30, hspace=0.62)
    stage = Stage(fig.add_subplot(gs[:, 0]))
    axf = fig.add_subplot(gs[0, 1])
    axw = fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs[1, 1:])
    axs = fig.add_subplot(gs[2, 1:], sharex=axd)

    fig.text(0.035, 0.935, "Image-only agent, adaptive vs frozen world model · "
             "the current changes twice, unannounced", fontsize=9.5, color=INK2)

    _panel(axf, "observation + predicted endpoint (blue)")
    im = axf.imshow(log_on["frame"][0], cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest")
    contour = [axf.contour(log_on["imag"][0], levels=[0.5], colors=[BLUE],
                           linewidths=1.5, origin="upper")]
    upd_belief = _belief_panel(axw)

    play_d = _progressive_strip(axd, "dist. to goal", n)
    axd.set_ylim(0, 1.25)
    axd.tick_params(labelbottom=False)
    line_off, = axd.plot([], [], color=INK2, lw=1.1)
    line_on, = axd.plot([], [], color=BLUE, lw=1.4)
    spans_d = _regime_spans(axd, acts, act_steps, upto=0)

    play_s = _progressive_strip(axs, "surprise (NIS)", n)
    axs.set_yscale("log")
    axs.set_ylim(3e-3, 3e2)
    axs.set_xlabel("step", fontsize=8, color=INK)
    axs.axhline(4.0, color=BASELINE, ls="--", lw=0.8)
    nis = np.asarray(log_on["nis"], dtype=float)
    line_nis, = axs.plot([], [], color=VIOLET, lw=0.9)
    flare = axs.scatter([], [], s=8, color=ALERT, zorder=5)
    spans_s = _regime_spans(axs, acts, act_steps, upto=0)

    t_all = np.arange(n)
    err_on = np.asarray(err_on)
    err_off = np.asarray(err_off)

    def upd(i):
        stage.update(i, i, log_on, log_off, acts, act_steps)
        im.set_data(log_on["frame"][i])
        contour[0].remove()
        contour[0] = axf.contour(log_on["imag"][i], levels=[0.5], colors=[BLUE],
                                 linewidths=1.5, origin="upper")
        upd_belief(acts[int(log_on["act"][i])][1],
                   log_on["cur_mu"][i], log_on["cur_cov"][i],
                   log_on["cur_mu"][max(0, i - 60):i:3])
        line_on.set_data(t_all[:i + 1], err_on[:i + 1])
        line_off.set_data(t_all[:i + 1], err_off[:i + 1])
        line_nis.set_data(t_all[:i + 1], nis[:i + 1])
        hot = t_all[:i + 1][nis[:i + 1] > 4.0]
        flare.set_offsets(np.c_[hot, nis[hot]] if len(hot) else
                          np.empty((0, 2)))
        for spans in (spans_d, spans_s):
            for start, s in spans:
                s.set_visible(i >= start)
        for ph in (play_d, play_s):
            ph.set_xdata([i, i])
        return ()

    frames = _hero_schedule(n, (act_steps, 2 * act_steps))
    anim = FuncAnimation(fig, upd, frames=frames, interval=67, blit=False)
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
                  window=(96, 118)):
    """The agent cycle around the first regime change. Each phase gets frames
    in which the thing it names visibly happens: plan draws the planned
    trajectory, act moves the craft with its thrust, adapt moves q(c)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Rectangle
    except ImportError:
        return
    t0, t1 = window
    fig = plt.figure(figsize=(10, 5.0))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1],
                          height_ratios=[0.72, 1.28],
                          left=0.035, right=0.965, top=0.86, bottom=0.07,
                          wspace=0.22, hspace=0.30)
    stage = Stage(fig.add_subplot(gs[:, 0]))
    axc = fig.add_subplot(gs[0, 1])
    axw = fig.add_subplot(gs[1, 1])

    fig.text(0.035, 0.93, "The agent cycle around a regime change — dotted: "
             "planned trajectory · arrow: applied thrust", fontsize=9.5,
             color=INK2)

    plan_line, = stage.ax.plot([], [], "o", color=BLUE, ms=3.0, alpha=0.9,
                               zorder=8)
    thrust = stage.ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=8,
                               arrowprops=dict(arrowstyle="-|>", color=INK,
                                               lw=1.8, mutation_scale=14))

    # Phase checklist: fixed texts; a highlight bar marks the active row.
    axc.set(xlim=(0, 1), ylim=(0, 1))
    axc.axis("off")
    ROWS = (("plan", "infer actions (VMP)", 0.78),
            ("act", "apply the first thrust", 0.50),
            ("adapt", "absorb chunk, update prior", 0.22))
    bar = Rectangle((0.02, 0), 0.96, 0.24, facecolor=GRID, alpha=0.5, lw=0)
    axc.add_patch(bar)
    words = []
    for name, desc, y in ROWS:
        words.append(axc.annotate(name, (0.06, y), fontsize=12, color=MUTED,
                                  va="center", fontweight="bold"))
        axc.annotate(desc, (0.36, y), fontsize=8, color=MUTED, va="center")

    upd_belief = _belief_panel(axw)
    adapt_arrow = axw.annotate("", xy=(0, 0), xytext=(0, 0), zorder=7,
                               arrowprops=dict(arrowstyle="-|>", color=INK,
                                               lw=1.2, mutation_scale=9))

    # Frame schedule: plan gets two frames (the trajectory draws itself),
    # act one (the craft moves), adapt two (q(c) jumps, then settles).
    schedule = []
    for t in range(t0, t1):
        if t % chunk == 0:
            schedule += [("adapt", t, 0), ("adapt", t, 1)]
        if t % exec_steps == 0:
            schedule += [("plan", t, 0), ("plan", t, 1)]
        schedule.append(("act", t, 0))

    def upd(k):
        phase, i, sub = schedule[k]
        pos_i = i if phase == "act" else i - 1
        stage.update(pos_i, i, log_on, log_off, acts, act_steps)
        stage.caption.set_text(stage.caption.get_text() + f" — {phase}")

        path_i = log_on["plan_path"][i]
        upto = len(path_i) // 2 if (phase == "plan" and sub == 0) else len(path_i)
        plan_line.set_data(path_i[:upto, 0], path_i[:upto, 1])
        plan_line.set_alpha(0.9 if phase == "plan" else 0.3)

        p, u = log_on["p"][pos_i], log_on["u"][i]
        un = u / max(np.linalg.norm(u), 1e-6)
        thrust.set_position(tuple(p + 0.09 * un))
        thrust.xy = tuple(p + 0.09 * un + 0.30 * u)
        thrust.arrow_patch.set_alpha(1.0 if phase == "act" else 0.0)

        row = {"plan": 0, "act": 1, "adapt": 2}[phase]
        bar.set_y(ROWS[row][2] - 0.12)
        for j, w in enumerate(words):
            w.set_color(INK if j == row else MUTED)

        show_new = not (phase == "adapt" and sub == 0)
        b_i = i if show_new else i - 1
        ell = upd_belief(acts[int(log_on["act"][i])][1],
                         log_on["cur_mu"][b_i], log_on["cur_cov"][b_i])
        ell.set_linewidth(2.2 if phase == "adapt" else 0.0)
        ell.set_edgecolor(AQUA)
        if phase == "adapt" and sub == 1:
            adapt_arrow.set_position(tuple(log_on["cur_mu"][i - 1]))
            adapt_arrow.xy = tuple(log_on["cur_mu"][i])
            adapt_arrow.arrow_patch.set_alpha(1.0)
        else:
            adapt_arrow.arrow_patch.set_alpha(0.0)
        return ()

    anim = FuncAnimation(fig, upd, frames=len(schedule), interval=180, blit=False)
    anim.save(path, writer=PillowWriter(fps=5), dpi=130)
    plt.close(fig)
    print(f"  saved {path}")
