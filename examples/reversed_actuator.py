"""The LeWM-Reacher result in miniature: reaching with an actuator wired backwards.

A point mass is driven to goals by planning as inference, with its x
actuator wired backwards from the first step and a new goal at step 30.
The frozen agent pushes the wrong way; the adaptive one learns the actuator matrix A from the transitions it
observes with the ContinuousTransition rules (one `Actuator.update` per
step) and replans through E[A].

The dynamics are written by hand to show the extension point: any
`jopa.Dynamics` with a ``mean`` plugs into `Planner` and `Actuator`. On
LeWM's latent it is a learned `NeuralLinear` node instead.

    python -m examples.reversed_actuator
    python -m examples.reversed_actuator --gif assets/reversed_actuator.gif   # needs matplotlib
"""
import argparse

import jax
import jax.numpy as jnp

from jopa import Actuator, Dynamics, Planner
from jopa.nodes import apply_matrix


@jax.tree_util.register_pytree_node_class
class PointMass(Dynamics):
    """Damped point mass in the plane; the state is its position."""
    order, noise_var = 2, 1e-4

    def mean(self, states, u):
        prev, x = states
        return x + 0.8 * (x - prev) + 0.05 * u

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


TRUE_A = jnp.diag(jnp.array([-1.0, 1.0]))       # x actuator wired backwards
NEW_GOAL_AT = 30


def run(adapt, steps=60, seed=0):
    """Final distance to the goal, the final actuator belief, and the
    trajectory: positions (steps + 1, 2), goals (steps, 2), the mean and sd
    of A's diagonal (steps, 2)."""
    model, planner = PointMass(), Planner(horizon=5)
    plan = jax.jit(lambda act, prev, x, goal, key: planner(model, (prev, x), goal, key, 2,
                                                           actuator=act)[0][0])
    update = jax.jit(lambda act, *data: act.update(model, *data))
    act, key = Actuator.nominal(2, noise_var=model.noise_var), jax.random.key(seed)
    prev = x = jnp.zeros(2)
    goal = jnp.array([1.0, -0.5])
    trace = {"x": [x], "goal": [], "g": [], "g_sd": []}
    for t in range(steps):
        if t == NEW_GOAL_AT:
            goal = jnp.array([-1.0, 0.5])
        key, k = jax.random.split(key)
        u = plan(act, prev, x, goal, k)
        nxt = model.mean((prev, x), apply_matrix(u, TRUE_A))            # the real plant
        if adapt:                                           # one observed transition
            act = update(act, (prev[None], x[None]), u[None], nxt[None])
        prev, x = x, nxt
        trace["x"].append(x)
        trace["goal"].append(goal)
        trace["g"].append(jnp.diag(act.mean))
        trace["g_sd"].append(jnp.diag(act.sd))
    return float(jnp.linalg.norm(x - goal)), act, {k: jnp.stack(v) for k, v in trace.items()}


def animate(path, frozen, adaptive):
    """Both runs side by side, and the adaptive agent's belief about A₁₁."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    XLIM, YLIM = (-1.6, 3.4), (-1.9, 1.9)
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), facecolor="#fcfcfb",
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    planes = axes[:2]
    for ax, title in zip(planes, ("frozen: plans with A = I", "adaptive: plans with E[A]"), strict=True):
        ax.set(xlim=XLIM, ylim=YLIM, aspect="equal", xticks=[], yticks=[])
        ax.set_title(title, color=ink, fontsize=11, fontweight="bold", loc="left")
    gax = axes[2]
    steps = len(adaptive["g"])
    gax.set(xlim=(0, steps), ylim=(-1.6, 1.6))
    gax.axhline(0, color=grid, lw=0.8)
    gax.axhline(float(TRUE_A[0, 0]), color=muted, lw=1.2, ls="--", label="true A₁₁")
    gax.set_title("adaptive: belief about A₁₁", color=ink, fontsize=11,
                  fontweight="bold", loc="left")
    gax.set_xlabel("step", color=muted)
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        for side in ax.spines.values():
            side.set_color(grid)
    band = [None]
    line, = gax.plot([], [], color=blue, lw=2, label="posterior mean ± 2 sd")
    gax.legend(loc="lower left", frameon=False, fontsize=9)
    artists = []
    for ax, run_, col in zip(planes, (frozen, adaptive), (orange, blue), strict=True):
        goal, = ax.plot([], [], marker="*", ms=16, color=ink, ls="")
        trail, = ax.plot([], [], color=col, lw=1.5, alpha=0.5)
        dot, = ax.plot([], [], "o", ms=9, color=col)
        gone = ax.text(0.97, 0.05, "", transform=ax.transAxes, ha="right", color=col,
                       fontsize=10, fontweight="bold")
        artists.append((run_, goal, trail, dot, gone))
    status = fig.text(0.02, 0.02, "", color=muted, fontsize=10)

    def frame(t):
        for run_, goal, trail, dot, gone in artists:
            goal.set_data([run_["goal"][t][0]], [run_["goal"][t][1]])
            lo = max(0, t - 15)
            trail.set_data(run_["x"][lo:t + 2, 0], run_["x"][lo:t + 2, 1])
            x = run_["x"][t + 1]
            dot.set_data([x[0]], [x[1]])
            outside = not (XLIM[0] < x[0] < XLIM[1] and YLIM[0] < x[1] < YLIM[1])
            gone.set_text(f"off screen, {float(jnp.linalg.norm(x - run_['goal'][t])):.0f} from goal →"
                          if outside else "")
        m, sd = adaptive["g"][:t + 1, 0], adaptive["g_sd"][:t + 1, 0]
        line.set_data(range(t + 1), m)
        if band[0] is not None:
            band[0].remove()
        band[0] = gax.fill_between(range(t + 1), m - 2 * sd, m + 2 * sd, color=blue, alpha=0.2,
                                   lw=0)
        status.set_text(f"step {t}   x actuator wired backwards"
                        + ("   new goal" if t >= NEW_GOAL_AT else ""))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FuncAnimation(fig, frame, frames=steps).save(path, writer=PillowWriter(fps=8), dpi=90)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gif", help="also write an animation of both runs to this path")
    args = p.parse_args()
    runs = []
    for adapt in (False, True):
        err, act, trace = run(adapt)
        runs.append(trace)
        belief = ", ".join(f"{m:+.2f} ± {s:.2f}" for m, s in zip(act.mean.ravel(), act.sd.ravel(), strict=True))
        print(f"{'adaptive' if adapt else 'frozen  '}  distance to goal {err:6.3f}"
              f"   A = [{belief}]")
    if args.gif:
        animate(args.gif, *runs)
