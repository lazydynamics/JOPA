"""Overdamped boat with headless PIL rendering to 28x28 grayscale.

A thrust-controlled boat in a square patch of water. Drag dominates, so
thrust sets velocity directly and position is the whole state — the
dynamics are exactly linear (affine) in it:

    p' = p + dt·(gain·u + current)

`current` and `gain` are plain attributes: mutate them mid-stream to change
the physics under the agent (regime switches for continual learning).
"""
import numpy as np
from PIL import Image, ImageDraw


class Boat:
    """Boat: thrust-controlled disc in the arena [-1, 1]².

    State: (px, py).
    Action: 2-D thrust, per-axis clip to [-max_thrust, max_thrust].
    """

    def __init__(self, current=(0.0, 0.0), gain=1.0, dt=0.2,
                 max_thrust=1.0, img_size=28):
        self.current = np.asarray(current, dtype=float)
        self.gain, self.dt = gain, dt
        self.max_thrust = max_thrust
        self.img_size = img_size
        self.state = np.zeros(2)

    def reset(self, p=None, seed=None):
        rng = np.random.RandomState(seed)
        p = rng.uniform(-0.8, 0.8, size=2) if p is None else np.asarray(p, dtype=float)
        self.state = p
        return self.state.copy()

    def step(self, u):
        u = np.clip(np.asarray(u, dtype=float), -self.max_thrust, self.max_thrust)
        p = self.state + self.dt * (self.gain * u + self.current)
        self.state = np.clip(p, -0.75, 0.75)   # banks, well inside the water:
        return self.state.copy()               # the boat never melts into the border

    def render(self):
        """Render current state as (img_size, img_size) float32 array in [0, 1]."""
        s = self.img_size
        scale = 8
        big = s * scale
        img = Image.new("L", (big, big), 0)
        draw = ImageDraw.Draw(img)

        # Faint border — a fixed spatial reference for the encoder.
        m = scale
        draw.rectangle([m, m, big - m, big - m], outline=70, width=scale // 2)

        # The boat: bright disc at p (arena [-1,1] → image, y up).
        cx = (self.state[0] + 1.0) / 2.0 * (big - 4 * m) + 2 * m
        cy = (1.0 - self.state[1]) / 2.0 * (big - 4 * m) + 2 * m
        r = 2.2 * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

        arr = np.array(img.resize((s, s), Image.LANCZOS), dtype=np.float32) / 255.0
        return arr
