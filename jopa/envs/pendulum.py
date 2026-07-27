"""Simple pendulum with headless PIL rendering to 28x28 grayscale."""
import numpy as np
from PIL import Image, ImageDraw

from ..config import DEFAULT_IMG_SIZE

# Angular speed bound (rad/s): keeps the explicit Euler integration stable at
# the default dt and keeps the bob inside the rendered frame.
_MAX_ANGULAR_SPEED = 8.0
# The scene is drawn at this multiple of the output resolution and downsampled,
# which anti-aliases the thin rod.
_RENDER_SUPERSAMPLE = 8
_ROD_LENGTH_FRACTION = 0.38     # rod length as a fraction of the frame
_BOB_RADIUS_SUPERSAMPLES = 2    # bob radius in supersampled pixels
_ROD_GRAY, _BOB_GRAY, _PIVOT_GRAY = 200, 255, 180


class SimplePendulum:
    """Pendulum: rod + bob, torque-controlled.

    State: (theta, theta_dot)
      theta = 0 → hanging down, pi → upright.
    Action: scalar torque clipped to [-max_torque, max_torque].
    """

    def __init__(self, g=9.81, m=1.0, l=1.0, dt=0.05,
                 max_torque=50.0, img_size=DEFAULT_IMG_SIZE):
        self.g, self.m, self.l, self.dt = g, m, l, dt
        self.max_torque = max_torque
        self.img_size = img_size
        self.state = np.array([0.0, 0.0])

    def reset(self, theta=None, theta_dot=None, seed=None):
        rng = np.random.RandomState(seed)
        if theta is None:
            theta = rng.uniform(-np.pi, np.pi)
        if theta_dot is None:
            theta_dot = rng.uniform(-1.0, 1.0)
        self.state = np.array([theta, theta_dot])
        return self.state.copy()

    def step(self, torque):
        torque = float(np.clip(torque, -self.max_torque, self.max_torque))
        theta, omega = self.state
        alpha = (-self.g / self.l) * np.sin(theta) + torque / (self.m * self.l ** 2)
        omega = np.clip(omega + alpha * self.dt,
                        -_MAX_ANGULAR_SPEED, _MAX_ANGULAR_SPEED)
        theta = ((theta + omega * self.dt) + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([theta, omega])
        return self.state.copy()

    def render(self):
        """Render current state as (img_size, img_size) float32 array in [0, 1]."""
        s = self.img_size
        scale = _RENDER_SUPERSAMPLE
        big = s * scale
        img = Image.new("L", (big, big), 0)
        draw = ImageDraw.Draw(img)

        cx, cy = big // 2, big // 2
        rod_len = big * _ROD_LENGTH_FRACTION

        # Rod endpoint (theta=0 → straight down)
        ex = cx + rod_len * np.sin(self.state[0])
        ey = cy + rod_len * np.cos(self.state[0])

        # Draw rod
        draw.line([(cx, cy), (ex, ey)], fill=_ROD_GRAY, width=max(1, scale))
        # Draw bob
        r = scale * _BOB_RADIUS_SUPERSAMPLES
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=_BOB_GRAY)
        # Draw pivot
        r2 = scale
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=_PIVOT_GRAY)

        arr = np.array(img.resize((s, s), Image.LANCZOS), dtype=np.float32) / 255.0
        return arr
