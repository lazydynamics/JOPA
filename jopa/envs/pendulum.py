"""Simple pendulum with headless PIL rendering to 28x28 grayscale."""
import numpy as np
from PIL import Image, ImageDraw


class SimplePendulum:
    """Pendulum: rod + bob, torque-controlled.

    State: (theta, theta_dot)
      theta = 0 → hanging down, pi → upright.
    Action: scalar torque clipped to [-max_torque, max_torque].
    """

    def __init__(self, g=9.81, m=1.0, l=1.0, dt=0.05,
                 max_torque=50.0, img_size=28):
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
        omega = np.clip(omega + alpha * self.dt, -8.0, 8.0)
        theta = ((theta + omega * self.dt) + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([theta, omega])
        return self.state.copy()

    def render(self):
        """Render current state as (img_size, img_size) float32 array in [0, 1]."""
        s = self.img_size
        scale = 8
        big = s * scale
        img = Image.new("L", (big, big), 0)
        draw = ImageDraw.Draw(img)

        cx, cy = big // 2, big // 2
        rod_len = big * 0.38

        # Rod endpoint (theta=0 → straight down)
        ex = cx + rod_len * np.sin(self.state[0])
        ey = cy + rod_len * np.cos(self.state[0])

        # Draw rod
        draw.line([(cx, cy), (ex, ey)], fill=200, width=max(1, scale))
        # Draw bob
        r = scale * 2
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=255)
        # Draw pivot
        r2 = scale
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=180)

        arr = np.array(img.resize((s, s), Image.LANCZOS), dtype=np.float32) / 255.0
        return arr
