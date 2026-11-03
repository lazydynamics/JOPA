"""Reacher with a declared actuator change, for the adaptation experiment.

``gym.make("shift:ReacherShift-v0", gain=[g1, g2])`` is upstream's
``ReacherDMControlWrapper`` with each commanded torque multiplied by its
joint's gain before it reaches the simulator (``gain=[-1, 1]``: shoulder wired
backwards; ``[0.3, 0.3]``: weak actuators). Everything else, including
rendering, success test and variation space, is upstream's.
"""
import numpy as np
from gymnasium.envs.registration import register
from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper


class ReacherShift(ReacherDMControlWrapper):
    def __init__(self, gain=(1.0, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.gain = np.asarray(gain, dtype=np.float64)

    def step(self, action):
        return super().step(np.clip(np.asarray(action) * self.gain, -1.0, 1.0))


register(id="ReacherShift-v0", entry_point="shift:ReacherShift")
