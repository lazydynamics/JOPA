"""Sensor and latent geometry, passed rather than imported.

A second environment needs its own frame size and latent split, so these travel
as one object. Anything that imports them as module constants can only ever
serve one environment.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorSpec:
    """Frame geometry and the pose/motion split the graph is built around."""

    img_size: int
    n_frames: int
    pose_dim: int
    motion_dim: int
    channels: int
    motion_hidden_dim: int

    @property
    def dim(self) -> int:
        """Latent dimension the transition and planner operate on."""
        return self.pose_dim + self.motion_dim


# The canonical pixel Reacher: four 64x64 frames, pose[4] + motion[2].
REACHER = SensorSpec(
    img_size=64,
    n_frames=4,
    pose_dim=4,
    motion_dim=2,
    channels=64,
    motion_hidden_dim=256,
)

# Local-refit policy: how many replay neighbours a refit uses, and how much
# precision to grant those encoded latents (they are observations, not data —
# the encoder's measured sigma is about 0.5).
NEIGHBORS = 512
REFIT_OBS_PRECISION = 4.0
