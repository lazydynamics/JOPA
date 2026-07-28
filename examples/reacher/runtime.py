"""Reusable closed-loop runtime for the pixel Reacher.

Loads a frozen sensor and its conjugate dynamics, attaches the encoded replay
used for local refits, and runs episodes. Shared by evaluation and by the
figure scripts so they cannot drift apart.
"""
from __future__ import annotations

import copy
import glob
import json
import pickle
from pathlib import Path

import numpy as np

from jopa import Agent, Block, JointModel, PoseMotionObservation
from jopa.nn.vae import PoseMotionVAE, load_params

from .spec import NEIGHBORS, REACHER, REFIT_OBS_PRECISION

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "outputs/reacher_pixels"

# Geometry lives in `spec`; these are re-exported so existing callers and the
# figure scripts keep one import site.
IMG_SIZE = REACHER.img_size
N_FRAMES = REACHER.n_frames
POSE_DIM = REACHER.pose_dim
MOTION_DIM = REACHER.motion_dim
SENSOR_CHANNELS = REACHER.channels
MOTION_HIDDEN_DIM = REACHER.motion_hidden_dim
POSE_GOAL_PRECISION = 200.0
MOTION_GOAL_PRECISION = 30.0
GOAL_EIGENVALUE_FLOOR = 0.5
SUBGOAL_TRUST = 1.0
HORIZON = 6
EVAL_STEPS = 240
# Conditional action prior: effort is cheap while the belief is far from the
# goal and expensive inside the damping basin. A flat-top (quartic) kernel
# keeps travel cost unmodified outside the basin, which a Gaussian or rational
# blend does not.
ACTION_PRECISION_TRAVEL = 0.2
ACTION_PRECISION_HOLD = 6.0
ACTION_PRIOR_SCALE = 1.3
ACTION_PRIOR_POWER = 4.0
CAMERA_DISTANCE = 0.34
WIDE_CAMERA_DISTANCE = 0.55
WIDE_CAMERA_ELEVATION = -70.0
WIDE_CAMERA_AZIMUTH = 90.0
DISPLAY_SIZE = 320
ACTION_REPEAT = 2
JOINT2_START_LIMIT = 2.8
JOINT2_GOAL_LIMIT = 2.6
MIN_TASK_SEPARATION = 0.05


def action_precision(distance):
    """Effort is cheap far from the goal and expensive inside the basin."""
    blend = float(np.exp(-((distance / ACTION_PRIOR_SCALE) ** ACTION_PRIOR_POWER)))
    return ACTION_PRECISION_TRAVEL + (
        ACTION_PRECISION_HOLD - ACTION_PRECISION_TRAVEL) * blend


def latent_subgoal(belief_mean, goal_mean, trust=SUBGOAL_TRUST):
    delta = np.asarray(goal_mean) - np.asarray(belief_mean)
    distance = float(np.linalg.norm(delta))
    if distance <= trust:
        return goal_mean
    return belief_mean + trust * delta / distance


# The goal is handed over as an image, so show it as one: the target pose is
# rendered from the same camera and composited under the live frame. The arena is
# a uniform grey, so the goal's arm is the pixels differing from the arena colour,
# taken as the per-channel median of the goal render.
#
# The ghost is lifted toward white rather than darkened: on a mid-grey arena that
# is the direction with contrast to spare (66 levels against 26), so it survives
# being downscaled into a small panel, and only the live arm keeps its hue.
GHOST_STRENGTH = 0.85
GHOST_LIGHT = 0.55
ARENA_TOLERANCE = 24.0


def ghosted(live, goal_render):
    """Composite a rendered goal pose under a live frame as a pale double."""
    goal = np.asarray(goal_render, dtype=np.float32)
    frame = np.asarray(live, dtype=np.float32)
    arena = np.median(goal.reshape(-1, 3), axis=0)
    # Ghost only where the goal's arm is and the live frame's is not: the target
    # marker sits in both renders, so differencing against the live frame keeps
    # its colour, and an overlapping live arm stays in front of its own ghost.
    mask = ((np.abs(goal - arena).sum(-1) > ARENA_TOLERANCE)
            & (np.abs(goal - frame).sum(-1) > ARENA_TOLERANCE))
    pale = arena + GHOST_LIGHT * (255.0 - arena)
    out = frame.copy()
    out[mask] = (1.0 - GHOST_STRENGTH) * out[mask] + GHOST_STRENGTH * pale
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_bank(directory, sensor_hash, include_rest=False):
    """Encoded replay for local refits.

    `include_rest` adds `rest_bank.npz`, extra low-speed babble collected for
    the holding regime. It is off by default because the reported metrics were
    measured without it — turning it on is a change to the experiment, not a
    change to the code, and needs its own measurement.
    """
    means, controls = [], []
    sources = sorted(glob.glob(str(directory / f"encodings_train_{sensor_hash}*.npz")))
    extra = directory / "rest_bank.npz"
    if include_rest and extra.is_file():
        sources.append(str(extra))
    for source in sources:
        archive = np.load(source)
        count = json.loads(str(archive["metadata"]))["trajectories"]
        means += [np.asarray(archive[f"means_{i:05d}"]) for i in range(count)]
        controls += [np.asarray(archive[f"controls_{i:05d}"]) for i in range(count)]
    return means, controls


class ReacherLoop:
    """Frozen sensor + conjugate dynamics driving MuJoCo Reacher from pixels."""

    def __init__(self, directory=ARTIFACTS, record=False,
                 include_rest=False):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        sensor_hash = manifest["checkpoints"]["sensor"]["sha256"][:16]
        self.sensor = PoseMotionVAE(
            pose_dim=POSE_DIM, motion_dim=MOTION_DIM, ch=SENSOR_CHANNELS,
            n_frames=N_FRAMES, img_size=IMG_SIZE,
            motion_hidden_dim=MOTION_HIDDEN_DIM)
        params = load_params(self.sensor, directory / "pose_motion_sensor.msgpack")
        with (directory / "conjugate_dynamics.pkl").open("rb") as handle:
            self.transition = pickle.load(handle)
        offsets = np.asarray(
            manifest["calibration"]["log_std_offsets"], dtype=np.float32)
        self.observation = PoseMotionObservation(
            self.sensor, params, log_std_offsets=offsets)
        self.background = np.load(directory / "background.npy", allow_pickle=False)
        self.transition.attach_replay(
            *_load_bank(directory, sensor_hash, include_rest),
            neighbors=NEIGHBORS,
            refresh_every=1, obs_prec=REFIT_OBS_PRECISION)

        import gymnasium as gym
        import mujoco

        self._mujoco = mujoco
        self.environment = gym.make("Reacher-v5", render_mode="rgb_array")
        self.world = self.environment.unwrapped.model
        self.data = self.environment.unwrapped.data
        self._fingertip = self.world.geom("fingertip").id
        self._target = self.world.geom("target").id
        self._renderer = mujoco.Renderer(self.world, 2 * IMG_SIZE, 2 * IMG_SIZE)
        self._camera = self._make_camera(CAMERA_DISTANCE)
        self._rgba = self.world.geom_rgba[self._target].copy()
        self._size = self.world.geom_size[self._target].copy()
        self.record = record
        if record:
            self._display = mujoco.Renderer(self.world, DISPLAY_SIZE, DISPLAY_SIZE)
            self._wide = self._make_camera(
                WIDE_CAMERA_DISTANCE, WIDE_CAMERA_ELEVATION,
                WIDE_CAMERA_AZIMUTH)

    def _make_camera(self, distance, elevation=-90.0, azimuth=90.0):
        camera = self._mujoco.MjvCamera()
        camera.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0, 0, 0]
        camera.distance = distance
        camera.elevation, camera.azimuth = elevation, azimuth
        return camera

    def sensor_frame(self):
        """Grayscale frame with the target hidden, matched to training."""
        self.world.geom_rgba[self._target] = self._rgba * np.array(
            [1, 1, 1, 0], np.float32)
        self._renderer.update_scene(self.data, camera=self._camera)
        gray = self._renderer.render().copy().mean(2).astype(np.float32)
        self.world.geom_rgba[self._target] = self._rgba
        return gray.reshape(IMG_SIZE, 2, IMG_SIZE, 2).mean((1, 3)) / 255.0

    def display_frame(self):
        self.world.geom_rgba[self._target] = np.array([0.62, 0.42, 0.35, 1.0],
                                                      np.float32)
        self.world.geom_size[self._target] = self._size * 2.2
        self._display.update_scene(self.data, camera=self._wide)
        image = self._display.render().copy()
        self.world.geom_size[self._target] = self._size
        self.world.geom_rgba[self._target] = self._rgba
        return image

    def reset(self, pose, target_xy):
        self._mujoco.mj_resetData(self.world, self.data)
        self.data.qpos[:2] = pose
        self.data.qpos[2:4] = target_xy
        self.data.qvel[:] = 0
        self._mujoco.mj_forward(self.world, self.data)

    def fingertip(self):
        return self.data.geom_xpos[self._fingertip][:2].copy()

    def fingertip_of(self, pose):
        position, velocity = self.data.qpos.copy(), self.data.qvel.copy()
        self.data.qpos[:2] = pose
        self.data.qvel[:] = 0
        self._mujoco.mj_forward(self.world, self.data)
        result = self.fingertip()
        self.data.qpos[:], self.data.qvel[:] = position, velocity
        self._mujoco.mj_forward(self.world, self.data)
        return result

    def act(self, control):
        self.data.ctrl[:] = np.clip(control, -1, 1)
        for _ in range(ACTION_REPEAT):
            self._mujoco.mj_step(self.world, self.data)

    def sample_task(self, seed):
        rng = np.random.RandomState(seed)
        for _ in range(200):
            start = rng.uniform(-np.pi, np.pi, 2)
            goal = rng.uniform(-np.pi, np.pi, 2)
            start[1] *= JOINT2_START_LIMIT / np.pi
            goal[1] *= JOINT2_GOAL_LIMIT / np.pi
            if np.linalg.norm(self.fingertip_of(start)
                              - self.fingertip_of(goal)) > MIN_TASK_SEPARATION:
                return start, goal
        raise RuntimeError("no sufficiently separated task for this seed")

    def agent_for(self, goal_frame, **overrides):
        model = JointModel([Block("z", copy.deepcopy(self.transition),
                                  observe=self.observation)])
        precision = self.observation.goal_precision(
            goal_frame, background=self.background,
            pose_mean_precision=POSE_GOAL_PRECISION,
            motion_precision=MOTION_GOAL_PRECISION,
            eigenvalue_floor=GOAL_EIGENVALUE_FLOOR)
        settings = dict(horizon=HORIZON, action_precision=action_precision,
                        goal_precision=precision, goal_schedule="stage",
                        subgoal=latent_subgoal, relin_surprise=1e9,
                        relin_radius=0.25, adapt_hold=False)
        settings.update(overrides)
        agent = Agent(model, **settings)
        agent.goal(goal_frame)
        return agent

    def episode(self, seed, steps=EVAL_STEPS, **overrides):
        start_pose, goal_pose = self.sample_task(seed)
        goal_xy = self.fingertip_of(goal_pose)
        self.reset(goal_pose, goal_xy)
        goal_frame = self.sensor_frame()
        agent = self.agent_for(goal_frame, **overrides)
        self.reset(start_pose, goal_xy)
        history = [self.sensor_frame()] * N_FRAMES
        error, control, frames = [], [], []
        for _ in range(steps):
            action = agent.step(np.stack(history[-N_FRAMES:]))
            if self.record:
                frames.append(self.display_frame())
            self.act(action)
            history.append(self.sensor_frame())
            error.append(float(np.linalg.norm(self.fingertip() - goal_xy) * 100.0))
            control.append(np.asarray(action, dtype=float))
        return {
            "seed": seed, "agent": agent, "frames": frames,
            "frame_size": DISPLAY_SIZE,
            "error": np.asarray(error), "control": np.asarray(control),
            "start_cm": float(np.linalg.norm(
                self.fingertip_of(start_pose) - goal_xy) * 100.0),
            "goal_frame": goal_frame,
        }

    def close(self):
        self._renderer.close()
        if self.record:
            self._display.close()
        self.environment.close()
