"""Artifact plumbing for the pixel Reacher: paths, manifest, replay, encodings."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jopa import Gaussian, LearnedLinear
from jopa.nn.vae import PoseMotionVAE

from .runtime import (
    ACTION_PRECISION_HOLD,
    ACTION_PRECISION_TRAVEL,
    ACTION_PRIOR_POWER,
    ACTION_PRIOR_SCALE,
    IMG_SIZE,
    MOTION_DIM,
    MOTION_GOAL_PRECISION,
    MOTION_HIDDEN_DIM,
    N_FRAMES,
    NEIGHBORS,
    POSE_DIM,
    POSE_GOAL_PRECISION,
    REFIT_OBS_PRECISION,
    SUBGOAL_TRUST,
)
from .validation import (
    MANIFEST_SCHEMA,
    ManifestError,
    checkpoint_sha256,
    split_trajectories,
)

REPLAY_PATTERN = "ck_*.pkl"
PREDICTION_WEIGHT = 10.0
VMP_ITERATIONS = 12
SENSOR_SEED = 8
SPLIT_SEED = 41
VALIDATION_REPEATS = 2
# Flat action precision, superseded by the conditional prior in ``runtime``
# and recorded in the manifest for provenance.
ACTION_PRECISION = 1.0
REFIT_REFRESH_DISTANCE = 0.0
# Physical audit protocol: a hold counts when the mean fingertip error over the
# final 20 closed-loop steps stays under 3 cm.
HOLD_WINDOW = 20
HOLD_THRESHOLD_CM = 3.0
REACH_THRESHOLD_CM = 1.0
SETTLED_WINDOW = 60
SETTLED_FRACTION = 0.95
SURVEY_POSES = 20
DEFAULT_POSE_SEED = 515151
MONTAGE_FPS = 25



def paths(args):
    output = Path(args.outdir)
    return {
        "out": output,
        "manifest": output / "manifest.json",
        "sensor": output / "pose_motion_sensor.msgpack",
        "bootstrap_dynamics": output / "conjugate_dynamics_bootstrap.pkl",
        "dynamics": output / "conjugate_dynamics.pkl",
        "background": output / "background.npy",
        "sensor_diagnostics": output / "sensor_diagnostics.json",
        "runtime": output / "runtime.json",
    }


def sensor_for(args):
    return PoseMotionVAE(
        pose_dim=POSE_DIM,
        motion_dim=MOTION_DIM,
        ch=args.channels,
        n_frames=N_FRAMES,
        img_size=IMG_SIZE,
        motion_hidden_dim=MOTION_HIDDEN_DIM,
    )


def architecture_for(args):
    return sensor_for(args).architecture()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray) or hasattr(value, "shape"):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True) + "\n")


def update_runtime(path, mode, seconds, **extra):
    report = {}
    if Path(path).is_file():
        try:
            report = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            report = {}
    report[mode] = {
        "seconds": float(seconds),
        **json_value(extra),
    }
    write_json(path, report)


def load_pixel_action_replay(args, names=None):
    # A frozen manifest replays its own recorded banks; None discovers them.
    if names is None:
        names = sorted(
            path.name for path in Path(args.data).glob(REPLAY_PATTERN))
    usable = []
    provenance = []
    for name in names:
        source = Path(args.data) / name
        if not source.is_file():
            continue
        with source.open("rb") as handle:
            loaded = pickle.load(handle)
        accepted = 0
        for chunk in loaded:
            # Only tuple fields zero and one are authorized. A legacy third
            # simulator-state field is deliberately never indexed.
            frames = np.asarray(chunk[0])
            controls = np.asarray(chunk[1])
            if frames.ndim != 3 or controls.ndim != 2:
                continue
            if len(frames) != len(controls) + 1:
                continue
            if len(frames) < N_FRAMES + args.rollout:
                continue
            if frames.shape[-2:] != (IMG_SIZE, IMG_SIZE):
                old_height, old_width = frames.shape[-2:]
                if old_height != old_width or old_height % IMG_SIZE:
                    raise ValueError(
                        f"cannot downsample {frames.shape[-2:]} "
                        f"to {(IMG_SIZE, IMG_SIZE)}")
                scale = old_height // IMG_SIZE
                frames = frames.reshape(
                    len(frames), IMG_SIZE, scale,
                    IMG_SIZE, scale).mean((2, 4))
            frames = frames.astype(np.float32)
            if frames.max(initial=0.0) > 1.0:
                frames /= 255.0
            usable.append((frames, controls.astype(np.float32)))
            accepted += 1
        provenance.append({
            "path": str(source.resolve()),
            "sha256": checkpoint_sha256(source),
            "loaded_chunks": len(loaded),
            "accepted_chunks": int(accepted),
        })
    if len(usable) < 3:
        raise ValueError(
            f"found only {len(usable)} usable pixel/action trajectories")
    return usable, provenance


def split_replay(args, manifest=None):
    frozen = None
    if manifest is not None and manifest.get("replay_provenance"):
        frozen = [
            Path(entry["path"]).name
            for entry in manifest["replay_provenance"]]
    replay, provenance = load_pixel_action_replay(args, names=frozen)
    split = split_trajectories(replay, seed=SPLIT_SEED)
    if manifest is not None:
        expected = manifest["split"]["indices"]
        if split["indices"] != expected:
            raise ManifestError(
                "current replay does not reproduce manifest split indices")
        if manifest.get("replay_provenance") != provenance:
            raise ManifestError(
                "replay files differ from the frozen manifest")
    return replay, split, provenance


def cache_path(output, name, sensor_hash):
    return output / f"encodings_{name}_{sensor_hash[:16]}.npz"


def encode_trajectory(observation, trajectory, batch_size=512):
    frames, controls = trajectory
    count = len(frames) - observation.n_frames + 1
    windows = np.stack([
        frames[index:index + observation.n_frames]
        for index in range(count)])
    means = []
    log_stds = []
    for start in range(0, count, batch_size):
        mean, log_std = observation.encode_batch(
            windows[start:start + batch_size])
        means.append(np.asarray(mean))
        log_stds.append(np.asarray(log_std))
    return {
        "means": np.concatenate(means),
        "log_stds": np.concatenate(log_stds),
        "controls": controls[observation.n_frames - 1:],
        "images": frames[observation.n_frames - 1:],
    }


def save_encoding_cache(path, encoded, metadata):
    arrays = {"metadata": np.asarray(json.dumps(metadata))}
    for index, item in enumerate(encoded):
        for name in ("means", "log_stds", "controls", "images"):
            arrays[f"{name}_{index:05d}"] = item[name]
    np.savez_compressed(path, **arrays)


def load_encoding_cache(path, expected_metadata):
    if not Path(path).is_file():
        return None
    try:
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive["metadata"]))
        if metadata != expected_metadata:
            return None
        encoded = []
        for index in range(metadata["trajectories"]):
            encoded.append({
                name: np.asarray(archive[f"{name}_{index:05d}"])
                for name in ("means", "log_stds", "controls", "images")
            })
        return encoded
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def encodings_for(
        args, split, observation, sensor_hash, *, force=False):
    result = {}
    for name in ("train", "calibration", "test"):
        metadata = {
            "sensor_sha256": sensor_hash,
            "split": name,
            "indices": split["indices"][name],
            "trajectories": len(split[name]),
            "calibrated": False,
        }
        path = cache_path(Path(args.outdir), name, sensor_hash)
        encoded = (
            None if force
            else load_encoding_cache(path, metadata))
        if encoded is None:
            encoded = [
                encode_trajectory(observation, trajectory)
                for trajectory in split[name]]
            save_encoding_cache(path, encoded, metadata)
        result[name] = encoded
    return result


def gaussian_sequence(item, offsets):
    messages = []
    for mean, log_std in zip(item["means"], item["log_stds"]):
        calibrated = np.clip(
            log_std + offsets, -6.0, 2.0)
        precision = jnp.diag(jnp.exp(-2.0 * jnp.asarray(calibrated)))
        mean_array = jnp.asarray(mean)
        messages.append(Gaussian(
            eta=precision @ mean_array, lam=precision))
    return messages


def fit_conjugate_dynamics(encoded, offsets):
    dimension = POSE_DIM + MOTION_DIM
    transition = LearnedLinear(
        dimension, du=2, offset=True,
        n_iterations=VMP_ITERATIONS,
        prior_a_cov=0.5, init_a_cov=0.5,
        prior_b_cov=100.0, init_b_cov=100.0,
        prior_W_df=dimension + 2)
    transition.learn(
        [gaussian_sequence(item, offsets) for item in encoded],
        [item["controls"] for item in encoded])
    return transition


def checkpoint_record(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": checkpoint_sha256(path),
    }


def training_configuration(args):
    return {
        "pose_steps": int(args.pose_steps),
        "controlled_steps": int(args.controlled_steps),
        "rollout": int(args.rollout),
        "prediction_weight": float(PREDICTION_WEIGHT),
        "batch_size": int(args.batch),
        "sensor_seed": int(SENSOR_SEED),
        "vmp_iterations": int(VMP_ITERATIONS),
        "pixels_and_actions_only": True,
        "bootstrap_predictor_deployed": False,
    }


def base_manifest(args, split, provenance):
    return {
        "schema": MANIFEST_SCHEMA,
        "phase": "initializing",
        "architecture": architecture_for(args),
        "training": training_configuration(args),
        "split": {
            "seed": int(split["seed"]),
            "ratios": split["ratios"],
            "indices": split["indices"],
        },
        "replay_provenance": provenance,
        "checkpoints": {},
        "calibration": None,
        "admission": None,
        "reproducibility": None,
        "scientific_policy": {
            "sensor_input": ["pixels", "actions"],
            "simulator_labels_read_during_training": False,
            "simulator_labels_read_during_validation": False,
            "bootstrap_linear_predictor_disposable": True,
            "deployed_dynamics": (
                "JOPA conjugate q(A,B,W), refit locally at the belief "
                "from frozen pixel-latent replay"),
            "planner": "infer_actions_exact",
            "amended_planning": {
                "pose_goal_precision": float(POSE_GOAL_PRECISION),
                "motion_goal_precision": float(MOTION_GOAL_PRECISION),
                "action_precision": float(ACTION_PRECISION),
                "action_prior": {
                    "kind": "conditional flat-top on latent goal distance",
                    "travel_precision": float(ACTION_PRECISION_TRAVEL),
                    "hold_precision": float(ACTION_PRECISION_HOLD),
                    "scale": float(ACTION_PRIOR_SCALE),
                    "power": float(ACTION_PRIOR_POWER),
                },
                "subgoal": "latent trust region",
                "subgoal_trust": float(SUBGOAL_TRUST),
                "localize_neighbors": int(NEIGHBORS),
                "refit_obs_prec": float(REFIT_OBS_PRECISION),
                "rollout_gate_localized": True,
                "rationale": (
                    "oracle diagnostics showed a single global linear model "
                    "has vanishing velocity-to-pose coupling on angle "
                    "embeddings and motion-heavy goal precision suppresses "
                    "movement; state was used only on the diagnostic bench, "
                    "never in the deployed loop"),
            },
        },
        "physical_audit": {
        },
    }


def attach_encoded_replay(transition, encoded, neighbors,
                          obs_prec=REFIT_OBS_PRECISION,
                          refresh_distance=REFIT_REFRESH_DISTANCE):
    transition.attach_replay(
        [item["means"] for item in encoded],
        [item["controls"] for item in encoded],
        neighbors=neighbors, refresh_every=1, obs_prec=obs_prec,
        refresh_distance=refresh_distance)
    return transition
