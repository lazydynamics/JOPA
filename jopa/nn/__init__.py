"""Pretrained pixel sensors: fixed encoders that turn frames into Gaussian
messages for the message-passing core."""
from .layers import LOG_STD_CLIP, PROB_CLIP
from .persistence import (
    VAEAdapter,
    load_params,
    make_encode_decode,
    save_params,
)
from .pose_motion import PoseMotionVAE, train_pose_motion_vae
from .vae import VAE, train_vae

__all__ = [
    # sensors
    "VAE",
    "PoseMotionVAE",
    # training
    "train_vae",
    "train_pose_motion_vae",
    # persistence & adapters
    "save_params",
    "load_params",
    "make_encode_decode",
    "VAEAdapter",
    # shared clips
    "LOG_STD_CLIP",
    "PROB_CLIP",
]
