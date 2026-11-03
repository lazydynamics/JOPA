"""Pixel sensors: a convolutional VAE whose encoder emits Gaussian messages."""
from .vae import VAE, load_params, save_params, train_vae

__all__ = ["VAE", "load_params", "save_params", "train_vae"]
