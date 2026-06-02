"""JOPA — Bayesian inference and message passing for learning & control."""
from .distributions import (
    Gaussian, Wishart,
    combine_gaussians, gaussian_mean, gaussian_mean_cov, gaussian_prior,
    near_identity_prior, wishart_mean, vague_gaussian,
)
from .blocks import (
    Block, JointModel,
    Observation, Frozen, LearnedVAE,
    LearnedLinear, LearnedAffine, KnownPhysics,
    LinearCoupling,
)

__all__ = [
    "Gaussian", "Wishart",
    "combine_gaussians", "gaussian_mean", "gaussian_mean_cov",
    "gaussian_prior", "near_identity_prior", "wishart_mean", "vague_gaussian",
    "Block", "JointModel",
    "Observation", "Frozen", "LearnedVAE",
    "LearnedLinear", "LearnedAffine", "KnownPhysics",
    "LinearCoupling",
]
