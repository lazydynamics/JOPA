"""JOPA — Bayesian inference and message passing for learning & control."""
from .distributions import (
    Gaussian, Wishart,
    combine_gaussians, gaussian_mean, gaussian_mean_cov, gaussian_prior,
    gaussian_logpdf, near_identity_prior, wishart_mean, vague_gaussian,
)
from .blocks import (
    Block, JointModel,
    Observation, Frozen, LearnedVAE, LearnedJEPA,
    LearnedLinear, LearnedAffine, KnownPhysics,
    LinearCoupling,
)
from .agent import Agent

__all__ = [
    "Agent",
    "Gaussian", "Wishart",
    "combine_gaussians", "gaussian_mean", "gaussian_mean_cov",
    "gaussian_prior", "gaussian_logpdf", "near_identity_prior",
    "wishart_mean", "vague_gaussian",
    "Block", "JointModel",
    "Observation", "Frozen", "LearnedVAE", "LearnedJEPA",
    "LearnedLinear", "LearnedAffine", "KnownPhysics",
    "LinearCoupling",
]
