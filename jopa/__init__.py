"""JOPA — Bayesian inference and message passing for learning & control."""
from .agent import Agent
from .distributions import (
    Gaussian,
    Wishart,
    combine_gaussians,
    gaussian_logpdf,
    gaussian_mean,
    gaussian_mean_cov,
    gaussian_prior,
    near_identity_prior,
    vague_gaussian,
    wishart_mean,
)
from .graph import Block, JointModel, LinearCoupling
from .observations import (
    Frozen,
    LearnedVAE,
    Observation,
    PoseMotionObservation,
    VAEObservation,
)
from .transitions import (
    KnownPhysics,
    LearnedAffine,
    LearnedDelayLinear,
    LearnedLinear,
)

__all__ = [
    # distributions
    "Gaussian", "Wishart",
    "combine_gaussians", "gaussian_logpdf", "gaussian_mean",
    "gaussian_mean_cov", "gaussian_prior", "near_identity_prior",
    "vague_gaussian", "wishart_mean",
    # observations
    "Frozen", "LearnedVAE", "Observation",
    "PoseMotionObservation", "VAEObservation",
    # transitions
    "KnownPhysics", "LearnedAffine", "LearnedDelayLinear", "LearnedLinear",
    # graph
    "Block", "JointModel", "LinearCoupling",
    # agent
    "Agent",
]
