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
from .diagnostics import (
    BeliefGap, FilterStepDiagnostics, TransitionMetrics,
    belief_gap, block_precision, covariance_trace, filter_with_diagnostics,
    gaussian_kl, transition_metrics, transition_residuals,
)

__all__ = [
    "Gaussian", "Wishart",
    "combine_gaussians", "gaussian_mean", "gaussian_mean_cov",
    "gaussian_prior", "near_identity_prior", "wishart_mean", "vague_gaussian",
    "Block", "JointModel",
    "Observation", "Frozen", "LearnedVAE",
    "LearnedLinear", "LearnedAffine", "KnownPhysics",
    "LinearCoupling",
    "BeliefGap", "FilterStepDiagnostics", "TransitionMetrics",
    "belief_gap", "block_precision", "covariance_trace", "filter_with_diagnostics",
    "gaussian_kl", "transition_metrics", "transition_residuals",
]
