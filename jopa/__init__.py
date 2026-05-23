"""JOPA — Joint Observation-Planning Architecture. Message passing in JAX."""
from .distributions import (
    Gaussian, Wishart,
    combine_gaussians, gaussian_mean, gaussian_mean_cov, gaussian_prior,
    near_identity_prior, wishart_mean,
)
from .inference import infer, plan, InferenceResult, PlanResult
from .em import variational_em, EMResult, Trajectory
from .nodes.transition import CTMeta
from .nodes.observation import vae_predict
from .nn.vae import VAEAdapter
from . import data
from . import nn

__all__ = [
    "Gaussian", "Wishart",
    "combine_gaussians", "gaussian_mean", "gaussian_mean_cov",
    "gaussian_prior", "near_identity_prior", "wishart_mean",
    "infer", "plan", "InferenceResult", "PlanResult",
    "variational_em", "EMResult", "Trajectory",
    "CTMeta",
    "vae_predict",
    "VAEAdapter",
]
