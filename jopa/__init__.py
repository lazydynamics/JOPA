"""JOPA — Joint Observation-Planning Architecture. Message passing in JAX."""
from .distributions import Gaussian, Wishart, combine_gaussians, gaussian_mean, gaussian_mean_cov, gaussian_prior, wishart_mean
from .inference import infer, plan, InferenceResult, PlanResult
from .em import variational_em, EMResult
from .nodes.transition import CTMeta
from .nodes.observation import vae_observe, vae_predict
from . import data
from . import nn
