"""JOPA — Bayesian inference and message passing for learning & control."""
from . import fg
from .agent import Agent
from .distributions import (
    Gaussian,
    MatrixNormal,
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
from .learn import fit_neural_linear, load_neural_linear, save_neural_linear
from .nodes import Actuator, Dynamics, NeuralLinear
from .observations import (
    Frozen,
    LearnedVAE,
    Observation,
    VAEObservation,
)
from .planning import Planner
from .transitions import KnownPhysics, LearnedLinear

__all__ = [
    "Actuator",
    # agent
    "Agent",
    # graph
    "Block",
    "Dynamics",
    # observations
    "Frozen",
    # distributions
    "Gaussian",
    "JointModel",
    # transitions
    "KnownPhysics",
    "LearnedLinear",
    "LearnedVAE",
    "LinearCoupling",
    # learned dynamics, the actuator node, planning as inference (jopa.fg)
    "MatrixNormal",
    "NeuralLinear",
    "Observation",
    "Planner",
    "VAEObservation",
    "Wishart",
    "combine_gaussians",
    "fg",
    "fit_neural_linear",
    "gaussian_logpdf",
    "gaussian_mean",
    "gaussian_mean_cov",
    "gaussian_prior",
    "load_neural_linear",
    "near_identity_prior",
    "save_neural_linear",
    "vague_gaussian",
    "wishart_mean",
]
