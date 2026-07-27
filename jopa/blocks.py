"""Compatibility re-exports — see `jopa.observations`, `jopa.transitions`,
`jopa.graph`. The underscore names are imported by `jopa.agent` and by
examples; an import linter must not strip them."""
from .graph import (  # noqa: F401
    Block,
    JointModel,
    LinearCoupling,
    _augment_prior_u,
    _default_action_prior,
    _identity_meta,
    _known_plan_cache,
)
from .observations import (  # noqa: F401
    Frozen,
    LearnedVAE,
    Observation,
    PoseMotionObservation,
    VAEObservation,
    _as_observation,
    _kl_diag_vs_full,
    _vae_m_step_loss,
)
from .transitions import (  # noqa: F401
    KnownPhysics,
    LearnedAffine,
    LearnedDelayLinear,
    LearnedLinear,
)

__all__ = [
    "Block", "Frozen", "JointModel", "KnownPhysics",
    "LearnedAffine", "LearnedDelayLinear", "LearnedLinear",
    "LearnedVAE", "LinearCoupling", "Observation",
    "PoseMotionObservation", "VAEObservation",
]
