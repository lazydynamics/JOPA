"""Compatibility re-exports — see `jopa.config` for the documented values."""
from .config import (
    INIT_A_COV,
    INIT_B_COV,
    PRIOR_A_COV,
    PRIOR_B_COV,
    PRIOR_W_DF,
)

__all__ = [
    "INIT_A_COV", "INIT_B_COV", "PRIOR_A_COV", "PRIOR_B_COV", "PRIOR_W_DF",
]
