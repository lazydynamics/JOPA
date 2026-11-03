"""Shared constants: numerical guards and the conjugate-prior defaults.

The guards keep linear algebra and gradients finite and must agree wherever
they appear. The defaults stay keyword arguments of the classes that use them;
only the literal lives here.
"""
# ---------------------------------------------------------------------------
# Numerical hygiene
# ---------------------------------------------------------------------------

# A variance/second-moment at or below this is numerically zero: used before a
# sqrt so the gradient stays finite, and to detect a degenerate R² denominator.
EPS_VARIANCE = 1e-12


# Encoder log-standard-deviation clip. exp(-6) ≈ 2.5e-3 bounds the reported
# precision (a sensor claiming more is over-confident and destabilises
# filtering); exp(2) ≈ 7.4 bounds it below the latent working scale.
LOG_STD_CLIP = (-6.0, 2.0)

# Bernoulli probability clip before log() in a reconstruction cross-entropy.
PROB_CLIP = (1e-6, 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# Conjugate priors and initial posteriors for the CT-factor VMP
# ---------------------------------------------------------------------------

PRIOR_A_COV: float = 1.0      # vec(A) prior variance
PRIOR_B_COV: float = 1.0      # vec(B) prior variance
INIT_A_COV: float = 100.0     # initial q(A) variance — wide so data dominates
INIT_B_COV: float = 100.0     # initial q(B) variance

# A Wishart on a d×d precision is proper for df > d − 1, but E[W⁻¹] — the
# predictive noise covariance — is finite only for df > d + 1. `d + this` is the
# smallest prior with both, and is the default wherever a df is not given.
WISHART_DF_MARGIN = 2


# ---------------------------------------------------------------------------
# Transition learning
# ---------------------------------------------------------------------------

# Coordinate-ascent VMP sweeps. The updates are conjugate closed forms, so these
# are "enough to converge on the trajectory lengths used here" rather than tuned.
VMP_ITERATIONS = 50               # LearnedLinear (chain smoothing in the loop)
