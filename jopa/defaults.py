"""Default priors and initial posteriors for the CT-factor VMP."""

PRIOR_W_DF: float = 4.0       # Wishart prior df on the dynamics noise precision
PRIOR_A_COV: float = 1.0      # vec(A) prior variance
PRIOR_B_COV: float = 1.0      # vec(B) prior variance
INIT_A_COV: float = 100.0     # initial q(A) variance — wide so data dominates
INIT_B_COV: float = 100.0     # initial q(B) variance
