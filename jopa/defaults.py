"""Single source of truth for Bayesian prior / initialisation defaults.

These constants are referenced by :mod:`jopa.em` and :mod:`jopa.inference`
so that the two entry points cannot drift apart.

Convention
----------
``PRIOR_*``  — variance of the (proper) prior used when accumulating VMP
              messages on transition / control matrices.
``INIT_*``   — variance of the initial posterior ``q`` before message
              passing starts; intentionally wide so the prior + data drive
              the result.
"""

# Wishart prior on the dynamics-noise precision W. df > d - 1 required for
# a proper density; 4.0 is vague for d ≤ 4 (the latent dim used in demos).
PRIOR_W_DF: float = 4.0

# Prior on vec(A): A is a d×d transition matrix, entries iid N(0, PRIOR_A_COV·I).
PRIOR_A_COV: float = 1.0

# Prior on vec(B): B is a d×du control matrix, entries iid N(0, PRIOR_B_COV·I).
PRIOR_B_COV: float = 1.0

# Initial posterior covariance for vec(A) and vec(B): wide → effectively
# uninformative starting belief; the prior + likelihood take over.
INIT_A_COV: float = 100.0
INIT_B_COV: float = 100.0
