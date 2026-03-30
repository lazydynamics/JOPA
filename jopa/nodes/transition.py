"""ContinuousTransition node:  y ~ N(A·x + B·u, W⁻¹).

a = vec(A) (state transition), b = vec(B) (control input).
When u=None, reduces to the standard y ~ N(A·x, W⁻¹).

Structured factorisation: q(y,x) q(a) q(b) q(W).
"""
import jax.numpy as jnp
from ..distributions import Gaussian, Wishart, gaussian_mean_cov, wishart_mean


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

class CTMeta:
    """Holds the transformation  a ↦ A = f(a)."""
    __slots__ = ("f",)

    def __init__(self, f):
        self.f = f


# ---------------------------------------------------------------------------
# Helpers — vectorised contractions for linear transforms
# ---------------------------------------------------------------------------

def _va_contraction(mW, Va, dy, dx):
    r"""Σ_{i,j} mW[j,i] · Va_block[j,:,i,:]   (dx × dx).

    Accounts for the extra precision on x (or extra Δ on W) that arises
    from uncertainty in the transition parameters ``a``.
    """
    Va_blocks = Va.reshape(dy, dx, dy, dx)            # (i, k, j, l)
    return jnp.einsum("ji, jkil -> kl", mW, Va_blocks)


def _E_AMA(EaaT, Ex_xx, dy, dx):
    r"""E_a[A · M · Aᵀ]  where M = E_{yx}[x xᵀ].

    Returns a (dy × dy) matrix:
        E[A M Aᵀ]_{ij} = Σ_{k,l} M_{kl} · E[a_{i·dx+k} · a_{j·dx+l}]
    where EaaT = E[a aᵀ] = outer(ma,ma) + Va.
    """
    EaaT_blocks = EaaT.reshape(dy, dx, dy, dx)
    result = jnp.einsum("kl, ikjl -> ij", Ex_xx, EaaT_blocks)
    return 0.5 * (result + result.T)


# ---------------------------------------------------------------------------
# Precompute shared quantities for one VMP iteration
# ---------------------------------------------------------------------------

class CTCache:
    """Pre-computed quantities constant within one VMP iteration."""
    __slots__ = ("mA", "mW", "mW_inv", "Va", "ma", "EaaT", "dy", "dx",
                 "mB", "Vb", "mb", "EbbT", "du")

    def __init__(self, q_a: Gaussian, q_W: Wishart, meta: CTMeta,
                 q_b: Gaussian | None = None):
        ma, Va = gaussian_mean_cov(q_a)
        mW = wishart_mean(q_W)
        mA = meta.f(ma)
        dy, dx = mA.shape

        self.mA = mA
        self.mW = mW
        self.mW_inv = jnp.linalg.inv(mW)
        self.Va = Va
        self.ma = ma
        self.EaaT = jnp.outer(ma, ma) + Va        # E[a aᵀ]
        self.dy = dy
        self.dx = dx

        if q_b is not None:
            mb, Vb = gaussian_mean_cov(q_b)
            du = len(mb) // dy
            self.mB = mb.reshape(dy, du)
            self.Vb = Vb
            self.mb = mb
            self.EbbT = jnp.outer(mb, mb) + Vb
            self.du = du
        else:
            self.mB = None
            self.Vb = None
            self.mb = None
            self.EbbT = None
            self.du = 0


# ---------------------------------------------------------------------------
# Forward message  (:y rule, structured)
# m_x  →  CT  →  α[t]
# ---------------------------------------------------------------------------

def ct_forward(m_x: Gaussian, c: CTCache, u=None) -> Gaussian:
    """Forward message through CT (structured :y rule).

    Predictive distribution for x[t] given the message about x[t-1].
    """
    mx, Vx = gaussian_mean_cov(m_x)
    my = c.mA @ mx
    if u is not None and c.mB is not None:
        my = my + c.mB @ u
    Vy = c.mA @ Vx @ c.mA.T + c.mW_inv
    Ly = jnp.linalg.inv(Vy)
    return Gaussian(eta=Ly @ my, lam=Ly)


# ---------------------------------------------------------------------------
# Backward message  (:x rule, structured)
# m_y  →  CT  →  β[t-1]
# ---------------------------------------------------------------------------

def ct_backward(m_y: Gaussian, c: CTCache, u=None) -> Gaussian:
    """Backward message through CT (structured :x rule).

    Message about x[t-1] given the message about x[t].
    """
    Wy = m_y.lam
    # inv(Wy⁻¹ + mW⁻¹) via Woodbury
    WymW = Wy - Wy @ jnp.linalg.solve(Wy + c.mW, Wy)

    Xi = (c.mA.T @ WymW @ c.mA
          + _va_contraction(c.mW, c.Va, c.dy, c.dx))

    # z = Aᵀ WymW m_y   (avoiding explicit inversion of Wy)
    z_helper = m_y.eta - Wy @ jnp.linalg.solve(Wy + c.mW, m_y.eta)
    z = c.mA.T @ z_helper

    # Subtract control offset: Aᵀ WymW B u
    if u is not None and c.mB is not None:
        z = z - c.mA.T @ WymW @ c.mB @ u

    return Gaussian(eta=z, lam=Xi)


# ---------------------------------------------------------------------------
# Joint marginal  q(y,x)  (:y_x marginal rule)
# ---------------------------------------------------------------------------

def ct_marginal_yx(m_y: Gaussian, m_x: Gaussian, c: CTCache,
                   u=None) -> Gaussian:
    """Joint posterior q(y, x) at a CT node — dim (dy + dx)."""
    W11 = m_y.lam + c.mW
    W12 = -(c.mW @ c.mA)
    W21 = -(c.mA.T @ c.mW)
    W22 = (m_x.lam + c.mA.T @ c.mW @ c.mA
           + _va_contraction(c.mW, c.Va, c.dy, c.dx))

    W = jnp.block([[W11, W12],
                    [W21, W22]])

    xi_y = m_y.eta
    xi_x = m_x.eta
    if u is not None and c.mB is not None:
        Bu = c.mB @ u
        xi_y = xi_y + c.mW @ Bu
        xi_x = xi_x - c.mA.T @ c.mW @ Bu

    xi = jnp.concatenate([xi_y, xi_x])
    return Gaussian(eta=xi, lam=W)


# ---------------------------------------------------------------------------
# VMP message to a  (:a rule, structured)
# ---------------------------------------------------------------------------

def ct_message_a(q_yx: Gaussian, c: CTCache, u=None) -> Gaussian:
    """VMP message from one CT node to transition parameters a = vec(A).

    Uses the Kronecker identity:  Λ_a = mW^T ⊗ E[x xᵀ].
    When control input u is present, the cross-moment E[x yᵀ] is adjusted
    to E[x (y - Bu)ᵀ].
    """
    dy, dx = c.dy, c.dx
    m_yx, V_yx = gaussian_mean_cov(q_yx)
    my, mx = m_yx[:dy], m_yx[dy:]
    Vx = V_yx[dy:, dy:]
    Vyx_cross = V_yx[:dy, dy:]

    my_eff = my
    if u is not None and c.mB is not None:
        my_eff = my - c.mB @ u

    Exy = Vyx_cross.T + jnp.outer(mx, my_eff)   # E[x (y-Bu)ᵀ]
    Exx = Vx + jnp.outer(mx, mx)                 # E[x xᵀ]

    eta_a = (Exy @ c.mW).T.ravel()               # (da,)
    lam_a = jnp.kron(c.mW.T, Exx)                # (da, da)

    return Gaussian(eta=eta_a, lam=lam_a)


# ---------------------------------------------------------------------------
# VMP message to b  (:b rule, structured)
# ---------------------------------------------------------------------------

def ct_message_b(q_yx: Gaussian, c: CTCache, u: jnp.ndarray) -> Gaussian:
    """VMP message from one CT node to control parameters b = vec(B).

    Analogous to the :a rule but with known input u replacing x.
    Λ_b = mW^T ⊗ (u uᵀ).
    """
    dy = c.dy
    m_yx, V_yx = gaussian_mean_cov(q_yx)
    my, mx = m_yx[:dy], m_yx[dy:]

    # Residual after removing state dynamics
    r = my - c.mA @ mx                         # E[y - Ax]

    Eur = jnp.outer(u, r)                       # (du, dy)
    uu = jnp.outer(u, u)                        # (du, du)

    eta_b = (Eur @ c.mW).T.ravel()              # (db,)
    lam_b = jnp.kron(c.mW.T, uu)               # (db, db)

    return Gaussian(eta=eta_b, lam=lam_b)


# ---------------------------------------------------------------------------
# VMP message to u  (per-timestep action inference, known B)
# ---------------------------------------------------------------------------

def ct_message_u(q_yx: Gaussian, c: CTCache) -> Gaussian:
    """VMP message from one CT node to the per-timestep action u[t].

    Given known B (from cache), infer u from the residual y - Ax.
    eta_u = Bᵀ · mW · E[y - Ax]
    Λ_u  = Bᵀ · mW · B
    """
    dy = c.dy
    m_yx, V_yx = gaussian_mean_cov(q_yx)
    my, mx = m_yx[:dy], m_yx[dy:]

    r = my - c.mA @ mx                             # E[y - Ax]

    eta_u = c.mB.T @ c.mW @ r                      # (du,)
    lam_u = c.mB.T @ c.mW @ c.mB                   # (du, du)

    return Gaussian(eta=eta_u, lam=lam_u)


# ---------------------------------------------------------------------------
# VMP message to W  (:W rule, structured)
# ---------------------------------------------------------------------------

def ct_message_W(q_yx: Gaussian, c: CTCache, u=None) -> jnp.ndarray:
    """VMP message from one CT node to precision W.

    Returns Δ (inv_scale contribution).  Full message: Wishart(dy+2, Δ).
    """
    dy, dx = c.dy, c.dx
    m_yx, V_yx = gaussian_mean_cov(q_yx)
    my, mx = m_yx[:dy], m_yx[dy:]
    Vy, Vx = V_yx[:dy, :dy], V_yx[dy:, dy:]
    Vyx_cross = V_yx[:dy, dy:]

    G1 = jnp.outer(my, my) + Vy
    G2 = (jnp.outer(my, mx) + Vyx_cross) @ c.mA.T

    Ex_xx = Vx + jnp.outer(mx, mx)
    E_AxxA = _E_AMA(c.EaaT, Ex_xx, dy, dx)

    delta = G1 - (G2 + G2.T) + E_AxxA

    if u is not None and c.mB is not None:
        Bu = c.mB @ u
        r = my - c.mA @ mx
        # Cross terms: -r(Bu)ᵀ - (Bu)rᵀ
        delta = delta - jnp.outer(r, Bu) - jnp.outer(Bu, r)
        # E[B(uuᵀ)Bᵀ] — includes both mean and variance of B
        uu = jnp.outer(u, u)
        delta = delta + _E_AMA(c.EbbT, uu, dy, c.du)

    return 0.5 * (delta + delta.T)               # enforce symmetry
