"""Estimator-independent limits on target robustness.

Let the admissible corruption set be ``A(k, 1) = {a : <=k nonzero blocks,
||a_i||_2 <= 1}``.  Its convex hull is the unit ball of

    nu_k(a) = max( max_i ||a_i||_2 ,  (1/k) sum_i ||a_i||_2 ),

because the support function of ``A(k,1)`` is the sum of the ``k`` largest block
norms, and that is dual to ``nu_k``.  Consequently, for any influence operator,

    R_k(M) = sup { ||M a|| : nu_k(a) <= 1 }

is an operator norm, and choosing an unbiased ``M`` is choosing a norm-minimal
*extension* of the map ``A xi -> G xi`` from ``range(A)`` to all of ``R^{3n}``.
The fundamental limit is therefore the norm of that map on the subspace,

    kappa^(k) = sup { ||G xi|| : nu_k(A xi) <= 1 } ,

which no unbiased estimator can beat.  By the Hahn--Banach theorem the limit is
*attained* whenever the downstream loss is a maximum of linear functionals --- a
scalar directional loss, or an ell-infinity loss in an anatomical frame ---
because a scalar functional always extends without increasing its norm, and the
extension can be done coordinatewise.  For a Euclidean loss on R^3 no such
guarantee exists, and the limit is attained for small ``k`` but is strict in
general.
"""
from __future__ import annotations

import numpy as np

from .risk import fibonacci_sphere

__all__ = ["nu_k", "scalar_risk", "scalar_design", "fundamental_limit",
           "scalar_limit"]


def nu_k(a: np.ndarray, k: int) -> float:
    """The domain norm whose unit ball is ``conv A(k, 1)``."""
    b = np.linalg.norm(np.asarray(a, float).reshape(-1, 3), axis=1)
    return float(max(b.max(), b.sum() / k))


def scalar_risk(m: np.ndarray, rho, k: int) -> float:
    """Worst-case corruption of a *scalar* target functional.

    With a scalar loss the influence operator is a row vector and the risk is
    in closed form -- the maximisation over the unit sphere in the vector case
    disappears entirely.
    """
    b = np.linalg.norm(np.asarray(m, float).reshape(-1, 3), axis=1)
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), b.shape)
    z = np.sort(rho * b)[::-1]
    return float(z[:int(min(k, z.size))].sum())


def scalar_design(A, g, rho, k: int = 1, Sigma=None, eta: float | None = None,
                  solver: str = "CLARABEL"):
    """Target-minimax design for a scalar functional ``g = c' G_x``.

    Solves ``min_m top-k_i rho_i ||m_i||`` subject to ``m A = g`` and, if
    ``Sigma`` and ``eta`` are given, a clean-efficiency cap.  No cutting planes
    are needed: the objective is exact and second-order cone representable.
    """
    import cvxpy as cp

    A = np.asarray(A, float)
    g = np.asarray(g, float).reshape(1, 6)
    n = A.shape[0] // 3
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,))
    m = cp.Variable((1, 3 * n))
    nb = cp.hstack([rho[i] * cp.norm(m[0, 3 * i:3 * i + 3], 2)
                    for i in range(n)])
    cons = [m @ A == g]
    if Sigma is not None and eta is not None:
        Sigma = np.asarray(Sigma, float)
        Si = np.linalg.inv(Sigma)
        K0 = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
        m0 = np.asarray(g @ K0, float).reshape(1, -1)
        V0 = float((m0 @ Sigma @ m0.T).ravel()[0])
        L = np.linalg.cholesky(Sigma + 1e-15 * np.eye(3 * n))
        cons.append(cp.sum_squares(m @ L) <= (1.0 + eta) * V0)
    prob = cp.Problem(cp.Minimize(cp.sum_largest(nb, int(min(k, n)))), cons)
    prob.solve(solver=solver)
    return np.asarray(m.value, float).ravel(), float(prob.value), prob.status


def _limit_for_direction(A, Gt, c, k, solver="CLARABEL"):
    import cvxpy as cp
    n = A.shape[0] // 3
    xi = cp.Variable(6)
    blk = cp.hstack([cp.norm(A[3 * i:3 * i + 3, :] @ xi, 2) for i in range(n)])
    prob = cp.Problem(cp.Maximize((np.asarray(c, float) @ Gt) @ xi),
                      [cp.max(blk) <= 1, cp.sum(blk) <= k])
    prob.solve(solver=solver)
    return float(prob.value), np.asarray(xi.value, float).ravel()


def scalar_limit(A, g, k: int = 1, solver: str = "CLARABEL") -> float:
    """``kappa^(k)`` for a scalar functional -- exact, and attained."""
    import cvxpy as cp
    A = np.asarray(A, float)
    g = np.asarray(g, float).ravel()
    n = A.shape[0] // 3
    xi = cp.Variable(6)
    blk = cp.hstack([cp.norm(A[3 * i:3 * i + 3, :] @ xi, 2) for i in range(n)])
    prob = cp.Problem(cp.Maximize(g @ xi),
                      [cp.max(blk) <= 1, cp.sum(blk) <= k])
    prob.solve(solver=solver)
    return float(prob.value)


def fundamental_limit(A, Gt, k: int = 1, loss: str = "2", n_start: int = 24,
                      n_iter: int = 25) -> float:
    """``kappa^(k) = sup{ ||G xi|| : nu_k(A xi) <= 1 }`` for a vector loss.

    For ``loss='inf'`` the maximisation is over the six signed axis directions
    and is exact.  For ``loss='2'`` it is a convex maximisation, solved by
    alternating the exact inner SOCP with the exact direction update
    ``c <- G xi / ||G xi||``; each step is non-decreasing and multi-start makes
    the value reliable.
    """
    A, Gt = np.asarray(A, float), np.asarray(Gt, float)
    if loss == "inf":
        return max(_limit_for_direction(A, Gt, c, k)[0]
                   for c in np.vstack([np.eye(Gt.shape[0]), -np.eye(Gt.shape[0])]))
    best = 0.0
    for c0 in fibonacci_sphere(n_start):
        c = c0.copy()
        for _ in range(n_iter):
            _, xi = _limit_for_direction(A, Gt, c, k)
            v = Gt @ xi
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                break
            cn = v / nv
            if np.linalg.norm(cn - c) < 1e-10:
                c = cn
                break
            c = cn
        _, xi = _limit_for_direction(A, Gt, c, k)
        best = max(best, float(np.linalg.norm(Gt @ xi)))
    return best
