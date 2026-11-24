"""Rigid paired-point registration estimators.

Baselines
---------
``kabsch``          unweighted orthogonal Procrustes (ordinary least squares)
``weighted_kabsch`` scalar-weighted Procrustes
``huber_irls``      residual-robust M-estimator
``ransac_register`` gross-outlier rejection
``gnc_tls``         graduated non-convexity on a truncated least-squares cost
                    (the robust core used by TEASER-style pipelines)

Target-Minimax Registration
---------------------------
``gls_influence``   the Gauss-Markov target-influence operator
``tmr_influence``   the target-minimax influence operator (convex program)
``tmr_weights``     the same objective restricted to the weighted-LS family
``tmr_register_se3``  iteratively relinearised SE(3) solver
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .geometry import (RigidTransform, project_to_so3, stack_jacobians,
                       target_jacobian, expm_so3)
from .risk import sparse_risk_exact, blocks_of

__all__ = [
    "kabsch", "weighted_kabsch", "gls_influence", "tmr_influence",
    "tmr_weights", "huber_irls", "ransac_register", "gnc_tls",
    "influence_to_gain", "tmr_register_se3", "complete_influence_to_K",
    "wls_influence", "TMRResult",
]


# --------------------------------------------------------------------------- #
#  Closed-form / classical estimators
# --------------------------------------------------------------------------- #
def weighted_kabsch(P: np.ndarray, Q: np.ndarray, w=None) -> RigidTransform:
    """Weighted orthogonal Procrustes: min_{R,t} sum_i w_i ||R p_i + t - q_i||^2."""
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    n = P.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, float).ravel()
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.ones(n)
    w = w / w.sum()
    pbar = w @ P
    qbar = w @ Q
    H = (P - pbar).T @ (w[:, None] * (Q - qbar))
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return RigidTransform(R, qbar - R @ pbar)


def kabsch(P: np.ndarray, Q: np.ndarray) -> RigidTransform:
    """Ordinary (unweighted) least-squares rigid registration."""
    return weighted_kabsch(P, Q, None)


def huber_irls(P, Q, delta=None, n_iter: int = 20,
               tol: float = 1e-9) -> RigidTransform:
    """Huber M-estimator by iteratively reweighted weighted-Procrustes."""
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    T = kabsch(P, Q)
    for _ in range(n_iter):
        r = np.linalg.norm(P @ T.R.T + T.t - Q, axis=1)
        d = delta
        if d is None:
            med = np.median(r)
            mad = np.median(np.abs(r - med))
            d = max(1.345 * 1.4826 * mad, 1e-9)
        w = np.where(r <= d, 1.0, d / np.maximum(r, 1e-12))
        Tn = weighted_kabsch(P, Q, w)
        moved = max(float(np.max(np.abs(Tn.R - T.R))),
                    float(np.max(np.abs(Tn.t - T.t))) / (1.0 + float(np.max(np.abs(T.t)))))
        T = Tn
        if moved < tol:
            break
    return T


def ransac_register(P, Q, thresh, n_hypo: int = 200, seed: int = 0,
                    min_set: int = 3) -> RigidTransform:
    """RANSAC over minimal 3-point sets with a final inlier refit."""
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    best_in, best_T = None, kabsch(P, Q)
    combos = list(itertools.combinations(range(n), min_set))
    if len(combos) > n_hypo:
        idx = rng.choice(len(combos), size=n_hypo, replace=False)
        combos = [combos[i] for i in idx]
    for S in combos:
        S = list(S)
        T = kabsch(P[S], Q[S])
        r = np.linalg.norm(P @ T.R.T + T.t - Q, axis=1)
        inl = r <= thresh
        if inl.sum() < 3:
            continue
        if best_in is None or inl.sum() > best_in.sum() or (
                inl.sum() == best_in.sum()
                and r[inl].sum() < np.linalg.norm(
                    P @ best_T.R.T + best_T.t - Q, axis=1)[inl].sum()):
            best_in, best_T = inl, kabsch(P[inl], Q[inl])
    if best_in is None:
        return kabsch(P, Q)
    return best_T


def gnc_tls(P, Q, barc2, mu_factor: float = 1.4, n_iter: int = 100
            ) -> RigidTransform:
    """Graduated non-convexity with a truncated-least-squares surrogate.

    ``barc2`` is the squared inlier threshold.  This is the robust rotation /
    translation core used by TEASER-style certifiable registration.
    """
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    n = P.shape[0]
    T = kabsch(P, Q)
    r2 = np.sum((P @ T.R.T + T.t - Q) ** 2, axis=1)
    mu = max(1e-9, barc2 / max(2.0 * np.max(r2) - barc2, 1e-9))
    w = np.ones(n)
    for _ in range(n_iter):
        T = weighted_kabsch(P, Q, w)
        r2 = np.sum((P @ T.R.T + T.t - Q) ** 2, axis=1)
        th_hi = (mu + 1.0) / mu * barc2
        th_lo = mu / (mu + 1.0) * barc2
        w = np.where(r2 >= th_hi, 0.0,
                     np.where(r2 <= th_lo, 1.0,
                              np.sqrt(barc2 * mu * (mu + 1.0)
                                      / np.maximum(r2, 1e-12)) - mu))
        w = np.clip(w, 0.0, 1.0)
        if mu > 1e6:
            break
        mu *= mu_factor
    if w.sum() < 1e-9:
        return kabsch(P, Q)
    return weighted_kabsch(P, Q, w)


# --------------------------------------------------------------------------- #
#  Linear influence operators
# --------------------------------------------------------------------------- #
def gls_influence(A: np.ndarray, G: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Gauss-Markov target-influence operator ``M = G (A' S^-1 A)^-1 A' S^-1``."""
    A = np.asarray(A, float)
    Si = np.linalg.inv(np.asarray(Sigma, float))
    K = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
    return np.asarray(G, float) @ K


def wls_influence(A: np.ndarray, G: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Influence operator of scalar-weighted least squares with weights ``w``."""
    A = np.asarray(A, float)
    n = A.shape[0] // 3
    W = np.repeat(np.asarray(w, float).ravel(), 3)
    AtW = A.T * W[None, :]
    K = np.linalg.solve(AtW @ A, AtW)
    return np.asarray(G, float) @ K


def complete_influence_to_K(M: np.ndarray, G: np.ndarray,
                            K0: np.ndarray) -> np.ndarray:
    """Lift a target-influence operator ``M`` to a full unbiased estimator ``K``.

    ``K = G^+ M + (I - G^+ G) K0`` satisfies ``K A = I`` and ``G K = M`` for any
    unbiased ``K0``; taking ``K0`` to be GLS makes the target-irrelevant part of
    the pose estimate minimum variance.
    """
    G = np.asarray(G, float)
    Gp = np.linalg.pinv(G)
    return Gp @ np.asarray(M, float) + (np.eye(G.shape[1]) - Gp @ G) @ K0


def influence_to_gain(M: np.ndarray, rho, k: int) -> float:
    return sparse_risk_exact(M, rho, k)


# --------------------------------------------------------------------------- #
#  Target-Minimax Registration -- convex design of the influence operator
# --------------------------------------------------------------------------- #
@dataclass
class TMRResult:
    M: np.ndarray
    risk: float          # exact R_k of the returned M (upper bound on optimum)
    lower_bound: float   # value of the last relaxation (lower bound)
    gap: float
    n_cuts: int
    variance: float
    status: str


def tmr_influence(A, G, Sigma, rho, k: int = 1, eta: float = 0.10,
                  n_init_dirs: int = 24, max_rounds: int = 40,
                  tol: float = 1e-3, n_cut: int = 4,
                  solver: str = "CLARABEL",
                  verbose: bool = False, V0: float | None = None
                  ) -> TMRResult:
    """Solve the target-minimax influence design problem

        minimise_M   R_k(M) = max_{||u||=1} topk_i rho_i ||M_i' u||
        subject to   M A = G,   tr(M Sigma M') <= (1 + eta) V0,

    where ``V0`` is the Gauss-Markov clean target variance.  The semi-infinite
    program is solved by a cutting-plane (exchange) method: each round solves an
    SOCP over a finite direction set ``U`` (a lower bound) and then adds the
    exact worst-case direction of the incumbent (an upper bound).
    """
    import cvxpy as cp

    A = np.asarray(A, float)
    G = np.asarray(G, float)
    Sigma = np.asarray(Sigma, float)
    n = A.shape[0] // 3
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,)).copy()
    k = int(min(k, n))

    M_gls = gls_influence(A, G, Sigma)
    if V0 is None:
        V0 = float(np.trace(M_gls @ Sigma @ M_gls.T))
    vmax = (1.0 + eta) * V0

    L = np.linalg.cholesky(Sigma + 1e-15 * np.eye(Sigma.shape[0]))

    M = cp.Variable((3, 3 * n))
    s = cp.Variable(nonneg=True)
    blocks = [M[:, 3 * i:3 * i + 3] for i in range(n)]
    base = [M @ A == G, cp.sum_squares(M @ L) <= vmax]

    from .risk import fibonacci_sphere
    U = list(fibonacci_sphere(n_init_dirs))

    best = TMRResult(M_gls, sparse_risk_exact(M_gls, rho, k), -np.inf, np.inf,
                     0, float(np.trace(M_gls @ Sigma @ M_gls.T)), "init")
    for rd in range(max_rounds):
        cons = list(base)
        for u in U:
            gains = cp.hstack([rho[i] * cp.norm(blocks[i].T @ u, 2)
                               for i in range(n)])
            cons.append(cp.sum_largest(gains, k) <= s)
        prob = cp.Problem(cp.Minimize(s), cons)
        try:
            prob.solve(solver=solver)
        except Exception:
            prob.solve(solver="SCS")
        if M.value is None:
            break
        Mv = np.asarray(M.value, float)
        lb = float(prob.value)
        ub, new_cuts = sparse_risk_exact(Mv, rho, k, return_top=n_cut)
        var = float(np.trace(Mv @ Sigma @ Mv.T))
        if ub < best.risk:
            best = TMRResult(Mv, ub, lb, ub - lb, len(U), var, prob.status)
        else:
            best.lower_bound = max(best.lower_bound, lb)
            best.gap = best.risk - best.lower_bound
            best.n_cuts = len(U)
        if verbose:
            print(f"  round {rd:2d}  lb={lb:.6f}  ub={ub:.6f}  "
                  f"gap={ub - lb:.2e}  |U|={len(U)}")
        if ub - lb <= tol * max(1.0, abs(ub)):
            break
        U.extend(new_cuts)
    best.gap = max(best.risk - best.lower_bound, 0.0)
    return best


def tmr_influence_sdp(A, G, Sigma, rho, eta: float = 0.10,
                      solver: str = "SCS", V0: float | None = None):
    """Direct SDP formulation of the ``k = 1`` problem (cross-check).

    ``min_M max_i rho_i sigma_max(M_i)`` s.t. ``M A = G`` and the variance cap.
    """
    import cvxpy as cp

    A = np.asarray(A, float)
    G = np.asarray(G, float)
    Sigma = np.asarray(Sigma, float)
    n = A.shape[0] // 3
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,))
    M_gls = gls_influence(A, G, Sigma)
    if V0 is None:
        V0 = float(np.trace(M_gls @ Sigma @ M_gls.T))
    L = np.linalg.cholesky(Sigma + 1e-15 * np.eye(Sigma.shape[0]))
    M = cp.Variable((3, 3 * n))
    s = cp.Variable(nonneg=True)
    cons = [M @ A == G, cp.sum_squares(M @ L) <= (1.0 + eta) * V0]
    for i in range(n):
        cons.append(rho[i] * cp.sigma_max(M[:, 3 * i:3 * i + 3]) <= s)
    prob = cp.Problem(cp.Minimize(s), cons)
    prob.solve(solver=solver)
    return np.asarray(M.value, float), float(prob.value), prob.status


def tmr_dual_k1(A, G, rho, solver: str = "SCS"):
    """Dual of the variance-free ``k = 1`` problem.

        max_L  <L, G>   s.t.  sum_i ||L A_i'||_* / rho_i <= 1.
    """
    import cvxpy as cp

    A = np.asarray(A, float)
    G = np.asarray(G, float)
    n = A.shape[0] // 3
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,))
    Lam = cp.Variable((3, 6))
    expr = sum(cp.normNuc(Lam @ A[3 * i:3 * i + 3, :].T) / rho[i]
               for i in range(n))
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(Lam, G))), [expr <= 1])
    prob.solve(solver=solver)
    return np.asarray(Lam.value, float), float(prob.value), prob.status


def tmr_weights(A, G, Sigma, rho, k: int = 1, eta: float = 0.10,
                n_restart: int = 12, seed: int = 0):
    """Best *scalar-weighted least squares* estimator for the same objective.

    Ablation used to show that target-minimax design is not a reweighting:
    the search is over ``w > 0`` with ``M = G (A'WA)^-1 A'W``.
    """
    from scipy.optimize import minimize

    A = np.asarray(A, float)
    G = np.asarray(G, float)
    Sigma = np.asarray(Sigma, float)
    n = A.shape[0] // 3
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,))
    M_gls = gls_influence(A, G, Sigma)
    V0 = float(np.trace(M_gls @ Sigma @ M_gls.T))
    vmax = (1.0 + eta) * V0
    rng = np.random.default_rng(seed)

    def obj(z):
        w = np.exp(np.clip(z, -12, 12))
        try:
            M = wls_influence(A, G, w)
        except np.linalg.LinAlgError:
            return 1e6
        v = float(np.trace(M @ Sigma @ M.T))
        pen = max(0.0, v - vmax) / max(vmax, 1e-12)
        return sparse_risk_exact(M, rho, k, n_start=32, n_iter=60) * (1 + 50 * pen)

    best_val, best_w = np.inf, np.ones(n)
    starts = [np.zeros(n)] + [0.7 * rng.standard_normal(n)
                              for _ in range(n_restart - 1)]
    for z0 in starts:
        res = minimize(obj, z0, method="Nelder-Mead",
                       options=dict(maxiter=1200, xatol=1e-7, fatol=1e-9))
        if res.fun < best_val:
            best_val, best_w = float(res.fun), np.exp(np.clip(res.x, -12, 12))
    M = wls_influence(A, G, best_w)
    return M, best_w / best_w.max(), sparse_risk_exact(M, rho, k)


# --------------------------------------------------------------------------- #
#  Nonlinear SE(3) solver
# --------------------------------------------------------------------------- #
def tmr_register_se3(P, Q, x, Sigma, rho, k: int = 1, eta: float = 0.10,
                     n_iter: int = 25, tol: float = 1e-12, **kw):
    """Iteratively relinearised target-minimax registration.

    At each iteration the Jacobians ``A`` (fiducials) and ``G`` (target) are
    rebuilt at the current pose, the influence operator is redesigned, lifted to
    a full unbiased estimator ``K``, and the pose is updated on SE(3).
    """
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    x = np.asarray(x, float).reshape(3)
    T = kabsch(P, Q)
    info = {}
    for it in range(n_iter):
        A = stack_jacobians(T.R, P)
        G = target_jacobian(T.R, x)
        res = tmr_influence(A, G, Sigma, rho, k=k, eta=eta, **kw)
        Si = np.linalg.inv(Sigma)
        K0 = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
        K = complete_influence_to_K(res.M, G, K0)
        b = (Q - (P @ T.R.T + T.t)).reshape(-1)
        xi = K @ b
        T = RigidTransform(expm_so3(xi[:3]) @ T.R, T.t + xi[3:])
        info = dict(iters=it + 1, step=float(np.linalg.norm(xi)),
                    risk=res.risk, gap=res.gap, variance=res.variance)
        if np.linalg.norm(xi) < tol * (1 + np.linalg.norm(T.t)):
            break
    return T, info


def transport_influence(M0: np.ndarray, R0: np.ndarray) -> np.ndarray:
    """Transport an influence operator designed at ``R0 = I`` to pose ``R0``.

    By the equivariance of the design problem (Proposition: frame invariance),
    if ``M0`` is optimal for ``(A(I), G(I), Sigma)`` and ``Sigma`` commutes with
    ``I_n (x) R0`` -- in particular for isotropic per-fiducial noise -- then
    ``R0 M0 (I_n (x) R0')`` is optimal for ``(A(R0), G(R0), Sigma)``.
    """
    M0 = np.asarray(M0, float)
    n = M0.shape[1] // 3
    return np.asarray(R0, float) @ M0 @ np.kron(np.eye(n), np.asarray(R0, float).T)


def make_fixed_design_estimator(P, x, M0, Sigma, n_iter: int = 8,
                                tol: float = 1e-12):
    """A deployable estimator whose influence design is fixed pre-operatively.

    ``M0`` is designed once from the plan (fiducial geometry, target, noise and
    threat model) and transported to the current pose at each relinearisation.
    """
    P = np.atleast_2d(np.asarray(P, float))
    x = np.asarray(x, float).reshape(3)
    Sigma = np.asarray(Sigma, float)
    Si = np.linalg.inv(Sigma)

    def est(Pp, Qp):
        T = kabsch(Pp, Qp)
        for _ in range(n_iter):
            A = stack_jacobians(T.R, Pp)
            G = target_jacobian(T.R, x)
            M = transport_influence(M0, T.R)
            K0 = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
            K = complete_influence_to_K(M, G, K0)
            b = (Qp - (Pp @ T.R.T + T.t)).reshape(-1)
            xi = K @ b
            T = RigidTransform(expm_so3(xi[:3]) @ T.R, T.t + xi[3:])
            if np.linalg.norm(xi) < tol:
                break
        return T
    return est


def tmr_influence_multitarget(A, G_list, Sigma, rho, k: int = 1, eta: float = 0.10,
                              n_init_dirs: int = 12, max_rounds: int = 40,
                              tol: float = 1e-3, n_cut: int = 3,
                              solver: str = "CLARABEL"):
    """Target-minimax design for a *set* of downstream targets.

    Given targets ``X = {x_1, ..., x_m}`` supplied as ``G_list = [G_{x_j}]`` --- a
    sampled trajectory, a sampled workspace region, or any finite collection of
    task points --- the loss is the largest displacement over the set,

        R_k(K; X) = max_j max_{a in A(k,rho)} || G_j K a || ,

    and the clean-efficiency constraint caps the largest variance over the set.
    The estimator ``K`` is the variable rather than the reduced operator ``M``,
    because the stacked Jacobian of a set of targets need not have full rank (a
    straight trajectory, for instance, gives rank 5).

    A single target recovers :func:`tmr_influence`; a sampled path recovers the
    trajectory functional.
    """
    import cvxpy as cp
    from .risk import fibonacci_sphere

    A = np.asarray(A, float)
    Gs = [np.asarray(g, float) for g in G_list]
    Sigma = np.asarray(Sigma, float)
    n = A.shape[0] // 3
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,)).copy()
    k = int(min(k, n))

    Si = np.linalg.inv(Sigma)
    K_gls = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
    V0 = max(float(np.trace(G @ K_gls @ Sigma @ K_gls.T @ G.T)) for G in Gs)
    L = np.linalg.cholesky(Sigma + 1e-15 * np.eye(Sigma.shape[0]))

    K = cp.Variable((6, 3 * n))
    s = cp.Variable(nonneg=True)
    base = [K @ A == np.eye(6)]
    for G in Gs:
        base.append(cp.sum_squares(G @ K @ L) <= (1.0 + eta) * V0)

    U0 = list(fibonacci_sphere(n_init_dirs))
    cuts = [(j, u) for j in range(len(Gs)) for u in U0[:max(2, n_init_dirs // 3)]]

    best_K, best_risk, best_lb = K_gls, np.inf, -np.inf
    for _ in range(max_rounds):
        cons = list(base)
        for j, u in cuts:
            Mj = Gs[j] @ K
            gains = cp.hstack([rho[i] * cp.norm(Mj[:, 3 * i:3 * i + 3].T @ u, 2)
                               for i in range(n)])
            cons.append(cp.sum_largest(gains, k) <= s)
        prob = cp.Problem(cp.Minimize(s), cons)
        try:
            prob.solve(solver=solver)
        except Exception:
            prob.solve(solver="SCS")
        if K.value is None:
            break
        Kv = np.asarray(K.value, float)
        # reject a numerically failed solve: unbiasedness must hold exactly
        if np.max(np.abs(Kv @ A - np.eye(6))) > 1e-6:
            break
        lb = float(prob.value)
        vals = []
        for j, G in enumerate(Gs):
            v, us = sparse_risk_exact(G @ Kv, rho, k, return_top=2)
            vals.append((v, j, us))
        ub = max(v for v, _, _ in vals)
        if ub < best_risk:
            best_K, best_risk = Kv, ub
        best_lb = max(best_lb, lb)
        if ub - lb <= tol * max(1.0, abs(ub)):
            break
        for v, j, us in sorted(vals, reverse=True)[:n_cut]:
            for u in us[:1]:
                cuts.append((j, u))
    var = max(float(np.trace(G @ best_K @ Sigma @ best_K.T @ G.T)) for G in Gs)
    return dict(K=best_K, risk=best_risk, lower_bound=best_lb,
                gap=max(best_risk - best_lb, 0.0), variance=var, V0=V0)


def make_K_estimator(P, x_ref, K0_design, Sigma, n_iter: int = 8):
    """Deployable estimator from a fixed ``K`` design, transported by pose.

    ``K`` transports as ``blkdiag(R0,R0) K0 (I_n (x) R0')`` under the same
    equivariance used for ``M``.
    """
    P = np.atleast_2d(np.asarray(P, float))
    K0_design = np.asarray(K0_design, float)
    n = P.shape[0]

    def est(Pp, Qp):
        T = kabsch(Pp, Qp)
        for _ in range(n_iter):
            B = np.kron(np.eye(2), T.R)
            K = B @ K0_design @ np.kron(np.eye(n), T.R.T)
            b = (Qp - (Pp @ T.R.T + T.t)).reshape(-1)
            xi = K @ b
            T = RigidTransform(expm_so3(xi[:3]) @ T.R, T.t + xi[3:])
            if np.linalg.norm(xi) < 1e-12:
                break
        return T
    return est


def average_influence(A, G, Sigma, rho, k: int = 1, eta: float = 0.10,
                      solver: str = "CLARABEL", V0: float | None = None):
    """Average-case counterpart of :func:`tmr_influence`.

    Minimises ``E ||M a||^2`` for a uniformly random k-sparse corruption --- a
    weighted squared Frobenius norm, by the average-case identity --- subject to
    the same unbiasedness and clean-efficiency constraints.  This is the design
    an average-error-driven method converges to given unlimited data, and is
    used as the population-optimal stand-in for learned correspondence
    weighting.
    """
    import cvxpy as cp

    A = np.asarray(A, float)
    G = np.asarray(G, float)
    Sigma = np.asarray(Sigma, float)
    n = A.shape[0] // 3
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,))
    M_gls = gls_influence(A, G, Sigma)
    if V0 is None:
        V0 = float(np.trace(M_gls @ Sigma @ M_gls.T))
    L = np.linalg.cholesky(Sigma + 1e-15 * np.eye(Sigma.shape[0]))

    M = cp.Variable((3, 3 * n))
    obj = cp.sum([rho[i] ** 2 * cp.sum_squares(M[:, 3 * i:3 * i + 3])
                  for i in range(n)])
    cons = [M @ A == G, cp.sum_squares(M @ L) <= (1.0 + eta) * V0]
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=solver)
    except Exception:
        prob.solve(solver="SCS")
    return np.asarray(M.value, float), prob.status


def scalar_weight_design(A, G, Sigma, rho, k: int = 1, eta: float = 0.10,
                         objective: str = "minimax", n_restart: int = 6,
                         seed: int = 0):
    """Best *scalar-weighted least squares* estimator for a chosen objective.

    ``objective='minimax'`` reproduces :func:`tmr_weights`; ``objective='average'``
    gives the metric-driven learned-weighting baseline, in the same weight class
    used by learned correspondence weighting (``A_s = S A`` with
    ``S = diag(s)``), granted its population optimum rather than a finite-sample
    approximation.
    """
    from scipy.optimize import minimize
    from .risk import average_risk

    A = np.asarray(A, float)
    G = np.asarray(G, float)
    Sigma = np.asarray(Sigma, float)
    n = A.shape[0] // 3
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,))
    M_gls = gls_influence(A, G, Sigma)
    vmax = (1.0 + eta) * float(np.trace(M_gls @ Sigma @ M_gls.T))
    rng = np.random.default_rng(seed)
    score = ((lambda M: sparse_risk_exact(M, rho, k, n_start=32, n_iter=60))
             if objective == "minimax" else
             (lambda M: average_risk(M, rho, k)))

    def obj(z):
        w = np.exp(np.clip(z, -12, 12))
        try:
            M = wls_influence(A, G, w)
        except np.linalg.LinAlgError:
            return 1e6
        v = float(np.trace(M @ Sigma @ M.T))
        pen = max(0.0, v - vmax) / max(vmax, 1e-12)
        return score(M) * (1.0 + 50.0 * pen)

    best_val, best_w = np.inf, np.ones(n)
    for z0 in [np.zeros(n)] + [0.7 * rng.standard_normal(n)
                               for _ in range(n_restart - 1)]:
        r = minimize(obj, z0, method="Nelder-Mead",
                     options=dict(maxiter=1500, xatol=1e-7, fatol=1e-9))
        if r.fun < best_val:
            best_val, best_w = float(r.fun), np.exp(np.clip(r.x, -12, 12))
    return wls_influence(A, G, best_w), best_w / best_w.max()



# Backwards-compatible alias: the multi-target solver was first written for a
# trajectory, which is the special case of a sampled path.
tmr_influence_trajectory = tmr_influence_multitarget
