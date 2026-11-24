"""Worst-case target displacement under k-sparse bounded fiducial corruption.

Let ``M in R^{3 x 3n}`` be the *target-influence operator* of a locally linear
unbiased estimator, i.e. the first-order map from a fiducial perturbation
``a = (a_1, ..., a_n)`` to the displacement of the transformed target,

    Delta x  =  M a  =  sum_i M_i a_i,      M_i in R^{3x3}.

The admissible corruption set is

    A(k, rho) = { a : #{i : a_i != 0} <= k,  ||a_i||_2 <= rho_i }.

Key identity (Theorem "exact sparse risk"): for any ``M``,

    R_k(M) = max_{||a|| in A(k,rho)} ||M a||_2
           = max_{||u||_2 = 1}  topk_i ( rho_i * ||M_i^T u||_2 ),

where ``topk`` sums the ``k`` largest entries.  The inner function of ``u`` is a
maximum of convex positively homogeneous functions, so a normalised
subgradient iteration is monotonically ascending; multi-start makes the value
reliable and it is cross-checked against brute-force enumeration in the tests.
"""
from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "blocks_of", "sparse_risk_exact", "sparse_risk_topk_bound",
    "single_block_risk", "clean_target_variance", "geometric_amplification",
    "fibonacci_sphere", "worst_case_perturbation",
]


def blocks_of(M: np.ndarray) -> np.ndarray:
    """Reshape ``(3, 3n)`` influence operator into ``(n, 3, 3)`` blocks."""
    M = np.asarray(M, dtype=float)
    n = M.shape[1] // 3
    return M.reshape(3, n, 3).transpose(1, 0, 2)


def fibonacci_sphere(m: int) -> np.ndarray:
    """``m`` quasi-uniform points on the unit 2-sphere."""
    i = np.arange(m, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / m)
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    theta = golden * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def _topk_objective(Mb: np.ndarray, rho: np.ndarray, U: np.ndarray, k: int):
    """Evaluate ``topk_i rho_i ||M_i^T u||`` for a batch of directions ``U``.

    Returns ``(values, gains)`` with ``gains[m, i] = rho_i ||M_i^T u_m||``.
    """
    # g[m, i, :] = M_i^T u_m
    g = np.einsum('icd,mc->mid', Mb, U)
    gains = rho[None, :] * np.linalg.norm(g, axis=2)
    if k >= gains.shape[1]:
        return gains.sum(axis=1), gains
    part = np.partition(gains, -k, axis=1)[:, -k:]
    return part.sum(axis=1), gains


def sparse_risk_exact(M: np.ndarray, rho, k: int, n_start: int = 128,
                      n_iter: int = 200, tol: float = 1e-12,
                      return_argmax: bool = False, return_top: int = 0):
    """Worst-case target displacement ``R_k(M)`` under ``A(k, rho)``.

    Solves ``max_{||u||=1} topk_i rho_i ||M_i^T u||`` by a normalised
    subgradient ascent from ``n_start`` quasi-uniform initialisations.
    """
    Mb = blocks_of(M)
    n = Mb.shape[0]
    rho = np.broadcast_to(np.asarray(rho, dtype=float).ravel(), (n,)).copy()
    k = int(min(k, n))

    U = fibonacci_sphere(max(n_start, 8))
    for _ in range(n_iter):
        _, gains = _topk_objective(Mb, rho, U, k)
        # subgradient of the top-k sum w.r.t. u
        if k >= n:
            sel = np.ones_like(gains, dtype=bool)
        else:
            thr = np.partition(gains, -k, axis=1)[:, [-k]]
            sel = gains >= thr
            # break ties so that exactly k blocks are selected
            excess = sel.sum(axis=1) - k
            for m in np.nonzero(excess > 0)[0]:
                idx = np.nonzero(sel[m])[0]
                sel[m, idx[k:]] = False
        Mu = np.einsum('icd,mc->mid', Mb, U)            # (m, n, 3)
        nrm = np.linalg.norm(Mu, axis=2)
        nrm = np.where(nrm < 1e-15, 1e-15, nrm)
        w = (rho[None, :] / nrm) * sel                   # (m, n)
        # grad = sum_i w_i M_i M_i^T u
        grad = np.einsum('mi,icd,mid->mc', w, Mb, Mu)
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        gn = np.where(gn < 1e-15, 1e-15, gn)
        Unew = grad / gn
        if np.max(np.abs(Unew - U)) < tol:
            U = Unew
            break
        U = Unew

    vals, gains = _topk_objective(Mb, rho, U, k)
    j = int(np.argmax(vals))
    val = float(vals[j])
    if return_top:
        # diverse near-optimal directions, used as cutting planes
        order = np.argsort(vals)[::-1]
        picks, sel = [], []
        for m in order:
            u = U[m]
            if all(abs(float(u @ w)) < 0.98 for w in sel):
                sel.append(u)
                picks.append(u)
            if len(picks) >= return_top:
                break
        return val, picks
    if not return_argmax:
        return val
    u = U[j]
    order = np.argsort(gains[j])[::-1][:k]
    return val, u, np.sort(order)


def worst_case_perturbation(M: np.ndarray, rho, k: int, **kw) -> np.ndarray:
    """A maximiser ``a in A(k, rho)`` of ``||M a||`` (first-order optimal)."""
    Mb = blocks_of(M)
    n = Mb.shape[0]
    rho = np.broadcast_to(np.asarray(rho, dtype=float).ravel(), (n,)).copy()
    _, u, S = sparse_risk_exact(M, rho, k, return_argmax=True, **kw)
    a = np.zeros((n, 3))
    for i in S:
        d = Mb[i].T @ u
        nd = np.linalg.norm(d)
        if nd > 1e-15:
            a[i] = rho[i] * d / nd
    return a


def single_block_risk(M: np.ndarray, rho) -> float:
    """``R_1(M) = max_i rho_i ||M_i||_2`` (exact, no search needed)."""
    Mb = blocks_of(M)
    rho = np.broadcast_to(np.asarray(rho, dtype=float).ravel(), (Mb.shape[0],))
    return float(np.max(rho * np.linalg.norm(Mb, ord=2, axis=(1, 2))))


def sparse_risk_topk_bound(M: np.ndarray, rho, k: int) -> float:
    """Convex upper bound: sum of the ``k`` largest ``rho_i ||M_i||_2``."""
    Mb = blocks_of(M)
    rho = np.broadcast_to(np.asarray(rho, dtype=float).ravel(), (Mb.shape[0],))
    s = rho * np.linalg.norm(Mb, ord=2, axis=(1, 2))
    k = int(min(k, s.size))
    return float(np.sort(s)[::-1][:k].sum())


def sparse_risk_bruteforce(M: np.ndarray, rho, k: int, n_dir: int = 20000) -> float:
    """Reference value by explicit support enumeration and dense sphere search."""
    Mb = blocks_of(M)
    n = Mb.shape[0]
    rho = np.broadcast_to(np.asarray(rho, dtype=float).ravel(), (n,))
    U = fibonacci_sphere(n_dir)
    best = 0.0
    for S in itertools.combinations(range(n), int(min(k, n))):
        g = np.stack([rho[i] * np.linalg.norm(U @ Mb[i], axis=1) for i in S])
        best = max(best, float(g.sum(axis=0).max()))
    return best


def clean_target_variance(M: np.ndarray, Sigma: np.ndarray) -> float:
    """``tr(M Sigma M^T)`` -- expected squared target error under clean noise."""
    M = np.asarray(M, dtype=float)
    return float(np.trace(M @ np.asarray(Sigma, dtype=float) @ M.T))


def geometric_amplification(A: np.ndarray, G: np.ndarray,
                            n_iter: int = 500) -> float:
    """Estimator-independent lever-arm constant

    ``kappa_x = sup { ||G xi|| : max_i ||A_i xi|| <= 1 }``.

    Computed by bisection on ``||G xi||`` over the polytope-like feasible set
    using a simple projected ascent; ``A`` is ``(3n, 6)`` and ``G`` is ``(3, 6)``.
    """
    A = np.asarray(A, dtype=float)
    G = np.asarray(G, dtype=float)
    n = A.shape[0] // 3
    Ab = A.reshape(n, 3, 6)

    def feas_scale(xi):
        m = max(float(np.max(np.linalg.norm(Ab @ xi, axis=1))), 1e-15)
        return xi / m

    best = 0.0
    rng = np.random.default_rng(0)
    for _ in range(24):
        xi = feas_scale(rng.standard_normal(6))
        for _ in range(n_iter):
            d = G @ xi
            nd = np.linalg.norm(d)
            if nd < 1e-15:
                break
            grad = G.T @ (d / nd)
            xi = feas_scale(xi + 0.1 * grad)
        best = max(best, float(np.linalg.norm(G @ xi)))
    return best


def average_risk(M: np.ndarray, rho, k: int) -> float:
    """Mean squared target displacement under a *random* k-sparse corruption.

    If the support is a uniformly random k-subset and each corrupted block is
    drawn uniformly from the sphere of radius ``rho_i``, independently, then

        E ||M a||^2  =  (k / 3n) * sum_i rho_i^2 ||M_i||_F^2 ,

    because the cross terms vanish and ``E[u u'] = I/3`` for ``u`` uniform on
    the unit sphere.  The average-case objective is therefore a weighted squared
    Frobenius norm -- smooth, convex, and (Proposition: average-case collapse)
    minimised by the Gauss-Markov operator whenever the budget is uniform and
    the nominal noise is homoscedastic.
    """
    Mb = blocks_of(M)
    n = Mb.shape[0]
    rho = np.broadcast_to(np.asarray(rho, dtype=float).ravel(), (n,))
    k = int(min(k, n))
    return float(k / (3.0 * n) * np.sum(rho ** 2
                                        * np.linalg.norm(Mb, axis=(1, 2)) ** 2))


def average_risk_mc(M: np.ndarray, rho, k: int, n_mc: int = 200000,
                    seed: int = 0) -> float:
    """Monte-Carlo reference for :func:`average_risk` (used only in tests)."""
    Mb = blocks_of(M)
    n = Mb.shape[0]
    rho = np.broadcast_to(np.asarray(rho, dtype=float).ravel(), (n,))
    rng = np.random.default_rng(seed)
    k = int(min(k, n))
    tot = 0.0
    for _ in range(n_mc):
        S = rng.choice(n, size=k, replace=False)
        d = np.zeros(3)
        for i in S:
            u = rng.standard_normal(3)
            d += rho[i] * Mb[i] @ (u / np.linalg.norm(u))
        tot += float(d @ d)
    return tot / n_mc
