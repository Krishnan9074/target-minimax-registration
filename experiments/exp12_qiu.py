"""Experiment 12 -- correspondence with Qiu et al. (2018).

Their approximated problem (their Eq. 2) is

    max_{R,s} ||R p + s - p||   s.t.   ||R b_i + s - b_i|| <= c  for all i,

in which the rigid transform is the decision variable and no estimator appears.
We check two things: (a) that its optimal value is c * kappa_x, the quantity of
our fundamental-limit theorem; and (b) that, as they prove, only fiducials on the
convex hull of the fiducial set affect it.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, NonlinearConstraint
from scipy.spatial import ConvexHull

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.geometry import stack_jacobians, target_jacobian, expm_so3
from tmr.risk import fibonacci_sphere
from tmr.limits import fundamental_limit

OUT = os.path.join(ROOT, "results", "exp12_qiu.json")


def kappa_and_argmax(P, x, n_start=24, n_iter=30):
    n = len(P)
    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), x)
    xi = cp.Variable(6); c = cp.Parameter(3)
    blk = cp.hstack([cp.norm(A[3 * i:3 * i + 3, :] @ xi, 2) for i in range(n)])
    pr = cp.Problem(cp.Maximize((c @ G) @ xi), [cp.max(blk) <= 1])
    best, bxi = 0.0, np.zeros(6)
    for c0 in fibonacci_sphere(n_start):
        cc = c0.copy()
        for _ in range(n_iter):
            c.value = cc; pr.solve(solver="CLARABEL")
            v = G @ np.asarray(xi.value).ravel(); nv = np.linalg.norm(v)
            if nv < 1e-12: break
            cn = v / nv
            if np.linalg.norm(cn - cc) < 1e-11: cc = cn; break
            cc = cn
        c.value = cc; pr.solve(solver="CLARABEL")
        z = np.asarray(xi.value).ravel()
        if np.linalg.norm(G @ z) > best:
            best, bxi = float(np.linalg.norm(G @ z)), z
    return best, bxi


def qiu_exact(P, x, c, xi0):
    """Solve Qiu Eq. (2) directly, warm-started from the linearised optimum."""
    obj = lambda z: -np.linalg.norm(expm_so3(z[:3]) @ x + z[3:] - x)
    con = NonlinearConstraint(
        lambda z: np.linalg.norm(P @ expm_so3(z[:3]).T + z[3:] - P, axis=1),
        -np.inf, c)
    best = 0.0
    for sc in (1.0, 0.999, 0.99, 0.95):
        r = minimize(obj, c * sc * xi0, method="SLSQP", constraints=[con],
                     options=dict(maxiter=800, ftol=1e-14))
        g = np.linalg.norm(P @ expm_so3(r.x[:3]).T + r.x[3:] - P, axis=1).max()
        if g <= c * (1 + 1e-7):
            best = max(best, -float(r.fun))
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    corr, hull_rows = [], []
    for t in range(8):
        P = rng.uniform(-60, 60, (7, 3)); P -= P.mean(0)
        x = 90 * rng.standard_normal(3); x /= np.linalg.norm(x) / 90
        kap, xi0 = kappa_and_argmax(P, x)
        for c in (0.05, 0.2, 1.0):
            q = qiu_exact(P, x, c, xi0)
            if q > 0:
                corr.append(dict(trial=t, c=c, qiu=q, kappa=kap,
                                 ratio=q / (c * kap)))
    for t in range(10):
        n = 12
        P = rng.uniform(-60, 60, (n, 3)); P -= P.mean(0)
        x = 90 * rng.standard_normal(3); x /= np.linalg.norm(x) / 90
        hull = sorted(set(ConvexHull(P).vertices.tolist()))
        if len(hull) == n:
            continue
        G = target_jacobian(np.eye(3), x)
        full = fundamental_limit(stack_jacobians(np.eye(3), P), G, k=n,
                                 loss="2", n_start=16, n_iter=25)
        ho = fundamental_limit(stack_jacobians(np.eye(3), P[hull]), G,
                               k=len(hull), loss="2", n_start=16, n_iter=25)
        hull_rows.append(dict(trial=t, n=n, n_hull=len(hull),
                              full=full, hull_only=ho, ratio=ho / full))
    json.dump(dict(correspondence=corr, hull=hull_rows), open(OUT, "w"), indent=2)
    r = np.array([d["ratio"] for d in corr])
    h = np.array([d["ratio"] for d in hull_rows])
    print("wrote", OUT)
    print(f"(a) Qiu optimum / (c * kappa_x): n={len(r)} solves, "
          f"max deviation from 1 = {np.abs(r - 1).max():.2e}")
    print(f"(b) dropping interior fiducials: n={len(h)} configurations, "
          f"max relative change in kappa_x = {np.abs(h - 1).max():.2e}")
