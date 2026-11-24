"""Experiment 9 -- how tight are the two estimator-independent limits?

Compares the achieved minimax optimum against (a) the elementary counting bound
(k/n) rho kappa_x and (b) the exact limit kappa^(k) of Theorem "limit", for both
Euclidean and scalar target losses.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys
from multiprocessing import Pool
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.geometry import stack_jacobians, target_jacobian
from tmr.estimators import tmr_influence
from tmr.risk import geometric_amplification
from tmr.limits import fundamental_limit, scalar_design, scalar_limit

OUT = os.path.join(ROOT, "results", "exp9_limits.json")


def one(args):
    trial, k = args
    rng = np.random.default_rng(9000 + trial)
    n = int(rng.integers(5, 9))
    P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
    x = 80 * rng.standard_normal(3); x /= np.linalg.norm(x) / 80
    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), x)
    one_ = np.ones(n)
    achieved = tmr_influence(A, G, np.eye(3 * n), one_, k=k, eta=1e6).risk
    new = fundamental_limit(A, G, k, loss="2")
    old = (k / n) * geometric_amplification(A, G)
    c = rng.standard_normal(3); c /= np.linalg.norm(c)
    _, s_ach, _ = scalar_design(A, c @ G, one_, k)
    s_lim = scalar_limit(A, c @ G, k)
    return dict(trial=trial, k=k, n=n, achieved=achieved, limit_new=new,
                limit_old=old, scalar_achieved=s_ach, scalar_limit=s_lim)


if __name__ == "__main__":
    jobs = [(t, k) for t in range(20) for k in (1, 2, 3)]
    with Pool(5) as pool:
        recs = list(pool.imap_unordered(one, jobs))
    json.dump(recs, open(OUT, "w"))
    g = lambda f: np.array([f(r) for r in recs])
    rn = g(lambda r: r["achieved"] / r["limit_new"])
    ro = g(lambda r: r["achieved"] / r["limit_old"])
    rs = g(lambda r: r["scalar_achieved"] / max(r["scalar_limit"], 1e-12))
    print("wrote", OUT)
    print(f"Euclidean loss, achieved / exact limit : "
          f"median {np.median(rn):.4f}  range [{rn.min():.4f}, {rn.max():.4f}]")
    print(f"Euclidean loss, achieved / counting bd : "
          f"median {np.median(ro):.3f}  range [{ro.min():.3f}, {ro.max():.3f}]")
    print(f"Scalar loss,    achieved / exact limit : "
          f"median {np.median(rs):.6f} range [{rs.min():.6f}, {rs.max():.6f}]")
