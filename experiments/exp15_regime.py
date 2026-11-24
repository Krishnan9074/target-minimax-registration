"""Experiment 15 -- when does target-minimax design *not* help?

The framework says where the gain comes from: every unbiased estimator has
worst-case risk at least ``rho * kappa^(k)`` (the estimator-independent limit),
Gauss--Markov sits somewhere above it, and TMR can only recover the slack
between the two.  So the ratio TMR/GM should be predicted by

    slack = R_k(GM) / (rho * kappa^(k)),

and there should be nothing to gain when Gauss--Markov is already near the
floor.  Four sweeps probe the regimes where that is expected to happen:

  target   the target moves from the fiducial centroid, through the convex
           hull of the fiducials, out to 160 mm (a target inside the hull has
           no lever arm to exploit);
  n        the number of fiducials from the minimum 4 to 40 (the unbiasedness
           constraint M A = G leaves less freedom the smaller n is);
  cond     the fiducial cloud is flattened toward coplanar (aspect ratio s), or
           one fiducial is placed far from the rest;
  noise    heteroscedastic spread from none (homoscedastic) upward (under
           homoscedastic noise Gauss--Markov is already unweighted LS and its
           influence is already spread).

Everything is locally linear, so the exact k-sparse risk identity is used and
no black-box attack is needed.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.spatial import ConvexHull

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.geometry import stack_jacobians, target_jacobian
from tmr.estimators import gls_influence, tmr_influence
from tmr.risk import sparse_risk_exact, geometric_amplification
from tmr.limits import fundamental_limit

OUT = os.path.join(ROOT, "results", "exp15_regime.json")
N_GEOM = 24
ETA = 0.10
SWEEPS = {
    "target": [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0],
    "n": [4, 5, 6, 7, 9, 12, 20, 40],
    "cond": [1.0, 0.5, 0.25, 0.1, 0.03, "outlier"],
    "noise": [0.0, 0.2, 0.35, 0.6],
}


def hull_signed_distance(P, x):
    """Negative inside conv(P), positive outside (largest facet violation)."""
    try:
        h = ConvexHull(P)
    except Exception:
        return float("nan")
    return float(np.max(h.equations[:, :3] @ x + h.equations[:, 3]))


def make_case(sweep, level, trial):
    rng = np.random.default_rng(70000 + 1000 * list(SWEEPS).index(sweep) + trial)
    n, d, s, hs, outlier = 7, 80.0, 1.0, 0.35, False
    if sweep == "target":
        d = float(level)
    elif sweep == "n":
        n = int(level)
    elif sweep == "cond":
        if level == "outlier":
            outlier = True
        else:
            s = float(level)
    elif sweep == "noise":
        hs = float(level)
    P = rng.uniform(-60, 60, size=(n, 3))
    P[:, 2] *= s
    if outlier:
        P[0] = 3.0 * P[0] / np.linalg.norm(P[0]) * 60.0
    P -= P.mean(0)
    u = rng.standard_normal(3)
    x = d * u / np.linalg.norm(u)
    sig = 0.35 * np.exp(hs * rng.standard_normal(n))
    Sigma = np.diag(np.repeat(sig ** 2, 3))
    return P, x, Sigma


def one(args):
    sweep, level, trial, k = args
    P, x, Sigma = make_case(sweep, level, trial)
    n = len(P)
    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), x)
    rho = np.ones(n)
    M_gls = gls_influence(A, G, Sigma)
    t0 = time.time()
    r = tmr_influence(A, G, Sigma, rho, k=k, eta=ETA)
    Si = np.linalg.inv(Sigma)
    return dict(sweep=sweep, level=level, trial=trial, k=k, n=n,
                target_dist=float(np.linalg.norm(x)),
                hull_dist=hull_signed_distance(P, x),
                cond=float(np.linalg.cond(A.T @ Si @ A)),
                aspect=float(np.linalg.svd(P - P.mean(0), compute_uv=False)[2]
                             / np.linalg.svd(P - P.mean(0), compute_uv=False)[0]),
                kappa_x=geometric_amplification(A, G),
                limit=fundamental_limit(A, G, k, loss="2", n_start=8, n_iter=12),
                gls=sparse_risk_exact(M_gls, rho, k), tmr=r.risk, gap=r.gap,
                var_ratio=float(r.variance / np.trace(M_gls @ Sigma @ M_gls.T)),
                seconds=time.time() - t0)


if __name__ == "__main__":
    jobs = [(sw, lv, t, k) for sw, lvs in SWEEPS.items() for lv in lvs
            for t in range(N_GEOM) for k in (1, 2)]
    print(f"{len(jobs)} instances", flush=True)
    with Pool(5) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            recs.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT, flush=True)
    for sw, lvs in SWEEPS.items():
        print(f"\n=== {sw} ===  level  k  TMR/GM  GM/limit  TMR/limit  in-hull")
        for lv in lvs:
            for k in (1, 2):
                sub = [r for r in recs if r["sweep"] == sw and r["level"] == lv
                       and r["k"] == k]
                rat = np.median([r["tmr"] / r["gls"] for r in sub])
                sl = np.median([r["gls"] / max(r["limit"], 1e-12) for r in sub])
                tl = np.median([r["tmr"] / max(r["limit"], 1e-12) for r in sub])
                ih = np.mean([r["hull_dist"] < 0 for r in sub])
                print(f"  {str(lv):>8} {k}  {rat:6.3f}  {sl:8.3f}  {tl:9.3f}  {ih:6.0%}")
