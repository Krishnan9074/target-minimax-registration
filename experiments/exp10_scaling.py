"""Experiment 10 -- does the benefit survive as the number of fiducials grows?

As n increases, target leverage is spread over more landmarks and the worst-case
single-block gain falls for *every* estimator.  The question is whether the
*relative* advantage of target-minimax design survives, or whether it is an
artefact of the small fiducial counts used in the rest of the paper.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time
from multiprocessing import Pool
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.geometry import stack_jacobians, target_jacobian
from tmr.estimators import gls_influence, tmr_influence
from tmr.risk import sparse_risk_exact
from tmr.limits import fundamental_limit

OUT = os.path.join(ROOT, "results", "exp10_scaling.json")
NS = [5, 7, 9, 12, 16, 20, 25, 30, 40]


def one(args):
    n, trial, k = args
    rng = np.random.default_rng(20000 + 97 * n + trial)
    P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
    d = rng.standard_normal(3)
    x = 80.0 * d / np.linalg.norm(d)
    sig = 0.35 * np.exp(0.35 * rng.standard_normal(n))
    Sigma = np.diag(np.repeat(sig ** 2, 3))
    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), x)
    rho = np.ones(n)
    M_gls = gls_influence(A, G, Sigma)
    t0 = time.time()
    r = tmr_influence(A, G, Sigma, rho, k=k, eta=0.10)
    dt = time.time() - t0
    return dict(n=n, trial=trial, k=k,
                gls=sparse_risk_exact(M_gls, rho, k), tmr=r.risk,
                gap=r.gap, seconds=dt,
                limit=fundamental_limit(A, G, k, loss="2", n_start=8, n_iter=12))


if __name__ == "__main__":
    jobs = [(n, t, k) for n in NS for t in range(12) for k in (1, 2)]
    with Pool(5) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            recs.append(r)
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)
    print(f'{"n":>4} {"k":>3} {"GLS risk":>10} {"TMR risk":>10} {"ratio":>8} '
          f'{"vs limit":>9} {"solve s":>8}')
    for n in NS:
        for k in (1, 2):
            sub = [r for r in recs if r["n"] == n and r["k"] == k]
            if not sub: continue
            g = np.median([r["gls"] for r in sub])
            t = np.median([r["tmr"] for r in sub])
            rat = np.median([r["tmr"] / r["gls"] for r in sub])
            lim = np.median([r["tmr"] / max(r["limit"], 1e-12) for r in sub])
            s = np.median([r["seconds"] for r in sub])
            print(f"{n:>4} {k:>3} {g:10.4f} {t:10.4f} {rat:8.3f} {lim:9.3f} {s:8.2f}")
