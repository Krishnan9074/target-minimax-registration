"""Experiment 7 -- what is actually doing the work: minimax, or the design class?

A 2x2 factorial over (objective) x (design class), with Gauss-Markov as the
reference point:

                       scalar weights            full influence operator
    average case   metric-driven learned          average-case design
                   weighting baseline
    worst case     TMR restricted to weights      TMR (this paper)

The average-case column is the population optimum of a learned correspondence
weighting method -- i.e. the best any amount of training data could reach --- so
the comparison is maximally favourable to that baseline.  All five estimators
are locally linear, so their exact worst-case risk is given in closed form by
the k-sparse risk identity; Experiment 2 established that this matches a direct
black-box attack to 0.1% in the median.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.geometry import stack_jacobians, target_jacobian
from tmr.estimators import (gls_influence, tmr_influence, average_influence,
                            scalar_weight_design)
from tmr.risk import sparse_risk_exact, average_risk

OUT = os.path.join(ROOT, "results", "exp7_factorial.json")
ETA = 0.10


def geometry(trial, homoscedastic=False):
    """Random geometry, or the L1 vertebra when trial < 0."""
    if trial < 0:
        from tmr.anatomy import build_vertebra
        from experiments.exp5_vertebra import SIGMA_BY_LANDMARK
        v = build_vertebra()
        names = list(SIGMA_BY_LANDMARK)
        P = v.fiducials(names)
        sig = np.array([SIGMA_BY_LANDMARK[nm] for nm in names])
        x = v.tip["left" if trial == -1 else "right"]
        if homoscedastic:
            sig = np.full(len(sig), float(sig.mean()))
        return P, x, np.diag(np.repeat(sig ** 2, 3)), "L1 vertebra"
    rng = np.random.default_rng(5000 + trial)
    n = 7
    P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
    d = rng.standard_normal(3)
    x = 80.0 * d / np.linalg.norm(d)
    sig = (np.full(n, 0.35) if homoscedastic
           else 0.35 * np.exp(0.35 * rng.standard_normal(n)))
    return P, x, np.diag(np.repeat(sig ** 2, 3)), (
        "random-homoscedastic" if homoscedastic else "random")


def one(args):
    trial, k, homo = args
    P, x, Sigma, kind = geometry(trial, homoscedastic=homo)
    n = len(P)
    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), x)
    rho = np.ones(n)
    var = lambda M: float(np.trace(M @ Sigma @ M.T))

    designs = {}
    designs["GLS"] = gls_influence(A, G, Sigma)
    designs["learned-avg (scalar)"] = scalar_weight_design(
        A, G, Sigma, rho, k=k, eta=ETA, objective="average", seed=abs(trial))[0]
    designs["avg (full)"] = average_influence(A, G, Sigma, rho, k=k,
                                              eta=ETA)[0]
    designs["TMR-W (scalar)"] = scalar_weight_design(
        A, G, Sigma, rho, k=k, eta=ETA, objective="minimax", seed=abs(trial))[0]
    designs["TMR (full)"] = tmr_influence(A, G, Sigma, rho, k=k, eta=ETA).M

    return dict(trial=trial, k=k, kind=kind, n=n,
                results={nm: dict(worst=sparse_risk_exact(M, rho, k),
                                  average=average_risk(M, rho, k),
                                  var=var(M))
                         for nm, M in designs.items()})


ORDER = ["GLS", "learned-avg (scalar)", "avg (full)", "TMR-W (scalar)",
         "TMR (full)"]

if __name__ == "__main__":
    jobs = ([(t, k, False) for t in list(range(24)) + [-1, -2] for k in (1, 2)]
            + [(t, k, True) for t in range(24) for k in (1, 2)])
    with Pool(5) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            recs.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)

    for kind in ("random", "random-homoscedastic", "L1 vertebra"):
        for k in (1, 2):
            sub = [r for r in recs if r["kind"] == kind and r["k"] == k]
            if not sub:
                continue
            print(f"\n=== {kind}, k={k}  (worst-case risk relative to GLS) ===")
            for nm in ORDER:
                w = np.array([r["results"][nm]["worst"] / r["results"]["GLS"]["worst"]
                              for r in sub])
                a = np.array([r["results"][nm]["average"] / r["results"]["GLS"]["average"]
                              for r in sub])
                v = np.array([r["results"][nm]["var"] / r["results"]["GLS"]["var"]
                              for r in sub])
                print(f"  {nm:22s} worst {np.median(w):.3f}   "
                      f"average {np.median(a):.3f}   variance {np.median(v):.3f}")
