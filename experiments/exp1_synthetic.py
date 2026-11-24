"""Experiment 1 -- robustness / efficiency trade-off on random geometries.

For every random fiducial configuration we compute, exactly and in closed form,
the worst-case k-sparse target displacement and the clean target variance of

    LS   : unweighted least squares,
    GLS  : Gauss-Markov (nominal-covariance) weighting,
    TMR  : target-minimax design at several clean-efficiency budgets eta,

and record the ratios that define the robustness-efficiency Pareto frontier.
All risks are per unit corruption budget (rho = 1 mm), so they scale linearly
with rho.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import random_instance, first_order_operators
from tmr.estimators import gls_influence, tmr_influence, wls_influence
from tmr.risk import sparse_risk_exact, geometric_amplification

ETAS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 1e6]
OUT = os.path.join(ROOT, "results", "exp1_synthetic.json")


def one(args):
    trial, n, target_dist, k, anis = args
    rng = np.random.default_rng(hash((trial, n, int(target_dist), k)) % 2 ** 31)
    inst = random_instance(rng, n=n, spread=60.0, target_dist=target_dist,
                           sigma=0.35, rho=1.0, k=k, anisotropy=anis)
    A, G = first_order_operators(inst)
    S, rho = inst.Sigma, inst.rho

    M_ls = wls_influence(A, G, np.ones(n))
    M_gls = gls_influence(A, G, S)
    v = lambda M: float(np.trace(M @ S @ M.T))
    rec = dict(trial=trial, n=n, target_dist=target_dist, k=k, anisotropy=anis,
               kappa=geometric_amplification(A, G),
               ls=dict(risk=sparse_risk_exact(M_ls, rho, k), var=v(M_ls)),
               gls=dict(risk=sparse_risk_exact(M_gls, rho, k), var=v(M_gls)),
               tmr={})
    for eta in ETAS:
        r = tmr_influence(A, G, S, rho, k=k, eta=eta)
        rec["tmr"][f"{eta:g}"] = dict(risk=r.risk, var=r.variance, gap=r.gap,
                                      lb=r.lower_bound, status=r.status)
    return rec


if __name__ == "__main__":
    jobs = [(t, n, d, k, a)
            for n in (5, 7, 9)
            for d in (40.0, 80.0, 120.0)
            for k in (1, 2)
            for a in (0.0, 0.6)
            for t in range(20)]
    print(f"{len(jobs)} instances, {len(ETAS)} designs each")
    with Pool(12) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            recs.append(r)
            if (i + 1) % 60 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)

    # quick summary
    import collections
    agg = collections.defaultdict(list)
    for r in recs:
        for eta in ("0.05", "0.1", "0.2"):
            agg[(r["k"], eta)].append(r["tmr"][eta]["risk"] / r["gls"]["risk"])
        agg[(r["k"], "ls")].append(r["ls"]["risk"] / r["gls"]["risk"])
    for key in sorted(agg, key=lambda t: (t[0], str(t[1]))):
        a = np.array(agg[key])
        print(f"k={key[0]} {key[1]:>4}: risk / GLS  median {np.median(a):.3f}  "
              f"[p10 {np.percentile(a, 10):.3f}, p90 {np.percentile(a, 90):.3f}]")
