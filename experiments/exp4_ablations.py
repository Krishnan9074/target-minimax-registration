"""Experiment 4 -- ablations that isolate what TMR actually contributes.

A1  clean-efficiency constraint removed          -> why eta is necessary
A2  fiducial-minimax objective instead of target -> separation from minimax
                                                    point-cloud alignment
A3  dense i.i.d. Gaussian corruption             -> TMR is not a better
                                                    heteroscedastic weighting
A4  target moved, fiducials fixed                -> the design is task adaptive
A5  threat-model mismatch (design k vs attack k) -> graceful degradation
A6  design restricted to scalar weights          -> separation from reweighting
                                                    and from c-optimal design
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.geometry import stack_jacobians, target_jacobian, random_rotation
from tmr.estimators import (gls_influence, tmr_influence, tmr_weights,
                            tmr_influence_trajectory, wls_influence)
from tmr.risk import sparse_risk_exact

OUT = os.path.join(ROOT, "results", "exp4_ablations.json")
SIGMA = 0.35


def setup(trial, n=7, target_dist=80.0):
    rng = np.random.default_rng(3000 + trial)
    P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
    d = rng.standard_normal(3)
    x = target_dist * d / np.linalg.norm(d)
    Sigma = SIGMA ** 2 * np.eye(3 * n)
    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), x)
    return P, x, Sigma, A, G, np.ones(n), rng


def one(trial):
    P, x, S, A, G, rho, rng = setup(trial)
    n = len(P)
    fro = lambda M: float(np.linalg.norm(M, "fro") ** 2)
    var = lambda M: float(np.trace(M @ S @ M.T))
    M_gls = gls_influence(A, G, S)
    rec = dict(trial=trial)

    # ---- A1: with and without the clean-efficiency constraint -------------- #
    a1 = {}
    for eta in (0.05, 0.10, 0.20, 1.0, 1e6):
        r = tmr_influence(A, G, S, rho, k=1, eta=eta)
        a1[f"{eta:g}"] = dict(risk=r.risk, var_ratio=r.variance / var(M_gls))
    rec["A1"] = dict(gls_risk=sparse_risk_exact(M_gls, rho, 1),
                     gls_var=var(M_gls), sweep=a1)

    # ---- A2: fiducial-minimax (Geng-style) design vs target-minimax -------- #
    A_list = [A[3 * i:3 * i + 3, :] for i in range(n)]
    fid = tmr_influence_trajectory(A, A_list, S, rho, k=1, eta=0.10)
    K_fid = fid["K"]
    tgt = tmr_influence(A, G, S, rho, k=1, eta=0.10)
    Si = np.linalg.inv(S)
    K_gls = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
    rec["A2"] = dict(
        fid_obj_gls=max(sparse_risk_exact(Ai @ K_gls, rho, 1) for Ai in A_list),
        fid_obj_fid=fid["risk"],
        target_risk_gls=sparse_risk_exact(M_gls, rho, 1),
        target_risk_fid=sparse_risk_exact(G @ K_fid, rho, 1),
        target_risk_tmr=tgt.risk)

    # ---- A3: dense i.i.d. Gaussian corruption ------------------------------ #
    a3 = {"gls": fro(M_gls)}
    for eta in (0.05, 0.10, 0.20):
        a3[f"{eta:g}"] = fro(tmr_influence(A, G, S, rho, k=1, eta=eta).M)
    rec["A3"] = a3

    # ---- A4: move the target, keep the fiducials ---------------------------- #
    a4 = []
    for dist in (20.0, 40.0, 80.0, 160.0):
        for rep in range(2):
            d = rng.standard_normal(3)
            xx = dist * d / np.linalg.norm(d)
            Gx = target_jacobian(np.eye(3), xx)
            Mg = gls_influence(A, Gx, S)
            r = tmr_influence(A, Gx, S, rho, k=1, eta=0.10)
            lev_g = np.linalg.norm(Mg.reshape(3, n, 3).transpose(1, 0, 2),
                                   ord=2, axis=(1, 2))
            lev_t = np.linalg.norm(r.M.reshape(3, n, 3).transpose(1, 0, 2),
                                   ord=2, axis=(1, 2))
            a4.append(dict(dist=dist, gls_risk=sparse_risk_exact(Mg, rho, 1),
                           tmr_risk=r.risk,
                           lev_gls=list(map(float, lev_g)),
                           lev_tmr=list(map(float, lev_t))))
    rec["A4"] = a4

    # ---- A5: design-k / attack-k mismatch ---------------------------------- #
    designs = {kd: tmr_influence(A, G, S, rho, k=kd, eta=0.10).M
               for kd in (1, 2, 3)}
    rec["A5"] = {f"design{kd}": {f"attack{ka}":
                                 sparse_risk_exact(Md, rho, ka)
                                 for ka in (1, 2, 3)}
                 for kd, Md in designs.items()}
    rec["A5"]["gls"] = {f"attack{ka}": sparse_risk_exact(M_gls, rho, ka)
                        for ka in (1, 2, 3)}

    # ---- A6: scalar-weight restriction ------------------------------------- #
    M_w, w, risk_w = tmr_weights(A, G, S, rho, k=1, eta=0.10, n_restart=4,
                                 seed=trial)
    M_t = tmr_influence(A, G, S, rho, k=1, eta=0.10).M
    # distance from the unrestricted optimum to the weighted-LS manifold,
    # minimised over the weights rather than sampled
    from scipy.optimize import minimize as _min
    fdist = lambda z: float(np.linalg.norm(
        wls_influence(A, G, np.exp(np.clip(z, -10, 10))) - M_t, "fro"))
    dists = [ _min(fdist, z0, method="Nelder-Mead",
                   options=dict(maxiter=2500, xatol=1e-9, fatol=1e-12)).fun
              for z0 in [np.zeros(n)] + [0.8 * rng.standard_normal(n)
                                         for _ in range(5)] ]
    rec["A6"] = dict(risk_tmr=float(sparse_risk_exact(M_t, rho, 1)),
                     risk_wls=float(risk_w),
                     risk_gls=float(sparse_risk_exact(M_gls, rho, 1)),
                     weights=list(map(float, w)),
                     min_dist_to_wls_manifold=float(np.min(dists)),
                     norm_tmr=float(np.linalg.norm(M_t, "fro")))
    return rec


if __name__ == "__main__":
    with Pool(10) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, range(24))):
            recs.append(r)
            print(f'  {i+1}/24', flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT, f"({len(recs)} geometries)")

    g = lambda f: np.array([f(r) for r in recs])
    print("\nA1  clean-efficiency budget")
    for eta in ("0.05", "0.1", "0.2", "1", "1e+06"):
        rr = g(lambda r: r["A1"]["sweep"][eta]["risk"] / r["A1"]["gls_risk"])
        vv = g(lambda r: r["A1"]["sweep"][eta]["var_ratio"])
        print(f"  eta={eta:>6}: risk/GLS {np.median(rr):.3f}   "
              f"variance/GLS {np.median(vv):.3f}")

    print("\nA2  fiducial-minimax vs target-minimax (both at eta=0.10)")
    print(f"  fiducial objective: GLS {np.median(g(lambda r: r['A2']['fid_obj_gls'])):.3f}"
          f" -> fiducial design {np.median(g(lambda r: r['A2']['fid_obj_fid'])):.3f}")
    print(f"  target   objective: GLS {np.median(g(lambda r: r['A2']['target_risk_gls'])):.3f}"
          f" -> fiducial design {np.median(g(lambda r: r['A2']['target_risk_fid'])):.3f}"
          f" -> target design {np.median(g(lambda r: r['A2']['target_risk_tmr'])):.3f}")

    print("\nA3  dense i.i.d. Gaussian corruption (||M||_F^2, lower is better)")
    for eta in ("0.05", "0.1", "0.2"):
        print(f"  eta={eta}: ratio to GLS "
              f"{np.median(g(lambda r: r['A3'][eta] / r['A3']['gls'])):.3f}")

    print("\nA5  threat-model mismatch (risk / GLS at the same attack budget)")
    for kd in (1, 2, 3):
        row = [np.median(g(lambda r: r["A5"][f"design{kd}"][f"attack{ka}"]
                           / r["A5"]["gls"][f"attack{ka}"])) for ka in (1, 2, 3)]
        print(f"  design k={kd}: " + "  ".join(f"attack k={ka}: {v:.3f}"
                                               for ka, v in zip((1, 2, 3), row)))

    print("\nA6  scalar-weight restriction (risk / GLS)")
    print(f"  full influence design {np.median(g(lambda r: r['A6']['risk_tmr'] / r['A6']['risk_gls'])):.3f}"
          f"   best scalar weights {np.median(g(lambda r: r['A6']['risk_wls'] / r['A6']['risk_gls'])):.3f}")
    print(f"  relative distance from TMR optimum to the weighted-LS manifold: "
          f"{np.median(g(lambda r: r['A6']['min_dist_to_wls_manifold'] / r['A6']['norm_tmr'])):.3f}")
