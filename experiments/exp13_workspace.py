"""Experiment 13 -- a workspace of targets, not a single prescribed point.

A robot rarely acts at exactly one point. It operates over a region: a set of
grasp or drill sites, a swept trajectory, or a volume of interest. The design
problem generalises directly, because the risk is a maximum over the target set,

    R_k(K; X) = max_{x in X} max_{a in A(k,rho)} || G_x K a || ,

and the cutting-plane scheme simply indexes its cuts by (target, direction)
pairs instead of by direction alone.

This experiment asks two questions:

  (a) does designing for the whole workspace still beat Gauss-Markov over that
      workspace, or does the guarantee wash out as the region grows; and
  (b) how much is lost by designing for a single representative target -- the
      centre of the region -- and then operating everywhere in it?
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
from tmr.estimators import (gls_influence, tmr_influence,
                            tmr_influence_multitarget)
from tmr.risk import sparse_risk_exact, fibonacci_sphere

OUT = os.path.join(ROOT, "results", "exp13_workspace.json")
RADII = [0.0, 10.0, 20.0, 40.0, 80.0]          # workspace radius in mm
N_TARGET = 9                                    # targets sampled in the region


def workspace(centre, radius, rng):
    """Centre plus a quasi-uniform shell of targets at the given radius."""
    if radius <= 0:
        return np.array([centre])
    dirs = fibonacci_sphere(N_TARGET - 1)
    return np.vstack([centre, centre + radius * dirs])


def one(args):
    trial, radius, k = args
    rng = np.random.default_rng(60000 + trial)
    n = 7
    P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
    d = rng.standard_normal(3)
    centre = 80.0 * d / np.linalg.norm(d)
    X = workspace(centre, radius, rng)

    sig = 0.35 * np.exp(0.35 * rng.standard_normal(n))
    Sigma = np.diag(np.repeat(sig ** 2, 3))
    A = stack_jacobians(np.eye(3), P)
    Gs = [target_jacobian(np.eye(3), x) for x in X]
    rho = np.ones(n)
    Si = np.linalg.inv(Sigma)
    K_gls = np.linalg.solve(A.T @ Si @ A, A.T @ Si)

    # worst case over the whole workspace, for three designs
    r_gls = max(sparse_risk_exact(G @ K_gls, rho, k) for G in Gs)

    # (i) designed for the centre target only, then operated over the region
    M_pt = tmr_influence(A, Gs[0], Sigma, rho, k=k, eta=0.10).M
    Gp = np.linalg.pinv(Gs[0])
    K_pt = Gp @ M_pt + (np.eye(6) - Gp @ Gs[0]) @ K_gls
    r_pt = max(sparse_risk_exact(G @ K_pt, rho, k) for G in Gs)

    # (ii) designed for the whole workspace
    res = tmr_influence_multitarget(A, Gs, Sigma, rho, k=k, eta=0.10)
    r_ws = res["risk"]

    return dict(trial=trial, radius=radius, k=k, n_targets=len(X),
                gls=r_gls, point_design=r_pt, workspace_design=r_ws,
                gap=res["gap"], var_ratio=res["variance"] / res["V0"])


if __name__ == "__main__":
    jobs = [(t, r, k) for t in range(16) for r in RADII for k in (1, 2)]
    with Pool(5) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            recs.append(r)
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)
    print(f'{"radius":>7} {"k":>3} {"centre-only/GLS":>16} {"workspace/GLS":>14} '
          f'{"cost of point design":>21}')
    for rad in RADII:
        for k in (1, 2):
            sub = [r for r in recs if r["radius"] == rad and r["k"] == k]
            if not sub:
                continue
            pt = np.median([r["point_design"] / r["gls"] for r in sub])
            ws = np.median([r["workspace_design"] / r["gls"] for r in sub])
            pen = np.median([r["point_design"] / r["workspace_design"]
                             for r in sub])
            print(f"{rad:7.0f} {k:>3} {pt:16.3f} {ws:14.3f} {pen:20.3f}x")
