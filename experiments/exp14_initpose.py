"""Experiment 14 -- does the relinearised solver depend on where it starts?

The design is built from Jacobians evaluated at a nominal pose, which raises the
usual question for any SE(3) estimator: what happens under a poor initial
guess?

For paired-point registration with known correspondences the question has a
structural answer. The initialising fit is orthogonal Procrustes, which is a
closed-form global optimum, so the "initialisation" is not a free parameter at
all. We nonetheless force the iteration to start from deliberately wrong poses,
up to a full half-turn and 100 mm of translation, and measure the disagreement
with the Procrustes-initialised result.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.geometry import (random_rotation, expm_so3, stack_jacobians,
                          target_jacobian, RigidTransform)
from tmr.estimators import (tmr_influence, complete_influence_to_K,
                            make_fixed_design_estimator, transport_influence)

OUT = os.path.join(ROOT, "results", "exp14_initpose.json")
PERTURB = [(0, 0), (5, 5), (10, 10), (20, 20), (45, 50), (90, 100), (170, 200)]


def run_from(P, Q, x, M0, Sigma, T_init, n_iter=60, tol=1e-14):
    """Relinearised TMR started from an arbitrary pose."""
    n = len(P)
    Si = np.linalg.inv(Sigma)
    T = T_init
    for it in range(n_iter):
        A = stack_jacobians(T.R, P)
        G = target_jacobian(T.R, x)
        M = transport_influence(M0, T.R)
        K0 = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
        K = complete_influence_to_K(M, G, K0)
        xi = K @ (Q - (P @ T.R.T + T.t)).reshape(-1)
        T = RigidTransform(expm_so3(xi[:3]) @ T.R, T.t + xi[3:])
        if np.linalg.norm(xi) < tol:
            break
    return T, it + 1


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    recs = []
    for trial in range(12):
        n = 7
        P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
        d = rng.standard_normal(3)
        x = 80.0 * d / np.linalg.norm(d)
        sig = 0.35 * np.exp(0.35 * rng.standard_normal(n))
        Sigma = np.diag(np.repeat(sig ** 2, 3))
        Rs, ts = random_rotation(rng), rng.standard_normal(3) * 20
        Q = P @ Rs.T + ts + rng.normal(0, 1, (n, 3)) * sig[:, None]

        A0 = stack_jacobians(np.eye(3), P)
        G0 = target_jacobian(np.eye(3), x)
        M0 = tmr_influence(A0, G0, Sigma, np.ones(n), k=1, eta=0.10).M

        ref_est = make_fixed_design_estimator(P, x, M0, Sigma)
        T_ref = ref_est(P, Q)
        x_ref = T_ref.R @ x + T_ref.t

        for ang, tr in PERTURB:
            w = rng.standard_normal(3)
            w = w / np.linalg.norm(w) * np.radians(ang)
            dt = rng.standard_normal(3)
            dt = dt / np.linalg.norm(dt) * tr
            T_init = RigidTransform(expm_so3(w) @ Rs, ts + dt)
            T, iters = run_from(P, Q, x, M0, Sigma, T_init)
            recs.append(dict(trial=trial, angle_deg=ang, trans_mm=tr,
                             iters=iters,
                             target_diff=float(np.linalg.norm(
                                 (T.R @ x + T.t) - x_ref)),
                             rot_diff_deg=float(np.degrees(np.linalg.norm(
                                 np.array([T.R[2, 1] - T.R[1, 2],
                                           T.R[0, 2] - T.R[2, 0],
                                           T.R[1, 0] - T.R[0, 1]])
                                 - np.array([T_ref.R[2, 1] - T_ref.R[1, 2],
                                             T_ref.R[0, 2] - T_ref.R[2, 0],
                                             T_ref.R[1, 0] - T_ref.R[0, 1]]))
                                 / 2))))
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)
    print(f'{"init rot":>9} {"init trans":>11} {"target diff (mm)":>18} '
          f'{"iters":>7}')
    for ang, tr in PERTURB:
        sub = [r for r in recs if r["angle_deg"] == ang]
        print(f"{ang:6d} deg {tr:8d} mm "
              f"{max(r['target_diff'] for r in sub):18.2e} "
              f"{max(r['iters'] for r in sub):7d}")
