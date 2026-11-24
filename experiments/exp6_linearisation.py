"""Validity regime of the first-order target-displacement model."""
from __future__ import annotations
import os
import json, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.geometry import random_rotation, stack_jacobians, target_jacobian
from tmr.estimators import gls_influence, kabsch

OUT = os.path.join(ROOT, "results", "exp6_linearisation.json")
RHOS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

if __name__ == "__main__":
    rng = np.random.default_rng(13)
    rows = []
    for rho in RHOS:
        rel = []
        for _ in range(60):
            P = rng.uniform(-50, 50, size=(6, 3))
            Rs, ts = random_rotation(rng), rng.standard_normal(3) * 10
            Q = P @ Rs.T + ts
            x = P.mean(0) + np.array([80.0, 0.0, 0.0])
            A = stack_jacobians(Rs, P)
            G = target_jacobian(Rs, x)
            M = gls_influence(A, G, np.eye(18))
            a = rng.standard_normal((6, 3))
            a = rho * a / np.linalg.norm(a, axis=1, keepdims=True)
            T0, T1 = kabsch(P, Q), kabsch(P, Q + a)
            dt = (T1.R @ x + T1.t) - (T0.R @ x + T0.t)
            rel.append(float(np.linalg.norm(dt - M @ a.reshape(-1))
                             / np.linalg.norm(dt)))
        rows.append(dict(rho=rho, median=float(np.median(rel)),
                         p95=float(np.percentile(rel, 95)),
                         max=float(np.max(rel))))
        print(f"rho={rho:5.2f}  median {rows[-1]['median']:.2e}  "
              f"p95 {rows[-1]['p95']:.2e}")
    r = np.array([[x["rho"], x["median"]] for x in rows])
    slope = float(np.polyfit(np.log(r[:, 0]), np.log(r[:, 1]), 1)[0])
    json.dump(dict(rows=rows, loglog_slope=slope, n_per_rho=60), open(OUT, "w"))
    print(f"log-log slope {slope:.3f}  (1 => O(rho^2) remainder)")
