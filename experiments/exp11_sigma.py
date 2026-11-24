"""Experiment 11 -- what happens when the nominal covariance is wrong?

The clean-efficiency constraint is stated in terms of an assumed Sigma_hat.  In
deployment Sigma_hat is always wrong.  We design both Gauss-Markov and TMR under
Sigma_hat and evaluate under the true Sigma, asking two questions:

  (a) is the efficiency promise kept -- does the realised clean variance stay
      within (1 + eta) of the *true* Gauss-Markov optimum?
  (b) does the robustness advantage survive?

The worst-case risk itself does not depend on Sigma, so (b) isolates the effect
of misspecification on the design rather than on the metric.
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
from tmr.estimators import gls_influence, tmr_influence
from tmr.risk import sparse_risk_exact

OUT = os.path.join(ROOT, "results", "exp11_sigma.json")
DELTAS = [0.0, 0.2, 0.4, 0.8, 1.2]


def one(args):
    trial, delta, k = args
    rng = np.random.default_rng(30000 + trial)
    n = 8
    P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
    d = rng.standard_normal(3)
    x = 80.0 * d / np.linalg.norm(d)
    sig_true = 0.35 * np.exp(0.35 * rng.standard_normal(n))
    S_true = np.diag(np.repeat(sig_true ** 2, 3))
    rng2 = np.random.default_rng(40000 + trial + int(1000 * delta))
    sig_hat = sig_true * np.exp(delta * rng2.standard_normal(n))
    S_hat = np.diag(np.repeat(sig_hat ** 2, 3))

    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), x)
    rho = np.ones(n)
    var = lambda M: float(np.trace(M @ S_true @ M.T))

    M_gls_true = gls_influence(A, G, S_true)          # oracle reference
    V0_true = var(M_gls_true)
    M_gls_hat = gls_influence(A, G, S_hat)            # GLS with the wrong Sigma
    M_tmr_hat = tmr_influence(A, G, S_hat, rho, k=k, eta=0.10).M
    M_tmr_true = tmr_influence(A, G, S_true, rho, k=k, eta=0.10).M

    return dict(trial=trial, delta=delta, k=k,
                risk_gls_hat=sparse_risk_exact(M_gls_hat, rho, k),
                risk_tmr_hat=sparse_risk_exact(M_tmr_hat, rho, k),
                risk_gls_true=sparse_risk_exact(M_gls_true, rho, k),
                risk_tmr_true=sparse_risk_exact(M_tmr_true, rho, k),
                var_gls_hat=var(M_gls_hat) / V0_true,
                var_tmr_hat=var(M_tmr_hat) / V0_true,
                var_tmr_true=var(M_tmr_true) / V0_true)


if __name__ == "__main__":
    jobs = [(t, d, k) for t in range(24) for d in DELTAS for k in (1, 2)]
    with Pool(5) as pool:
        recs = list(pool.imap_unordered(one, jobs))
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)
    print(f'{"delta":>6} {"k":>3} {"TMR/GLS risk":>13} {"TMR var/oracle":>15} '
          f'{"GLS var/oracle":>15} {"promise kept":>13}')
    for d in DELTAS:
        for k in (1, 2):
            sub = [r for r in recs if r["delta"] == d and r["k"] == k]
            rat = np.median([r["risk_tmr_hat"] / r["risk_gls_hat"] for r in sub])
            vt = np.median([r["var_tmr_hat"] for r in sub])
            vg = np.median([r["var_gls_hat"] for r in sub])
            kept = np.mean([r["var_tmr_hat"] <= 1.10 * r["var_gls_hat"] + 1e-9
                            for r in sub])
            print(f"{d:6.1f} {k:>3} {rat:13.3f} {vt:15.3f} {vg:15.3f} {kept:12.0%}")
