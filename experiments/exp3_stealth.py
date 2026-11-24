"""Experiment 3 -- detectability of the corruption from residuals.

A defender who only sees the fiducial residuals must decide which measurement,
if any, was tampered with.  We sweep the corruption budget ``rho`` and, at each
value, measure

  * the target displacement the optimal corruption achieves,
  * the increase in fiducial registration error (FRE) it causes,
  * the AUC of the best possible residual-based detector, i.e. the probability
    that the corrupted fiducial has a larger residual than a clean one.

The regime that matters for TMR is the one where the AUC is close to chance and
the target displacement is already clinically relevant.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import Instance
from tmr.geometry import (random_rotation, stack_jacobians, target_jacobian)
from tmr.estimators import kabsch, gls_influence, tmr_influence, \
    make_fixed_design_estimator
from tmr.risk import worst_case_perturbation, sparse_risk_exact

OUT = os.path.join(ROOT, "results", "exp3_stealth.json")
RHOS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
SIGMA = 0.35
N_MC = 400


def one(trial):
    rng = np.random.default_rng(2000 + trial)
    n = 7
    P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
    d = rng.standard_normal(3)
    x = 80.0 * d / np.linalg.norm(d)
    Sigma = SIGMA ** 2 * np.eye(3 * n)
    Rt, tt = random_rotation(rng), rng.standard_normal(3) * 20
    A = stack_jacobians(Rt, P)
    G = target_jacobian(Rt, x)
    M_gls = gls_influence(A, G, Sigma)

    out = []
    for rho in RHOS:
        rv = np.full(n, rho)
        a_star = worst_case_perturbation(M_gls, rv, 1)
        i_bad = int(np.argmax(np.linalg.norm(a_star, axis=1)))
        disp_pred = sparse_risk_exact(M_gls, rv, 1)

        fre_clean, fre_att, r_bad, r_good, disp = [], [], [], [], []
        for _ in range(N_MC):
            e = rng.normal(0, SIGMA, size=(n, 3))
            Q = P @ Rt.T + tt + e
            T0 = kabsch(P, Q)
            x0 = T0.R @ x + T0.t
            res0 = np.linalg.norm(P @ T0.R.T + T0.t - Q, axis=1)
            fre_clean.append(np.sqrt((res0 ** 2).mean()))

            Qa = Q + a_star
            T1 = kabsch(P, Qa)
            res1 = np.linalg.norm(P @ T1.R.T + T1.t - Qa, axis=1)
            fre_att.append(np.sqrt((res1 ** 2).mean()))
            disp.append(np.linalg.norm(T1.R @ x + T1.t - x0))
            r_bad.append(res1[i_bad])
            r_good.append(np.delete(res1, i_bad))

        r_bad = np.array(r_bad)
        r_good = np.concatenate(r_good)
        auc = float((r_bad[:, None] > r_good[None, :]).mean()
                    + 0.5 * (r_bad[:, None] == r_good[None, :]).mean())
        out.append(dict(rho=rho, auc=auc,
                        disp_mean=float(np.mean(disp)),
                        disp_pred=float(disp_pred),
                        fre_clean=float(np.mean(fre_clean)),
                        fre_att=float(np.mean(fre_att)),
                        fre_ratio=float(np.mean(fre_att) / np.mean(fre_clean))))
    return dict(trial=trial, sweep=out)


if __name__ == "__main__":
    with Pool(12) as pool:
        recs = list(pool.imap_unordered(one, range(48)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)
    print(f'{"rho(mm)":>8} {"AUC":>7} {"FRE clean":>10} {"FRE attacked":>13} '
          f'{"target disp(mm)":>16}')
    for j, rho in enumerate(RHOS):
        auc = np.mean([r["sweep"][j]["auc"] for r in recs])
        fc = np.mean([r["sweep"][j]["fre_clean"] for r in recs])
        fa = np.mean([r["sweep"][j]["fre_att"] for r in recs])
        dm = np.mean([r["sweep"][j]["disp_mean"] for r in recs])
        print(f"{rho:8.2f} {auc:7.3f} {fc:10.3f} {fa:13.3f} {dm:16.3f}")
