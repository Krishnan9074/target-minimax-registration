"""Experiment 2 -- direct (non-linearised) attacks on every estimator.

Each estimator is attacked as a black box: we search over admissible sparse
corruptions ``a in A(k, rho)`` for the one that moves the *transformed target*
furthest.  This is the only fair comparison against residual-robust estimators
(Huber, RANSAC, GNC-TLS), whose influence operator is data dependent.

The noise model is heteroscedastic but per-fiducial isotropic, so GLS differs
from LS while the TMR design remains exactly transportable across poses.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import json
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import Instance, attack, clean_rmse
from tmr.geometry import (random_rotation, stack_jacobians, target_jacobian,
                          RigidTransform)
from tmr.estimators import (kabsch, huber_irls, ransac_register, gnc_tls,
                            gls_influence, tmr_influence,
                            make_fixed_design_estimator)
from tmr.risk import sparse_risk_exact, worst_case_perturbation

OUT = os.path.join(ROOT, "results", "exp2_nonlinear.json")
ETAS = (0.05, 0.10, 0.20)
N_REAL = 3


def build(trial, n=7, target_dist=80.0, rho=1.0, k=1):
    rng = np.random.default_rng(1000 + trial)
    P = rng.uniform(-60, 60, size=(n, 3))
    P -= P.mean(0)
    d = rng.standard_normal(3)
    x = target_dist * d / np.linalg.norm(d)
    sig = 0.35 * np.exp(0.35 * rng.standard_normal(n))     # 0.2 - 0.7 mm
    Sigma = np.diag(np.repeat(sig ** 2, 3))
    return Instance(P, x, Sigma, np.full(n, float(rho)), k,
                    random_rotation(rng), rng.standard_normal(3) * 20.0), sig


_DESIGN_CACHE = {}


def estimators_for(inst, sig, cache_key=None):
    n = inst.n
    ests = {
        "LS": lambda Pp, Qp: kabsch(Pp, Qp),
        "GLS": make_fixed_design_estimator(
            inst.P, inst.x,
            gls_influence(stack_jacobians(np.eye(3), inst.P),
                          target_jacobian(np.eye(3), inst.x), inst.Sigma),
            inst.Sigma),
        "Huber": lambda Pp, Qp: huber_irls(Pp, Qp),
        "RANSAC": lambda Pp, Qp: ransac_register(Pp, Qp,
                                                 thresh=2.5 * float(sig.mean())),
        "GNC-TLS": lambda Pp, Qp: gnc_tls(Pp, Qp,
                                          barc2=(3.0 * float(sig.mean())) ** 2),
    }
    A0 = stack_jacobians(np.eye(3), inst.P)
    G0 = target_jacobian(np.eye(3), inst.x)
    designs = {}
    for eta in ETAS:
        ck = (cache_key, eta)
        if ck not in _DESIGN_CACHE:
            # for a uniform budget the design is scale invariant in rho, so it
            # is solved once per (geometry, k, eta) at rho = 1 and reused; the
            # risk is rescaled at the point of use, never stored rescaled
            _DESIGN_CACHE[ck] = tmr_influence(A0, G0, inst.Sigma, np.ones(n),
                                              k=inst.k, eta=eta)
        r = copy.copy(_DESIGN_CACHE[ck])
        r.risk = _DESIGN_CACHE[ck].risk * float(inst.rho[0])
        designs[eta] = r
        ests[f"TMR-{eta:g}"] = make_fixed_design_estimator(
            inst.P, inst.x, r.M, inst.Sigma)
    return ests, designs


def one(args):
    trial, rho, k = args
    inst, sig = build(trial, rho=rho, k=k)
    ests, designs = estimators_for(inst, sig, cache_key=(trial, k))
    rng = np.random.default_rng(7000 + trial)

    A = stack_jacobians(inst.R_true, inst.P)
    G = target_jacobian(inst.R_true, inst.x)
    M_gls = gls_influence(A, G, inst.Sigma)
    warm_gls = worst_case_perturbation(M_gls, inst.rho, k)

    rec = dict(trial=trial, rho=rho, k=k, n=inst.n,
               target_dist=float(np.linalg.norm(inst.x)),
               predicted={"GLS": sparse_risk_exact(M_gls, inst.rho, k)},
               results={})
    for eta, r in designs.items():
        rec["predicted"][f"TMR-{eta:g}"] = r.risk

    Qs = [inst.clean_measurements(rng) for _ in range(N_REAL)]
    for name, est in ests.items():
        c_rmse, _ = clean_rmse(est, inst, n_mc=120, seed=trial)
        advs = []
        for j, Q in enumerate(Qs):
            v, _ = attack(est, inst.P, Q, inst.x, inst.rho, k,
                          warm=warm_gls, seed=trial * 17 + j)
            advs.append(v)
        rec["results"][name] = dict(clean_rmse=c_rmse,
                                    adv_mean=float(np.mean(advs)),
                                    adv_max=float(np.max(advs)),
                                    adv=list(map(float, advs)))
    return rec


if __name__ == "__main__":
    jobs = [(t, rho, k) for t in range(8) for rho in (0.5, 1.0, 2.0)
            for k in (1, 2)]
    print(f"{len(jobs)} configurations")
    with Pool(5) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            recs.append(r)
            print(f"  {i + 1}/{len(jobs)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)
