"""Experiment 5 -- transpedicular screw placement on a real L1 vertebra.

Geometry: L1 node cloud of the finite-element lumbar spine model of Zenodo
record 10.5281/zenodo.3961334 (CC-BY-4.0).  Only the rigid geometry is used.

Fiducials are the nine bony landmarks a navigation probe can reach on the
exposed posterior elements.  The clinical quantities are

  * the displacement of the planned screw *tip* in the vertebral body,
  * the maximum displacement along the planned screw *axis* (trajectory loss),
  * the resulting loss of pedicle-wall clearance, i.e. how much of the safe
    corridor margin the corruption consumes.

Landmark localisation error is heteroscedastic but per-fiducial isotropic:
sharp process tips are localised less reliably than flat laminar surfaces.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import Instance, attack, clean_rmse
from tmr.anatomy import build_vertebra
from tmr.geometry import (stack_jacobians, target_jacobian, random_rotation,
                          RigidTransform)
from tmr.estimators import (kabsch, huber_irls, ransac_register, gnc_tls,
                            gls_influence, tmr_influence,
                            tmr_influence_trajectory, make_fixed_design_estimator,
                            make_K_estimator)
from tmr.risk import sparse_risk_exact, worst_case_perturbation

OUT = os.path.join(ROOT, "results", "exp5_vertebra.json")

# per-landmark localisation sigma (mm): sharp tips are harder to hit repeatably
SIGMA_BY_LANDMARK = {
    "SP_tip": 0.60, "TP_left": 0.60, "TP_right": 0.60,
    "SAP_left": 0.45, "SAP_right": 0.45,
    "IAP_left": 0.45, "IAP_right": 0.45,
    "LAM_left": 0.30, "LAM_right": 0.30,
}
ETAS = (0.05, 0.10, 0.20)


def vertebra_instance(rho=1.0, k=1, seed=0, side="left"):
    v = build_vertebra()
    names = list(SIGMA_BY_LANDMARK)
    P = v.fiducials(names)
    sig = np.array([SIGMA_BY_LANDMARK[nm] for nm in names])
    Sigma = np.diag(np.repeat(sig ** 2, 3))
    rng = np.random.default_rng(seed)
    inst = Instance(P, v.tip[side], Sigma, np.full(len(P), float(rho)), k,
                    random_rotation(rng), rng.standard_normal(3) * 30)
    return v, inst, sig, names


def one(args):
    rho, k, side = args
    v, inst, sig, names = vertebra_instance(rho=rho, k=k, side=side)
    n = inst.n
    A0 = stack_jacobians(np.eye(3), inst.P)
    G0 = target_jacobian(np.eye(3), inst.x)

    ests = {
        "LS": lambda Pp, Qp: kabsch(Pp, Qp),
        "GLS": make_fixed_design_estimator(
            inst.P, inst.x, gls_influence(A0, G0, inst.Sigma), inst.Sigma),
        "Huber": lambda Pp, Qp: huber_irls(Pp, Qp),
        "RANSAC": lambda Pp, Qp: ransac_register(Pp, Qp,
                                                 thresh=2.5 * float(sig.mean())),
        "GNC-TLS": lambda Pp, Qp: gnc_tls(Pp, Qp,
                                          barc2=(3 * float(sig.mean())) ** 2),
    }
    predicted = {"GLS": sparse_risk_exact(gls_influence(A0, G0, inst.Sigma),
                                          inst.rho, k)}
    for eta in ETAS:
        r = tmr_influence(A0, G0, inst.Sigma, inst.rho, k=k, eta=eta)
        ests[f"TMR-{eta:g}"] = make_fixed_design_estimator(
            inst.P, inst.x, r.M, inst.Sigma)
        predicted[f"TMR-{eta:g}"] = r.risk

    # trajectory design (maximum displacement along the screw axis)
    traj = v.trajectory(side, 7)
    Gs = [target_jacobian(np.eye(3), xx) for xx in traj]
    Si = np.linalg.inv(inst.Sigma)
    K_gls = np.linalg.solve(A0.T @ Si @ A0, A0.T @ Si)
    predicted["GLS-traj"] = max(sparse_risk_exact(G @ K_gls, inst.rho, k)
                                for G in Gs)
    rt = tmr_influence_trajectory(A0, Gs, inst.Sigma, inst.rho, k=k, eta=0.10)
    ests["TMR-traj-0.1"] = make_K_estimator(inst.P, inst.x, rt["K"], inst.Sigma)
    predicted["TMR-traj-0.1"] = rt["risk"]

    rng = np.random.default_rng(11)
    Qs = [inst.clean_measurements(rng) for _ in range(2)]
    warm = worst_case_perturbation(
        gls_influence(stack_jacobians(inst.R_true, inst.P),
                      target_jacobian(inst.R_true, inst.x), inst.Sigma),
        inst.rho, k)

    res = {}
    for name, est in ests.items():
        c, _ = clean_rmse(est, inst, n_mc=120, seed=3)
        advs, traj_advs = [], []
        for j, Q in enumerate(Qs):
            a_val, a_star = attack(est, inst.P, Q, inst.x, inst.rho, k,
                                   warm=warm, seed=j)
            advs.append(a_val)
            # maximum displacement along the whole planned axis under the same
            # search, evaluated for the trajectory loss
            T0 = est(inst.P, Q)
            # the maximum-displacement point of a straight axis is an endpoint,
            # so attacking the two endpoints bounds the trajectory loss
            best = 0.0
            for probe in (traj[0], traj[-1]):
                v_t, a_t = attack(est, inst.P, Q, probe, inst.rho, k,
                                  warm=warm, seed=100 + j)
                T1 = est(inst.P, Q + a_t)
                d = np.max(np.linalg.norm(
                    traj @ T1.R.T + T1.t - (traj @ T0.R.T + T0.t), axis=1))
                best = max(best, d)
            traj_advs.append(best)
        res[name] = dict(clean_rmse=c, adv_mean=float(np.mean(advs)),
                         adv_max=float(np.max(advs)),
                         traj_adv_mean=float(np.mean(traj_advs)),
                         traj_adv_max=float(np.max(traj_advs)))
    return dict(rho=rho, k=k, side=side, n=n, landmarks=names,
                sigma=list(map(float, sig)), predicted=predicted, results=res)


if __name__ == "__main__":
    jobs = [(rho, k, side) for rho in (1.0, 2.0) for k in (1, 2)
            for side in ("left", "right")]
    with Pool(5) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            recs.append(r)
            print(f"  {i + 1}/{len(jobs)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(recs, open(OUT, "w"))
    print("wrote", OUT)
    for k in (1, 2):
        sub = [r for r in recs if r["k"] == k]
        print(f"\n=== L1 vertebra, k={k}, per mm of corruption budget ===")
        print(f'{"estimator":14s} {"clean RMSE":>11s} {"tip disp/rho":>13s} '
              f'{"axis disp/rho":>14s}')
        for nm in sub[0]["results"]:
            c = np.mean([r["results"][nm]["clean_rmse"] for r in sub])
            a = np.mean([r["results"][nm]["adv_mean"] / r["rho"] for r in sub])
            t = np.mean([r["results"][nm]["traj_adv_mean"] / r["rho"]
                         for r in sub])
            print(f"{nm:14s} {c:11.4f} {a:13.4f} {t:14.4f}")
