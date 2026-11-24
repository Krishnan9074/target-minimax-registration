"""Experiment 16 -- anisotropic localisation noise.

Every head-to-head result so far uses per-fiducial *isotropic* heteroscedastic
noise.  Optical and imaging localisers are anisotropic: the error along the
camera viewing axis (or the slice direction) is much larger than in-plane.
This experiment repeats the direct black-box comparison of Experiment 2 with

    Sigma_i = sigma_i^2 * C,   C = R_cam diag(1, 1, a) R_cam^T,  tr(C) = 3,

one shared camera axis per geometry and a variance ratio ``a`` swept from 1
(isotropic) to 57.  The trace is fixed so every level of anisotropy carries the
same total noise power; only its distribution over directions changes.

Two things change relative to Experiment 2.

1. The fixed-design shortcut ``transport_influence`` is exact only when Sigma
   commutes with ``I_n (x) R`` (per-fiducial isotropic noise).  Under
   anisotropic noise a design transported from the identity frame violates the
   clean-variance budget by a large factor (recorded here as
   ``shortcut_var_ratio``).  Every linear design is therefore made *at the
   nominal pose* and transported only by the small relinearisation increment.
2. Gauss--Markov is the anisotropic optimum (Danilchenko & Fitzpatrick 2011),
   recomputed exactly at every relinearisation.

The question: does "unweighted least squares is
adversarially safer than Gauss--Markov" survive when the weighting is
direction dependent, and does the TMR advantage survive?

The second half of the script (``factorial``) repeats the objective x design
class factorial of Experiment 7 under the same anisotropy, in closed form, and
adds a dense closed-form sweep of LS versus Gauss--Markov to locate any
crossover.
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
from tmr.geometry import (random_rotation, stack_jacobians, target_jacobian,
                          RigidTransform)
from tmr.estimators import (kabsch, huber_irls, ransac_register, gnc_tls,
                            gls_influence, tmr_influence, transport_influence,
                            complete_influence_to_K, expm_so3,
                            average_influence, scalar_weight_design)
from tmr.risk import sparse_risk_exact, worst_case_perturbation, average_risk

OUT = os.path.join(ROOT, "results", "exp16_aniso.json")
OUT_F = os.path.join(ROOT, "results", "exp16_aniso_factorial.json")
ETAS = (0.05, 0.10, 0.20)
LEVELS = (1.0, 4.0, 16.0, 57.0)
SWEEP = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 57.0, 100.0)
N_REAL = 3
ETA = 0.10


def aniso_block(a, R_cam):
    C = R_cam @ np.diag([1.0, 1.0, float(a)]) @ R_cam.T
    return C * (3.0 / np.trace(C))


def build(trial, a, k, n=7, target_dist=80.0, rho=1.0):
    rng = np.random.default_rng(1000 + trial)          # same geometries as exp2
    P = rng.uniform(-60, 60, size=(n, 3))
    P -= P.mean(0)
    d = rng.standard_normal(3)
    x = target_dist * d / np.linalg.norm(d)
    sig = 0.35 * np.exp(0.35 * rng.standard_normal(n))     # 0.2 - 0.7 mm
    R_true, t_true = random_rotation(rng), rng.standard_normal(3) * 20.0
    R_cam = random_rotation(np.random.default_rng(50000 + trial))
    C = aniso_block(a, R_cam)
    Sigma = np.zeros((3 * n, 3 * n))
    for i in range(n):
        Sigma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = sig[i] ** 2 * C
    return Instance(P, x, Sigma, np.full(n, float(rho)), k, R_true, t_true), sig


def pose_design_estimator(P, x, M_d, R_d, Sigma, n_iter=8, tol=1e-12):
    """Fixed influence design made at pose ``R_d``, transported by the small
    increment ``T.R R_d^T`` at each relinearisation."""
    Si = np.linalg.inv(Sigma)

    def est(Pp, Qp):
        T = kabsch(Pp, Qp)
        for _ in range(n_iter):
            A = stack_jacobians(T.R, Pp)
            G = target_jacobian(T.R, x)
            M = transport_influence(M_d, T.R @ R_d.T)
            K0 = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
            K = complete_influence_to_K(M, G, K0)
            b = (Qp - (Pp @ T.R.T + T.t)).reshape(-1)
            xi = K @ b
            T = RigidTransform(expm_so3(xi[:3]) @ T.R, T.t + xi[3:])
            if np.linalg.norm(xi) < tol:
                break
        return T
    return est


def gls_estimator(P, x, Sigma, n_iter=8, tol=1e-12):
    """Exact anisotropic Gauss--Markov, recomputed at every relinearisation."""
    Si = np.linalg.inv(Sigma)

    def est(Pp, Qp):
        T = kabsch(Pp, Qp)
        for _ in range(n_iter):
            A = stack_jacobians(T.R, Pp)
            K = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
            b = (Qp - (Pp @ T.R.T + T.t)).reshape(-1)
            xi = K @ b
            T = RigidTransform(expm_so3(xi[:3]) @ T.R, T.t + xi[3:])
            if np.linalg.norm(xi) < tol:
                break
        return T
    return est


def one(args):
    trial, a, k = args
    inst, sig = build(trial, a, k)
    n, P, x, S = inst.n, inst.P, inst.x, inst.Sigma
    R_d = inst.R_true
    A1, G1 = stack_jacobians(R_d, P), target_jacobian(R_d, x)
    A0, G0 = stack_jacobians(np.eye(3), P), target_jacobian(np.eye(3), x)
    rho = inst.rho

    M_gls = gls_influence(A1, G1, S)
    V0 = float(np.trace(M_gls @ S @ M_gls.T))
    M_ls = G1 @ np.linalg.solve(A1.T @ A1, A1.T)

    ests = {
        "LS": lambda Pp, Qp: kabsch(Pp, Qp),
        "GLS": gls_estimator(P, x, S),
        "Huber": lambda Pp, Qp: huber_irls(Pp, Qp),
        "RANSAC": lambda Pp, Qp: ransac_register(Pp, Qp,
                                                 thresh=2.5 * float(sig.mean())),
        "GNC-TLS": lambda Pp, Qp: gnc_tls(Pp, Qp,
                                          barc2=(3.0 * float(sig.mean())) ** 2),
    }
    predicted = {"GLS": sparse_risk_exact(M_gls, rho, k),
                 "LS": sparse_risk_exact(M_ls, rho, k)}
    shortcut = {}
    for eta in ETAS:
        r = tmr_influence(A1, G1, S, rho, k=k, eta=eta)
        ests[f"TMR-{eta:g}"] = pose_design_estimator(P, x, r.M, R_d, S)
        predicted[f"TMR-{eta:g}"] = r.risk
        if eta == ETA:
            # the invalid shortcut: design at the identity frame, transport
            r0 = tmr_influence(A0, G0, S, rho, k=k, eta=eta)
            Mt = transport_influence(r0.M, R_d)
            shortcut = dict(var_ratio=float(np.trace(Mt @ S @ Mt.T) / V0),
                            risk=sparse_risk_exact(Mt, rho, k),
                            var_ratio_pose=float(r.variance / V0))

    rng = np.random.default_rng(7000 + trial)
    warm = worst_case_perturbation(M_gls, rho, k)
    rec = dict(trial=trial, a=a, k=k, n=n, rho=float(rho[0]),
               target_dist=float(np.linalg.norm(x)),
               predicted=predicted, shortcut=shortcut, results={})
    Qs = [inst.clean_measurements(rng) for _ in range(N_REAL)]
    for name, est in ests.items():
        c_rmse, _ = clean_rmse(est, inst, n_mc=120, seed=trial)
        advs = []
        for j, Q in enumerate(Qs):
            v, _ = attack(est, P, Q, x, rho, k, warm=warm, seed=trial * 17 + j)
            advs.append(v)
        rec["results"][name] = dict(clean_rmse=c_rmse,
                                    adv_mean=float(np.mean(advs)),
                                    adv_max=float(np.max(advs)),
                                    adv=list(map(float, advs)))
    return rec


# --------------------------------------------------------------------------- #
#  Closed-form factorial and LS-vs-GLS sweep (Experiment 7 under anisotropy)
# --------------------------------------------------------------------------- #
def factorial_one(args):
    trial, a, k = args
    rng = np.random.default_rng(5000 + trial)          # same geometries as exp7
    n = 7
    P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
    d = rng.standard_normal(3)
    x = 80.0 * d / np.linalg.norm(d)
    sig = 0.35 * np.exp(0.35 * rng.standard_normal(n))
    R_cam = random_rotation(np.random.default_rng(60000 + trial))
    C = aniso_block(a, R_cam)
    Sigma = np.zeros((3 * n, 3 * n))
    for i in range(n):
        Sigma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = sig[i] ** 2 * C
    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), x)
    rho = np.ones(n)
    var = lambda M: float(np.trace(M @ Sigma @ M.T))
    designs = {"GLS": gls_influence(A, G, Sigma),
               "LS": G @ np.linalg.solve(A.T @ A, A.T)}
    if a in LEVELS:
        designs["learned-avg (scalar)"] = scalar_weight_design(
            A, G, Sigma, rho, k=k, eta=ETA, objective="average", seed=trial)[0]
        designs["avg (full)"] = average_influence(A, G, Sigma, rho, k=k,
                                                  eta=ETA)[0]
        designs["TMR-W (scalar)"] = scalar_weight_design(
            A, G, Sigma, rho, k=k, eta=ETA, objective="minimax", seed=trial)[0]
        designs["TMR (full)"] = tmr_influence(A, G, Sigma, rho, k=k, eta=ETA).M
    return dict(trial=trial, a=a, k=k, n=n,
                results={nm: dict(worst=sparse_risk_exact(M, rho, k),
                                  average=average_risk(M, rho, k), var=var(M))
                         for nm, M in designs.items()})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "factorial"):
        jobs = [(t, a, k) for t in range(24) for a in SWEEP for k in (1, 2)]
        print(f"factorial/sweep: {len(jobs)} configurations", flush=True)
        with Pool(6) as pool:
            recs = list(pool.imap_unordered(factorial_one, jobs))
        json.dump(recs, open(OUT_F, "w"))
        print("wrote", OUT_F, flush=True)
        for a in SWEEP:
            for k in (1, 2):
                sub = [r for r in recs if r["a"] == a and r["k"] == k]
                ls = np.median([r["results"]["LS"]["worst"]
                                / r["results"]["GLS"]["worst"] for r in sub])
                line = f"a={a:5.0f} k={k}  LS/GLS worst {ls:.3f}"
                if "TMR (full)" in sub[0]["results"]:
                    for nm in ("learned-avg (scalar)", "TMR-W (scalar)",
                               "TMR (full)"):
                        v = np.median([r["results"][nm]["worst"]
                                       / r["results"]["GLS"]["worst"] for r in sub])
                        line += f"  {nm} {v:.3f}"
                print(line, flush=True)
    if which in ("all", "attack"):
        jobs = [(t, a, k) for t in range(8) for a in LEVELS for k in (1, 2)]
        print(f"attack: {len(jobs)} configurations", flush=True)
        with Pool(6) as pool:
            recs = []
            for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
                recs.append(r)
                print(f"  {i + 1}/{len(jobs)}", flush=True)
        json.dump(recs, open(OUT, "w"))
        print("wrote", OUT, flush=True)
        for a in LEVELS:
            for k in (1, 2):
                sub = [r for r in recs if r["a"] == a and r["k"] == k]
                g = np.mean([r["results"]["GLS"]["adv_mean"] for r in sub])
                line = f"a={a:5.0f} k={k}  GLS {g:.3f}"
                for nm in ("LS", "Huber", "GNC-TLS", "TMR-0.1"):
                    v = np.mean([r["results"][nm]["adv_mean"] for r in sub])
                    line += f"  {nm} {v:.3f} ({v / g:.2f})"
                sv = np.median([r["shortcut"]["var_ratio"] for r in sub])
                print(line + f"  shortcut var/V0 {sv:.2f}", flush=True)
