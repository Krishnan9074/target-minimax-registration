"""Shared evaluation harness for the TMR experiments.

Everything is measured in millimetres.  Two quantities are reported for every
estimator:

``clean``  root-mean-square target error under nominal localisation noise only;
``adv``    worst-case target *displacement* caused by an admissible corruption,
           ``max_{a in A(k,rho)} || Ehat(Q + a)(x) - Ehat(Q)(x) ||``,
           obtained by directly attacking the estimator (no linearisation).
"""
from __future__ import annotations

import itertools
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tmr.geometry import (random_rotation, stack_jacobians, target_jacobian,
                          RigidTransform)
from tmr.estimators import (kabsch, weighted_kabsch, huber_irls,
                            ransac_register, gnc_tls, gls_influence,
                            tmr_influence, complete_influence_to_K, expm_so3)
from tmr.risk import sparse_risk_exact, worst_case_perturbation


# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    """One registration problem: fiducials, target, noise model, threat model."""
    P: np.ndarray                 # (n,3) fiducials in image coordinates
    x: np.ndarray                 # (3,)  surgical target
    Sigma: np.ndarray             # (3n,3n) nominal localisation covariance
    rho: np.ndarray               # (n,)  per-fiducial corruption budget
    k: int = 1
    R_true: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_true: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def n(self):
        return self.P.shape[0]

    def clean_measurements(self, rng):
        e = rng.multivariate_normal(np.zeros(3 * self.n), self.Sigma
                                    ).reshape(self.n, 3)
        return self.P @ self.R_true.T + self.t_true + e

    def noiseless(self):
        return self.P @ self.R_true.T + self.t_true

    def target_true(self):
        return self.R_true @ self.x + self.t_true


def random_instance(rng, n=7, spread=60.0, target_dist=80.0, sigma=0.35,
                    rho=1.0, k=1, anisotropy=0.0):
    P = rng.uniform(-spread, spread, size=(n, 3))
    P = P - P.mean(0)
    d = rng.standard_normal(3)
    x = target_dist * d / np.linalg.norm(d)
    Sigma = np.zeros((3 * n, 3 * n))
    for i in range(n):
        if anisotropy <= 0:
            C = np.eye(3)
        else:
            B = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(B)
            s = np.exp(anisotropy * rng.standard_normal(3))
            C = Q @ np.diag(s ** 2) @ Q.T
            C *= 3.0 / np.trace(C)
        Sigma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = sigma ** 2 * C
    return Instance(P, x, Sigma, np.full(n, float(rho)), k,
                    random_rotation(rng), rng.standard_normal(3) * 20.0)


# --------------------------------------------------------------------------- #
#  Estimator wrappers: every estimator maps (P, Q) -> RigidTransform
# --------------------------------------------------------------------------- #
def make_estimators(inst: Instance, etas=(0.05, 0.10, 0.20), tmr_k=None,
                    include_slow=True, tmr_kw=None):
    n, P, x = inst.n, inst.P, inst.x
    sig = float(np.sqrt(np.trace(inst.Sigma) / (3 * n)))
    Sinv_scale = inst.Sigma
    tmr_k = inst.k if tmr_k is None else tmr_k
    tmr_kw = tmr_kw or {}

    def linear_estimator(M_fn):
        """Build a pose estimator from a target-influence operator factory."""
        def est(Pp, Qp):
            T = kabsch(Pp, Qp)
            for _ in range(6):
                A = stack_jacobians(T.R, Pp)
                G = target_jacobian(T.R, x)
                M = M_fn(A, G)
                Si = np.linalg.inv(inst.Sigma)
                K0 = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
                K = complete_influence_to_K(M, G, K0)
                b = (Qp - (Pp @ T.R.T + T.t)).reshape(-1)
                xi = K @ b
                T = RigidTransform(expm_so3(xi[:3]) @ T.R, T.t + xi[3:])
                if np.linalg.norm(xi) < 1e-12:
                    break
            return T
        return est

    ests = {
        "LS (Kabsch)": lambda Pp, Qp: kabsch(Pp, Qp),
        "GLS": linear_estimator(
            lambda A, G: gls_influence(A, G, inst.Sigma)),
        "Huber IRLS": lambda Pp, Qp: huber_irls(Pp, Qp),
        "RANSAC": lambda Pp, Qp: ransac_register(Pp, Qp, thresh=2.5 * sig),
        "GNC-TLS": lambda Pp, Qp: gnc_tls(Pp, Qp, barc2=(3.0 * sig) ** 2),
    }
    if not include_slow:
        ests.pop("RANSAC")

    # TMR designs: the influence operator is precomputed at the identity-frame
    # linearisation and redesigned at every relinearisation step
    cache = {}

    def tmr_M(eta):
        def f(A, G):
            key = (eta, A.tobytes()[:64], G.tobytes()[:32])
            if key not in cache:
                cache[key] = tmr_influence(A, G, inst.Sigma, inst.rho,
                                           k=tmr_k, eta=eta, **tmr_kw).M
            return cache[key]
        return f

    for eta in etas:
        ests[f"TMR (eta={eta:g})"] = linear_estimator(tmr_M(eta))
    return ests


# --------------------------------------------------------------------------- #
#  Direct (non-linearised) worst-case attack
# --------------------------------------------------------------------------- #
def attack(est, P, Q, x, rho, k, n_restart=2, warm=None, seed=0,
           n_random=40, maxiter=160):
    """Maximise ||est(P, Q+a)(x) - est(P, Q)(x)|| over a in A(k, rho)."""
    n = P.shape[0]
    rho = np.broadcast_to(np.asarray(rho, float).ravel(), (n,))
    T0 = est(P, Q)
    x0 = T0.R @ x + T0.t
    rng = np.random.default_rng(seed)

    def disp(a):
        T = est(P, Q + a)
        return float(np.linalg.norm(T.R @ x + T.t - x0))

    def unpack(z, S):
        a = np.zeros((n, 3))
        for j, i in enumerate(S):
            v = z[3 * j:3 * j + 3]
            nv = np.linalg.norm(v)
            a[i] = rho[i] * v / nv if nv > 1.0 else rho[i] * v
        return a

    best_val, best_a = 0.0, np.zeros((n, 3))
    for S in itertools.combinations(range(n), int(min(k, n))):
        starts = []
        if warm is not None:
            w = np.zeros(3 * len(S))
            for j, i in enumerate(S):
                nv = np.linalg.norm(warm[i])
                if nv > 1e-12:
                    w[3 * j:3 * j + 3] = warm[i] / max(nv, 1e-12)
            if np.any(w):
                starts.append(w)
        # random probes on the boundary of the corruption set: robust
        # estimators are piecewise constant in ``a`` so a purely local search
        # can stall
        probes = rng.standard_normal((n_random, 3 * len(S)))
        probes /= np.linalg.norm(probes.reshape(n_random, len(S), 3),
                                 axis=2).repeat(3, axis=1)
        pv = np.array([disp(unpack(z, S)) for z in probes])
        best_val = max(best_val, float(pv.max()))
        if pv.max() > 0:
            j = int(np.argmax(pv))
            if pv[j] >= best_val - 1e-12:
                best_a = unpack(probes[j], S)
        starts += [probes[i] for i in np.argsort(pv)[::-1][:n_restart]]
        for z0 in starts:
            r = minimize(lambda z: -disp(unpack(z, S)), z0,
                         method="Nelder-Mead",
                         options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-8))
            if -r.fun > best_val:
                best_val, best_a = -float(r.fun), unpack(r.x, S)
    return best_val, best_a


def clean_rmse(est, inst: Instance, n_mc=200, seed=0):
    rng = np.random.default_rng(seed)
    xt = inst.target_true()
    errs = []
    for _ in range(n_mc):
        Q = inst.clean_measurements(rng)
        T = est(inst.P, Q)
        errs.append(np.linalg.norm(T.R @ inst.x + T.t - xt))
    errs = np.array(errs)
    return float(np.sqrt((errs ** 2).mean())), errs


def first_order_operators(inst: Instance):
    A = stack_jacobians(inst.R_true, inst.P)
    G = target_jacobian(inst.R_true, inst.x)
    return A, G
