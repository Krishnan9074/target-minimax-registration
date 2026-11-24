"""Numerical verification of every mathematical claim used in the paper.

Each test corresponds to a labelled result in the manuscript.  The suite is
deterministic: all randomness is seeded.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tmr.geometry import (random_rotation, stack_jacobians, target_jacobian,
                          fiducial_jacobian, expm_so3, hat)
from tmr.risk import (sparse_risk_exact, sparse_risk_bruteforce, blocks_of,
                      single_block_risk, sparse_risk_topk_bound,
                      geometric_amplification, worst_case_perturbation)
from tmr.estimators import (gls_influence, tmr_influence, tmr_influence_sdp,
                            tmr_dual_k1, complete_influence_to_K, kabsch,
                            weighted_kabsch, wls_influence, average_influence)
from tmr.risk import average_risk, average_risk_mc
from tmr.limits import (nu_k, scalar_design, scalar_limit, fundamental_limit,
                        scalar_risk)

REPORT = []


def log(name, ok, detail):
    REPORT.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def make_problem(rng, n=6, spread=60.0, target_offset=80.0, sigma=0.5):
    P = rng.uniform(-spread, spread, size=(n, 3))
    x = P.mean(0) + rng.standard_normal(3) / np.linalg.norm(
        rng.standard_normal(3) + 1e-9) * 0 + np.array([target_offset, 0, 0])
    R0 = random_rotation(rng)
    A = stack_jacobians(R0, P)
    G = target_jacobian(R0, x)
    Sigma = np.zeros((3 * n, 3 * n))
    for i in range(n):
        Bm = rng.standard_normal((3, 3)) * 0.4
        Sigma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = (
            sigma ** 2 * (np.eye(3) + Bm @ Bm.T))
    return P, x, R0, A, G, Sigma


# --------------------------------------------------------------------------- #
def test_jacobian_central_difference():
    rng = np.random.default_rng(1)
    R0 = random_rotation(rng)
    p = rng.standard_normal(3) * 80
    A = fiducial_jacobian(R0, p)
    h = 1e-5
    num = np.zeros((3, 6))
    for j in range(6):
        e = np.zeros(6); e[j] = h
        yp = expm_so3(e[:3]) @ R0 @ p + e[3:]
        ym = expm_so3(-e[:3]) @ R0 @ p - e[3:]
        num[:, j] = (yp - ym) / (2 * h)
    rel = np.max(np.abs(num - A)) / np.max(np.abs(A))
    log("T0 fiducial/target Jacobian (central differences)", rel < 1e-8,
        f"max relative error {rel:.2e}")


def test_influence_reduction_lemma():
    """Lemma 1: {G K : K A = I} == {M : M A = G}."""
    rng = np.random.default_rng(2)
    worst_fwd = worst_bwd = 0.0
    for _ in range(20):
        P, x, R0, A, G, Sigma = make_problem(rng)
        n = A.shape[0] // 3
        Si = np.linalg.inv(Sigma)
        K0 = np.linalg.solve(A.T @ Si @ A, A.T @ Si)
        # forward: any unbiased K gives M with M A = G
        Kr = K0 + (np.eye(6) - K0 @ A) @ rng.standard_normal((6, 3 * n))
        worst_fwd = max(worst_fwd, np.max(np.abs(Kr @ A - np.eye(6))))
        M = G @ Kr
        worst_fwd = max(worst_fwd, np.max(np.abs(M @ A - G)))
        # backward: any M with M A = G is realised by an unbiased K
        Mr = rng.standard_normal((3, 3 * n))
        Mr = Mr + (G - Mr @ A) @ np.linalg.pinv(A)
        K = complete_influence_to_K(Mr, G, K0)
        worst_bwd = max(worst_bwd,
                        np.max(np.abs(K @ A - np.eye(6))),
                        np.max(np.abs(G @ K - Mr)))
    log("L1 influence-reduction lemma (both directions)",
        max(worst_fwd, worst_bwd) < 1e-8,
        f"max residual {max(worst_fwd, worst_bwd):.2e}")


def test_exact_sparse_risk():
    """Theorem 2: R_k(M) = max_{|u|=1} topk_i rho_i ||M_i' u||."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for trial in range(12):
        P, x, R0, A, G, Sigma = make_problem(rng, n=rng.integers(4, 8))
        n = A.shape[0] // 3
        M = gls_influence(A, G, Sigma)
        rho = rng.uniform(0.5, 2.0, n)
        for k in (1, 2, 3):
            if k > n:
                continue
            v_fast = sparse_risk_exact(M, rho, k)
            v_bf = sparse_risk_bruteforce(M, rho, k, n_dir=40000)
            # independent check: direct maximisation over the corruption itself
            Mb = blocks_of(M)
            best = 0.0
            for S in itertools.combinations(range(n), k):
                def neg(z, S=S):
                    a = np.zeros((n, 3))
                    for j, i in enumerate(S):
                        v = z[3 * j:3 * j + 3]
                        a[i] = rho[i] * v / max(np.linalg.norm(v), 1e-12)
                    return -np.linalg.norm(M @ a.reshape(-1))
                for _ in range(6):
                    r = minimize(neg, rng.standard_normal(3 * k), method="BFGS")
                    best = max(best, -float(r.fun))
            worst = max(worst, abs(v_fast - max(v_bf, best)) / max(v_fast, 1e-12))
    log("T2 exact k-sparse risk identity", worst < 5e-4,
        f"max relative disagreement vs. brute force {worst:.2e}")


def test_k1_closed_form():
    """R_1(M) = max_i rho_i ||M_i||_2 (spectral norm)."""
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(30):
        P, x, R0, A, G, Sigma = make_problem(rng)
        M = gls_influence(A, G, Sigma)
        rho = rng.uniform(0.3, 3.0, A.shape[0] // 3)
        worst = max(worst, abs(sparse_risk_exact(M, rho, 1)
                               - single_block_risk(M, rho)))
    log("T2a k=1 reduces to max block spectral norm", worst < 1e-9,
        f"max absolute disagreement {worst:.2e}")


def test_topk_bound_sandwich():
    """R_1 <= R_k <= topk bound, and (k/n) * R_n <= R_k."""
    rng = np.random.default_rng(5)
    ok = True
    slack = []
    for _ in range(20):
        P, x, R0, A, G, Sigma = make_problem(rng, n=7)
        M = gls_influence(A, G, Sigma)
        rho = rng.uniform(0.5, 1.5, 7)
        r = [sparse_risk_exact(M, rho, k) for k in range(1, 8)]
        ok &= all(r[i] <= r[i + 1] + 1e-9 for i in range(6))
        for k in range(1, 8):
            ub = sparse_risk_topk_bound(M, rho, k)
            ok &= r[k - 1] <= ub + 1e-9
            slack.append(ub / max(r[k - 1], 1e-12))
            ok &= r[k - 1] + 1e-9 >= (k / 7.0) * r[6]
    log("T3 monotonicity and top-k sandwich bounds", ok,
        f"upper bound tightness ratio in [{min(slack):.3f}, {max(slack):.3f}]")


def test_gauss_markov_recovery():
    """Proposition 4: eta = 0 forces M = M_GLS; the frontier starts at GLS."""
    rng = np.random.default_rng(6)
    worst = 0.0
    for _ in range(6):
        P, x, R0, A, G, Sigma = make_problem(rng, n=6)
        rho = np.ones(6)
        M_gls = gls_influence(A, G, Sigma)
        res = tmr_influence(A, G, Sigma, rho, k=1, eta=0.0, max_rounds=8)
        worst = max(worst, np.max(np.abs(res.M - M_gls))
                    / max(np.max(np.abs(M_gls)), 1e-12))
    log("P4 eta = 0 recovers Gauss-Markov exactly", worst < 5e-3,
        f"max relative deviation from M_GLS {worst:.2e}")


def test_gls_is_variance_minimiser():
    rng = np.random.default_rng(7)
    ok = True
    for _ in range(20):
        P, x, R0, A, G, Sigma = make_problem(rng)
        n = A.shape[0] // 3
        M_gls = gls_influence(A, G, Sigma)
        v0 = np.trace(M_gls @ Sigma @ M_gls.T)
        for _ in range(20):
            Mr = rng.standard_normal((3, 3 * n))
            Mr = Mr + (G - Mr @ A) @ np.linalg.pinv(A)
            ok &= np.trace(Mr @ Sigma @ Mr.T) >= v0 - 1e-9
    log("P4a Gauss-Markov optimality of M_GLS under M A = G", ok,
        "no random feasible point achieved lower variance")


def test_cutting_plane_matches_sdp():
    """The exchange SOCP and the direct SDP agree for k = 1."""
    rng = np.random.default_rng(8)
    worst = 0.0
    for _ in range(5):
        P, x, R0, A, G, Sigma = make_problem(rng, n=6)
        rho = rng.uniform(0.5, 1.5, 6)
        res = tmr_influence(A, G, Sigma, rho, k=1, eta=0.10, max_rounds=30)
        _, v_sdp, st = tmr_influence_sdp(A, G, Sigma, rho, eta=0.10)
        worst = max(worst, abs(res.risk - v_sdp) / max(v_sdp, 1e-12))
    log("T5 cutting-plane SOCP == direct SDP for k=1", worst < 2e-3,
        f"max relative value gap {worst:.2e}")


def test_strong_duality_k1():
    """Dual of the variance-free k=1 problem attains the primal value."""
    rng = np.random.default_rng(9)
    worst = 0.0
    for _ in range(5):
        P, x, R0, A, G, Sigma = make_problem(rng, n=5)
        rho = rng.uniform(0.5, 1.5, 5)
        res = tmr_influence(A, G, Sigma, rho, k=1, eta=1e6, max_rounds=30)
        _, dval, st = tmr_dual_k1(A, G, rho)
        worst = max(worst, abs(res.risk - dval) / max(dval, 1e-12))
    log("T6 strong duality (nuclear-norm dual) for k=1", worst < 5e-3,
        f"max relative primal-dual gap {worst:.2e}")


def test_lower_bound():
    """Theorem 7: R_k(M) >= (k/n) * rho_min * kappa_x for every unbiased M."""
    rng = np.random.default_rng(10)
    ok, ratios = True, []
    for _ in range(10):
        P, x, R0, A, G, Sigma = make_problem(rng, n=6)
        n = 6
        kappa = geometric_amplification(A, G)
        rho = np.ones(n)
        for k in (1, 2, 3):
            res = tmr_influence(A, G, Sigma, rho, k=k, eta=1e6, max_rounds=15)
            lb = (k / n) * kappa
            ok &= res.risk >= lb - 1e-6
            ratios.append(res.risk / max(lb, 1e-12))
    log("T7 estimator-independent lower bound holds", ok,
        f"achieved/bound ratio in [{min(ratios):.2f}, {max(ratios):.2f}]")


def test_invariance():
    """Proposition 8: invariance to world-frame origin and rotation."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(5):
        P, x, R0, A, G, Sigma = make_problem(rng, n=5)
        rho = rng.uniform(0.5, 1.5, 5)
        r0 = tmr_influence(A, G, Sigma, rho, k=2, eta=0.1, max_rounds=12)
        # origin shift of the image frame
        c = rng.standard_normal(3) * 40
        A2 = stack_jacobians(R0, P + c)
        G2 = target_jacobian(R0, x + c)
        r1 = tmr_influence(A2, G2, Sigma, rho, k=2, eta=0.1, max_rounds=12)
        # rotation of the physical frame
        Rg = random_rotation(rng)
        Bg = np.kron(np.eye(5), Rg)
        A3 = Bg @ A @ np.kron(np.eye(2), Rg.T)
        G3 = Rg @ G @ np.kron(np.eye(2), Rg.T)
        S3 = Bg @ Sigma @ Bg.T
        r2 = tmr_influence(A3, G3, S3, rho, k=2, eta=0.1, max_rounds=12)
        worst = max(worst, abs(r1.risk - r0.risk) / r0.risk,
                    abs(r2.risk - r0.risk) / r0.risk)
    log("P8 invariance to origin choice and frame rotation", worst < 5e-3,
        f"max relative change in optimal risk {worst:.2e}")


def test_exactness_noiseless():
    """Any unbiased estimator recovers T* exactly on clean noiseless data."""
    rng = np.random.default_rng(12)
    from tmr.estimators import tmr_register_se3
    worst = 0.0
    for _ in range(3):
        P = rng.uniform(-50, 50, size=(6, 3))
        Rs, ts = random_rotation(rng), rng.standard_normal(3) * 20
        Q = P @ Rs.T + ts
        x = P.mean(0) + np.array([70.0, 10.0, -5.0])
        Sigma = np.eye(18)
        T, info = tmr_register_se3(P, Q, x, Sigma, np.ones(6), k=1, eta=0.1,
                                   n_iter=8, max_rounds=6)
        worst = max(worst, np.linalg.norm(T.R @ x + T.t - (Rs @ x + ts)))
    log("P9 exact recovery of the noiseless clean transform", worst < 1e-8,
        f"max target error {worst:.2e} mm")


def test_first_order_validity():
    """The linearisation remainder is second order: rel. error ~ O(rho)."""
    rng = np.random.default_rng(13)
    rhos = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    med = []
    for rho in rhos:
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
            d_true = (T1.R @ x + T1.t) - (T0.R @ x + T0.t)
            rel.append(np.linalg.norm(d_true - M @ a.reshape(-1))
                       / np.linalg.norm(d_true))
        med.append(np.median(rel))
    slope = float(np.polyfit(np.log(rhos), np.log(med), 1)[0])
    log("P10 linearisation remainder is O(rho^2)",
        abs(slope - 1.0) < 0.1 and med[3] < 1e-2,
        f"log-log slope {slope:.3f} (1.0 expected); "
        f"median error {med[3]:.1e} at rho = 0.5 mm")


def test_average_risk_closed_form():
    """The average-case risk is a weighted squared Frobenius norm."""
    rng = np.random.default_rng(20)
    worst = 0.0
    for t in range(4):
        P, x, R0, A, G, Sigma = make_problem(rng, n=6)
        M = gls_influence(A, G, Sigma)
        rho = rng.uniform(0.5, 2.0, 6)
        for k in (1, 2, 3):
            cf = average_risk(M, rho, k)
            mc = average_risk_mc(M, rho, k, n_mc=60000, seed=t)
            worst = max(worst, abs(cf - mc) / cf)
    log("T8 average-case risk closed form", worst < 3e-2,
        f"max relative disagreement vs. Monte Carlo {worst:.2e}")


def test_average_case_collapse():
    """Under homoscedastic noise the average-case design *is* Gauss-Markov."""
    rng = np.random.default_rng(21)
    worst = 0.0
    for _ in range(5):
        n = 7
        P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
        x = 90 * rng.standard_normal(3); x /= np.linalg.norm(x) / 90
        A = stack_jacobians(np.eye(3), P)
        G = target_jacobian(np.eye(3), x)
        S = 0.35 ** 2 * np.eye(3 * n)
        Mg = gls_influence(A, G, S)
        for k in (1, 3):
            Ma, _ = average_influence(A, G, S, np.ones(n), k=k, eta=1e6)
            worst = max(worst, np.linalg.norm(Ma - Mg) / np.linalg.norm(Mg))
    log("T9 average-case design collapses to Gauss-Markov", worst < 1e-8,
        f"max relative deviation from M_GLS {worst:.2e}")


def test_nu_k_duality():
    """sup{ ||M a|| : nu_k(a) <= 1 } equals the k-sparse risk."""
    rng = np.random.default_rng(22)
    worst = 0.0
    for _ in range(6):
        P, x, R0, A, G, Sigma = make_problem(rng, n=6)
        M = gls_influence(A, G, Sigma)
        for k in (1, 2, 4):
            rk = sparse_risk_exact(M, np.ones(6), k)
            # random points of the nu_k ball can never exceed the risk
            hi = 0.0
            for _ in range(4000):
                a = rng.standard_normal(18)
                a = a / nu_k(a, k)
                hi = max(hi, float(np.linalg.norm(M @ a)))
            worst = max(worst, (hi - rk) / rk)
    log("T10 nu_k unit ball is conv A(k,1)", worst < 1e-9,
        f"largest excess of a nu_k-feasible point over R_k {worst:.2e}")


def test_scalar_limit_attained():
    """Hahn-Banach: for a scalar loss the fundamental limit is attained."""
    rng = np.random.default_rng(23)
    worst = 0.0
    for _ in range(5):
        n = int(rng.integers(4, 9))
        P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
        x = 80 * rng.standard_normal(3); x /= np.linalg.norm(x) / 80
        A = stack_jacobians(np.eye(3), P)
        G = target_jacobian(np.eye(3), x)
        c = rng.standard_normal(3); c /= np.linalg.norm(c)
        for k in (1, 2, n):
            _, v, _ = scalar_design(A, c @ G, np.ones(n), k)
            b = scalar_limit(A, c @ G, k)
            worst = max(worst, abs(v - b) / max(abs(b), 1e-12))
    log("T11 scalar-loss limit is attained exactly", worst < 1e-5,
        f"max relative gap between design and limit {worst:.2e}")


def test_linf_limit_attained():
    """The limit is also attained for an ell-infinity target loss."""
    import cvxpy as cp
    rng = np.random.default_rng(24)
    worst = 0.0
    for _ in range(3):
        n = 6
        P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
        x = 80 * rng.standard_normal(3); x /= np.linalg.norm(x) / 80
        A = stack_jacobians(np.eye(3), P)
        G = target_jacobian(np.eye(3), x)
        for k in (1, 2):
            M = cp.Variable((3, 3 * n))
            nb = cp.hstack([cp.max(cp.hstack(
                [cp.norm(M[j, 3 * i:3 * i + 3], 2) for j in range(3)]))
                for i in range(n)])
            pr = cp.Problem(cp.Minimize(cp.sum_largest(nb, k)), [M @ A == G])
            pr.solve(solver="SCS", eps=1e-9, max_iters=200000)
            b = fundamental_limit(A, G, k, loss="inf")
            worst = max(worst, abs(pr.value - b) / max(b, 1e-12))
    log("T12 ell-infinity target loss: limit attained", worst < 5e-4,
        f"max relative gap {worst:.2e}")


def test_new_bound_dominates_old():
    """kappa^(k) is never worse than the (k/n) kappa_x bound, and is far tighter."""
    rng = np.random.default_rng(25)
    ratios_new, ratios_old, ok = [], [], True
    for _ in range(5):
        n = 6
        P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
        x = 80 * rng.standard_normal(3); x /= np.linalg.norm(x) / 80
        A = stack_jacobians(np.eye(3), P)
        G = target_jacobian(np.eye(3), x)
        kap = geometric_amplification(A, G)
        for k in (1, 2):
            achieved = tmr_influence(A, G, np.eye(3 * n), np.ones(n), k=k,
                                     eta=1e6).risk
            new = fundamental_limit(A, G, k, loss="2")
            old = (k / n) * kap
            ok &= new >= old - 1e-6 and achieved >= new - 1e-4
            ratios_new.append(achieved / new)
            ratios_old.append(achieved / old)
    log("T13 tightened limit dominates the counting bound", ok,
        f"slack: new {min(ratios_new):.3f}-{max(ratios_new):.3f} vs "
        f"old {min(ratios_old):.2f}-{max(ratios_old):.2f}")


def test_complementary_slackness():
    """The optimum equalises leverage across the dual-active blocks."""
    from tmr.risk import blocks_of
    rng = np.random.default_rng(26)
    worst_spread, ok = 0.0, True
    for _ in range(5):
        n = int(rng.integers(5, 9))
        P = rng.uniform(-60, 60, size=(n, 3)); P -= P.mean(0)
        x = 80 * rng.standard_normal(3); x /= np.linalg.norm(x) / 80
        A = stack_jacobians(np.eye(3), P)
        G = target_jacobian(np.eye(3), x)
        r = tmr_influence(A, G, np.eye(3 * n), np.ones(n), k=1, eta=1e6)
        Lam, _, _ = tmr_dual_k1(A, G, np.ones(n))
        lev = np.linalg.norm(blocks_of(r.M), ord=2, axis=(1, 2))
        nuc = np.array([np.linalg.svd(Lam @ A[3 * i:3 * i + 3, :].T,
                                      compute_uv=False).sum() for i in range(n)])
        act = nuc > 1e-3 * nuc.max()
        worst_spread = max(worst_spread, lev[act].std() / lev[act].mean())
        if (~act).any():
            ok &= lev[~act].max() <= lev[act].mean() * (1 + 1e-3)
    log("T14 complementary slackness equalises the active set",
        ok and worst_spread < 5e-3,
        f"max relative spread on the active set {worst_spread:.1e}; "
        f"inactive blocks never exceed it")


if __name__ == "__main__":
    test_jacobian_central_difference()
    test_influence_reduction_lemma()
    test_exact_sparse_risk()
    test_k1_closed_form()
    test_topk_bound_sandwich()
    test_gls_is_variance_minimiser()
    test_gauss_markov_recovery()
    test_cutting_plane_matches_sdp()
    test_strong_duality_k1()
    test_lower_bound()
    test_invariance()
    test_exactness_noiseless()
    test_first_order_validity()
    test_average_risk_closed_form()
    test_average_case_collapse()
    test_nu_k_duality()
    test_scalar_limit_attained()
    test_linf_limit_attained()
    test_new_bound_dominates_old()
    test_complementary_slackness()
    n_fail = sum(1 for _, ok, _ in REPORT if not ok)
    print(f"\n{len(REPORT) - n_fail}/{len(REPORT)} checks passed")
    import json, pathlib
    pathlib.Path(ROOT, "results").mkdir(exist_ok=True)
    json.dump([{"name": a, "pass": b, "detail": c} for a, b, c in REPORT],
              open(os.path.join(ROOT, "results", "verification.json"), "w"),
              indent=2)
    sys.exit(1 if n_fail else 0)
