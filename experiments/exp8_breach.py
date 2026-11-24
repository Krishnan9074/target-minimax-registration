"""Experiment 8 -- medial breach margin, the scalar functional that matters.

For a transpedicular screw the clinically binding quantity is not the full 3-D
displacement of the tip: it is how far the corridor moves *toward the medial
pedicle wall*, because that is the direction in which a breach reaches the
spinal canal.  That is a scalar directional functional, and it is exactly the
case in which the estimator-independent limit is attained (Hahn--Banach), the
worst-case risk has a closed form, and no sphere search is needed.

We evaluate at the pedicle isthmus --- the narrowest point of the corridor,
where the available margin is smallest --- and report the worst-case medial
displacement as a fraction of that margin.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmr.anatomy import build_vertebra
from tmr.geometry import stack_jacobians, target_jacobian
from tmr.estimators import gls_influence
from tmr.limits import scalar_design, scalar_limit, scalar_risk
from experiments.exp5_vertebra import SIGMA_BY_LANDMARK

OUT = os.path.join(ROOT, "results", "exp8_breach.json")


def isthmus_and_margin(v, side):
    """Narrowest point of the corridor and the in-bone clearance there (mm)."""
    tree = cKDTree(v.nodes)
    inside = lambda p: tree.query(np.atleast_2d(p))[0][0] < 3.0
    traj = v.trajectory(side, 60)
    u = traj[-1] - traj[0]
    u = u / np.linalg.norm(u)
    e1 = np.cross(u, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    dirs = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
    best = (np.inf, None)
    for c in traj:
        if not (-16.0 < c[1] < 2.0):        # restrict to the pedicle
            continue
        r = min(_ray(c, d, inside) for d in dirs)
        if r < best[0]:
            best = (r, c)
    return best[1], best[0]


def _ray(c, d, inside, r_max=12.0, step=0.25):
    r = 0.0
    while r < r_max and inside(c + (r + step) * d):
        r += step
    return r


if __name__ == "__main__":
    v = build_vertebra()
    names = list(SIGMA_BY_LANDMARK)
    P = v.fiducials(names)
    n = len(P)
    sig = np.array([SIGMA_BY_LANDMARK[nm] for nm in names])
    Sigma = np.diag(np.repeat(sig ** 2, 3))
    A = stack_jacobians(np.eye(3), P)
    rho = np.ones(n)
    Si = np.linalg.inv(Sigma)
    K0 = np.linalg.solve(A.T @ Si @ A, A.T @ Si)

    recs = []
    for side in ("left", "right"):
        xi_pt, margin = isthmus_and_margin(v, side)
        # medial direction: toward the mid-sagittal plane
        c = np.array([-np.sign(xi_pt[0]), 0.0, 0.0])
        G = target_jacobian(np.eye(3), xi_pt)
        g = c @ G
        m_gls = (g @ K0).ravel()
        row = dict(side=side, isthmus=list(map(float, xi_pt)),
                   margin_mm=float(margin), direction=list(map(float, c)))
        for k in (1, 2):
            _, v_tmr, _ = scalar_design(A, g, rho, k, Sigma=Sigma, eta=0.10)
            _, v_free, _ = scalar_design(A, g, rho, k)
            lim = scalar_limit(A, g, k)
            row[f"k{k}"] = dict(
                gls=scalar_risk(m_gls, rho, k),
                tmr=v_tmr, tmr_unconstrained=v_free, limit=lim,
                slack_tmr=v_tmr / lim, slack_gls=scalar_risk(m_gls, rho, k) / lim)
        recs.append(row)

    json.dump(recs, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    for r in recs:
        print(f"\n=== {r['side']} pedicle: isthmus clearance {r['margin_mm']:.2f} mm ===")
        for k in (1, 2):
            d = r[f"k{k}"]
            print(f"  k={k}  medial displacement per mm of corruption:")
            print(f"        GLS {d['gls']:.3f}   TMR(eta=0.10) {d['tmr']:.3f}   "
                  f"unconstrained {d['tmr_unconstrained']:.3f}   "
                  f"attainable limit {d['limit']:.3f}")
            print(f"        margin consumed at rho=1mm: GLS "
                  f"{100*d['gls']/r['margin_mm']:.1f}%  ->  TMR "
                  f"{100*d['tmr']/r['margin_mm']:.1f}%   "
                  f"(TMR is {d['slack_tmr']:.3f}x the limit)")
