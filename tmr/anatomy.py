"""Lumbar vertebra geometry for the surgical instantiation of TMR.

Geometry source
---------------
Finite-element mesh of the upper lumbar / lower thoracic spine, Zenodo record
10.5281/zenodo.3961334 (CC-BY-4.0).  We use only the *rigid geometry* of the L1
vertebral node cloud: node families 4/5/8 of the MED file.  No material model,
no simulation output and no patient data are used.

Anatomical frame (derived from the mesh, not assumed)
    x  left-right      (transverse-process span ~83 mm)
    y  posterior->anterior (+y anterior; spinous-process tip at y ~ -54 mm)
    z  caudo-cranial
with the origin at the L1 node-cloud centroid.

Landmarks are the bony points a navigation probe can reach on the exposed
posterior elements; the surgical target is a transpedicular screw trajectory
whose axis is fitted to the pedicle isthmus of the same mesh.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")

L1_FAMILIES = (4, 5, 8)
ZENODO_DOI = "10.5281/zenodo.3961334"


def load_l1_nodes(med_path: str | None = None) -> np.ndarray:
    """Return the centred L1 node cloud in the anatomical frame (mm)."""
    import h5py
    med_path = med_path or os.path.join(DATA, "L1-L2.med")
    with h5py.File(med_path, "r") as f:
        noe = f["ENS_MAA/All/-0000000000000000001-0000000000000000001/NOE"]
        coo = np.array(noe["COO"], dtype=float)
        fam = np.array(noe["FAM"])
    n = len(fam)
    X = coo.reshape(3, n).T
    P = X[np.isin(fam, L1_FAMILIES)]
    return P - P.mean(axis=0)


@dataclass
class VertebraModel:
    nodes: np.ndarray
    landmarks: dict            # name -> (3,) point
    entry: dict                # side -> screw entry point
    tip: dict                  # side -> screw tip
    body_extent: tuple

    @property
    def landmark_names(self):
        return list(self.landmarks)

    def fiducials(self, names=None) -> np.ndarray:
        names = names or self.landmark_names
        return np.array([self.landmarks[k] for k in names])

    def trajectory(self, side="left", m: int = 9) -> np.ndarray:
        """``m`` points sampled along the screw axis, entry -> tip."""
        s = np.linspace(0.0, 1.0, m)[:, None]
        return self.entry[side] * (1 - s) + self.tip[side] * s


def _extreme(P, w, top=40):
    """Mean of the ``top`` nodes with the largest projection onto ``w``."""
    s = P @ np.asarray(w, float)
    idx = np.argsort(s)[-top:]
    return P[idx].mean(axis=0)


def build_vertebra(med_path: str | None = None) -> VertebraModel:
    P = load_l1_nodes(med_path)
    xm = 0.5 * (P[:, 0].min() + P[:, 0].max())      # mid-sagittal plane
    P = P - np.array([xm, 0.0, 0.0])

    mid = np.abs(P[:, 0]) < 7.0
    L = P[:, 0] > 6.0
    R = P[:, 0] < -6.0

    lm = {}
    # spinous process tip: most posterior midline point
    lm["SP_tip"] = _extreme(P[mid], [0, -1, 0], 30)
    # transverse process tips: most lateral points
    lm["TP_left"] = _extreme(P, [1, 0, 0], 30)
    lm["TP_right"] = _extreme(P, [-1, 0, 0], 30)
    # superior / inferior articular processes: cranial / caudal extremes of the
    # posterior lateral elements
    post = P[:, 1] < -5.0
    for side, sgn, mask in (("left", 1.0, L), ("right", -1.0, R)):
        sub = P[mask & post]
        lm[f"SAP_{side}"] = _extreme(sub, [0.15 * sgn, -0.2, 1.0], 25)
        lm[f"IAP_{side}"] = _extreme(sub, [0.15 * sgn, -0.2, -1.0], 25)
        # laminar surface point: posterior surface between SP and the facet
        lam = sub[(np.abs(sub[:, 2]) < 8.0)
                  & (np.abs(sub[:, 0]) > 8.0) & (np.abs(sub[:, 0]) < 26.0)]
        lm[f"LAM_{side}"] = _extreme(lam, [0.3 * sgn, -1.0, 0.0], 25)

    # ---- transpedicular corridor: maximum-clearance axis ------------------ #
    # Entry points are taken from the mesh (junction of the transverse and
    # superior articular processes).  The axis is then selected by maximising
    # the clearance of the screw corridor inside the bone, which is the
    # criterion used when a transpedicular trajectory is planned.
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    RES = 3.0                                   # mesh occupancy radius (mm)

    def in_bone(pts):
        return tree.query(np.atleast_2d(pts))[0] < RES

    def corridor_radius(path, n_ray=12, r_max=12.0, step=0.5):
        """Min over path points of the in-bone radius perpendicular to the axis."""
        u = path[-1] - path[0]
        u = u / np.linalg.norm(u)
        e1 = np.cross(u, [0, 0, 1.0])
        e1 = e1 / max(np.linalg.norm(e1), 1e-9)
        e2 = np.cross(u, e1)
        ang = np.linspace(0, 2 * np.pi, n_ray, endpoint=False)
        dirs = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
        radii = np.full(len(path), r_max)
        for j, c in enumerate(path):
            for d in dirs:
                r = 0.0
                while r < r_max and in_bone(c + (r + step) * d)[0]:
                    r += step
                radii[j] = min(radii[j], r)
        return radii

    entry, tip = {}, {}
    body = P[P[:, 1] > 5.0]
    y_ant = float(np.percentile(body[:, 1], 99.0))
    y_tip = 0.85 * y_ant
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        # entry: posterior cortex at the TP / SAP junction
        reg = P[(sgn * P[:, 0] > 10.0) & (sgn * P[:, 0] < 24.0)
                & (np.abs(P[:, 2]) < 10.0)]
        e0 = reg[np.argsort(reg[:, 1])[:25]].mean(axis=0)
        best = (-np.inf, None)
        for alpha in np.radians(np.arange(0.0, 31.0, 2.5)):     # medial angle
            for beta in np.radians(np.arange(-16.0, 16.1, 2.5)):  # cranial
                u = np.array([-sgn * np.sin(alpha) * np.cos(beta),
                              np.cos(alpha) * np.cos(beta), np.sin(beta)])
                if u[1] < 0.2:
                    continue
                tp = e0 + ((y_tip - e0[1]) / u[1]) * u
                path = e0 + np.linspace(0, 1, 24)[:, None] * (tp - e0)
                if not in_bone(path).all():
                    continue
                score = float(corridor_radius(path[:14]).min())
                if score > best[0]:
                    best = (score, (e0, tp))
        if best[1] is None:                       # fall back to a straight axis
            u = np.array([-sgn * np.sin(np.radians(12.0)),
                          np.cos(np.radians(12.0)), 0.0])
            best = (np.nan, (e0, e0 + ((y_tip - e0[1]) / u[1]) * u))
        entry[side], tip[side] = best[1]

    return VertebraModel(P, lm, entry, tip,
                         tuple(float(v) for v in np.ptp(body, axis=0)))


if __name__ == "__main__":
    v = build_vertebra()
    print(f"L1 node cloud: {v.nodes.shape[0]} nodes, "
          f"extent {np.round(np.ptp(v.nodes, axis=0), 1)} mm")
    print(f"vertebral body extent (w,d,h) = {np.round(v.body_extent, 1)} mm")
    for k, p in v.landmarks.items():
        print(f"  {k:10s} {np.round(p, 1)}")
    for s in ("left", "right"):
        d = np.linalg.norm(v.tip[s] - v.entry[s])
        u = (v.tip[s] - v.entry[s]) / d
        med = np.degrees(np.arctan2(abs(u[0]), u[1]))
        cra = np.degrees(np.arcsin(u[2]))
        from scipy.spatial import cKDTree
        tr = cKDTree(v.nodes)
        dev = tr.query(v.trajectory(s, 40))[0].max()
        print(f"  screw {s:5s}: entry {np.round(v.entry[s], 1)} -> "
              f"tip {np.round(v.tip[s], 1)}  length {d:.1f} mm, "
              f"medial {med:.1f} deg, cranial {cra:+.1f} deg, "
              f"max dist to bone {dev:.1f} mm")
