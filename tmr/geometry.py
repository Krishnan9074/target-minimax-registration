"""SE(3) utilities and first-order registration Jacobians.

Conventions
-----------
A rigid transform ``T = (R, t)`` acts as ``T(y) = R y + t``.

Local (tangent-space) parameterisation.  Around a nominal pose ``T0 = (R0, t0)``
we use the *left* perturbation

    R = Exp(w) R0,      t = t0 + tau,      xi = (w, tau) in R^6.

To first order in ``xi`` the image of a point ``p`` moves by

    R p + t  ~=  R0 p + t0  -  [R0 p]_x w  +  tau,

so the 3x6 Jacobian of the transformed point w.r.t. ``xi`` is

    A(p) = [ -[R0 p]_x , I_3 ].

The same expression evaluated at a target ``x`` gives ``G_x = A(x)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "hat", "expm_so3", "logm_so3", "random_rotation", "fiducial_jacobian",
    "target_jacobian", "stack_jacobians", "apply_transform", "RigidTransform",
    "project_to_so3",
]


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix with ``hat(v) @ w == np.cross(v, w)``."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def expm_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map ``so(3) -> SO(3)``."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    W = hat(w)
    if theta < 1e-12:
        # second-order Taylor expansion; exact to O(theta^3)
        return np.eye(3) + W + 0.5 * W @ W
    return (np.eye(3)
            + (np.sin(theta) / theta) * W
            + ((1.0 - np.cos(theta)) / theta ** 2) * (W @ W))


def logm_so3(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`expm_so3` (principal branch, ``|theta| <= pi``)."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    theta = float(np.arccos(c))
    if theta < 1e-9:
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return 0.5 * v
    if np.pi - theta < 1e-6:
        # near-pi branch: recover axis from the symmetric part
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        axis = A[:, k] / (axis[k] if axis[k] > 1e-12 else 1.0)
        axis = axis / np.linalg.norm(axis)
        return theta * axis
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * np.sin(theta)) * v


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation in Frobenius norm (special orthogonal Procrustes)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform (Haar) random rotation via QR of a Gaussian matrix."""
    Q, Rm = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


@dataclass(frozen=True)
class RigidTransform:
    """A rigid transform ``y -> R y + t``."""

    R: np.ndarray
    t: np.ndarray

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        return apply_transform(self.R, self.t, pts)

    def compose_left(self, xi: np.ndarray) -> "RigidTransform":
        """Return ``Exp(w) R, t + tau`` for ``xi = (w, tau)``."""
        xi = np.asarray(xi, dtype=float).reshape(6)
        return RigidTransform(expm_so3(xi[:3]) @ self.R, self.t + xi[3:])

    @staticmethod
    def identity() -> "RigidTransform":
        return RigidTransform(np.eye(3), np.zeros(3))


def apply_transform(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    return pts @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float)


def fiducial_jacobian(R0: np.ndarray, p: np.ndarray) -> np.ndarray:
    """3x6 Jacobian ``A_i = [-[R0 p]_x, I]`` of the transformed fiducial."""
    return np.hstack([-hat(np.asarray(R0, dtype=float) @ np.asarray(p, float)),
                      np.eye(3)])


def target_jacobian(R0: np.ndarray, x: np.ndarray) -> np.ndarray:
    """3x6 Jacobian ``G_x`` of the transformed target (same form as ``A_i``)."""
    return fiducial_jacobian(R0, x)


def stack_jacobians(R0: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Stack per-fiducial Jacobians into ``A`` of shape ``(3n, 6)``."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return np.vstack([fiducial_jacobian(R0, p) for p in P])
