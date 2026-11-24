"""Target-Minimax Registration (TMR).

Rigid paired-point registration estimators designed to minimise the worst-case
displacement of a downstream surgical target under k-sparse, bounded integrity
corruption of the fiducial measurements.
"""
from .geometry import (
    hat, expm_so3, logm_so3, random_rotation, fiducial_jacobian,
    target_jacobian, stack_jacobians, apply_transform, RigidTransform,
)
from .risk import (
    sparse_risk_exact, sparse_risk_topk_bound, single_block_risk,
    clean_target_variance, geometric_amplification,
)
from .estimators import (
    kabsch, weighted_kabsch, gls_influence, tmr_influence, tmr_weights,
    huber_irls, ransac_register, gnc_tls, influence_to_gain,
    tmr_register_se3,
)

__all__ = [n for n in dir() if not n.startswith("_")]
