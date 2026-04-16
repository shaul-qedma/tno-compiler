"""Kalloor ensemble: optimal weights for mixing approximate compilations.

Given M compilations U_i of target V, find weights p_i minimizing
||Σ p_i U_i - V||_F², which is the convex QP (Kalloor et al. 2025):

    min  (1/2) p^T H p + f^T p
    s.t. p >= 0, Σ p_i = 1

where H_ij = 2 Re Tr(U_i† U_j),  f_i = -2 Re Tr(U_i† V).

The Mixing Lemma (Campbell 2017, Hastings 2017) then gives:
    d_diamond(U_ensemble, V) = O(||Σ p_i U_i - V||_F²)
"""

import numpy as np
from scipy.optimize import minimize


def ensemble_qp(gram, target_overlaps):
    """Solve the Kalloor QP for optimal ensemble weights.

    Args:
        gram: (M, M) array, gram[i,j] = Re Tr(U_i† U_j).
        target_overlaps: (M,) array, target_overlaps[i] = Re Tr(U_i† V).

    Returns:
        weights: (M,) optimal probabilities.
        qp_value: minimum of (1/2) p^T H p + f^T p.
            The squared Frobenius error is qp_value + Tr(V†V)/2,
            but since Tr(V†V) is constant, the QP value suffices
            for comparing ensembles.
    """
    M = len(target_overlaps)
    H = 2 * np.asarray(gram, dtype=float)
    f = -2 * np.asarray(target_overlaps, dtype=float)

    result = minimize(
        fun=lambda p: 0.5 * p @ H @ p + f @ p,
        x0=np.ones(M) / M,
        jac=lambda p: H @ p + f,
        method='SLSQP',
        bounds=[(0, None)] * M,
        constraints={'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
    )
    return result.x, float(result.fun)
