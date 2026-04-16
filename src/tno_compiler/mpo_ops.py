"""MPO primitive operations: decomposition, splitting, merging, canonical forms.

Ported from rqcopt-mpo/tn_helpers.py. Convention: each tensor has shape
(bond_l, phys_up, phys_dn, bond_r) with dummy dim-1 bonds at boundaries.
"""

import numpy as np
from scipy.linalg import rq


def identity_mpo(n_sites):
    """Identity MPO: each site is delta(k, b) with bond dim 1."""
    T = np.eye(2, dtype=complex)[np.newaxis, :, :, np.newaxis]
    return [T.copy() for _ in range(n_sites)]


def matrix_to_mpo(U):
    """Decompose a 2^n x 2^n matrix into an MPO via successive SVDs.

    Ported from rqcopt-mpo get_mpo_from_matrix.
    """
    n = int(round(np.log2(U.shape[0])))
    A = U.reshape((2, 2) * n)
    tensors = []
    for site in range(1, n):
        n_remaining = n - site + 1
        if site == 1:
            A = np.moveaxis(A, n_remaining, 1)
        else:
            A = np.moveaxis(A, n_remaining + 1, 2)
        shape = A.shape
        lim = 2 if site == 1 else 3
        mat = A.reshape(int(np.prod(shape[:lim])), int(np.prod(shape[lim:])))
        u, s, v = np.linalg.svd(mat, full_matrices=False)
        tensors.append(u.reshape(shape[:lim] + (u.shape[-1],)))
        A = (np.diag(s) @ v).reshape((u.shape[-1],) + shape[lim:])
        if site == n - 1:
            tensors.append(A)
    tensors[0] = tensors[0][np.newaxis, ...]
    tensors[-1] = tensors[-1][..., np.newaxis]
    return tensors


def trace_mpo(mpo):
    """Tr(MPO) by tracing physical indices then contracting bonds."""
    traced = [np.einsum('iaaj->ij', T) for T in mpo]
    result = traced[0]
    for T in traced[1:]:
        result = result @ T
    return np.einsum('ii->', result)


def canonicalize_tensor(T, left=True):
    """QR/RQ-canonicalize a single MPO tensor. Returns (Q, R)."""
    shape = T.shape
    if left:
        mat = T.reshape(-1, shape[-1])
        Q, R = np.linalg.qr(mat, mode='reduced')
        return Q.reshape(shape[:-1] + (Q.shape[-1],)), R
    else:
        mat = T.reshape(shape[0], -1)
        R, Q = rq(mat, mode='economic')
        return Q.reshape((Q.shape[0],) + shape[1:]), R


def split_merged_tensor(T, canonical='left', max_bond=128):
    """Split a merged 2-site tensor into two MPO tensors via SVD.

    Input: (bl, k1, b1, k2, b2, br) from merge_gate_with_mpo_pair.
    Output: T1 = (bl, k1, b1, d), T2 = (d, k2, b2, br).
    """
    assert T.ndim == 6
    A = np.moveaxis(T, 3, 2)  # -> (bl, k1, k2, b1, b2, br)
    shape = A.shape
    mat = A.reshape(shape[0] * shape[1] * shape[2],
                    shape[3] * shape[4] * shape[5])
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    if np.isfinite(u).all() and np.isfinite(s).all() and np.isfinite(v).all():
        if max_bond < len(s):
            u, s, v = u[:, :max_bond], s[:max_bond], v[:max_bond, :]
        if canonical == 'left':
            T1 = u.reshape(shape[:3] + (u.shape[-1],))
            T2 = (np.diag(s) @ v).reshape((len(s),) + shape[3:])
        else:
            T1 = (u @ np.diag(s)).reshape(shape[:3] + (len(s),))
            T2 = v.reshape((v.shape[0],) + shape[3:])
    else:
        if canonical == 'left':
            T1, T2 = np.linalg.qr(mat, mode='reduced')
        else:
            T2, T1 = rq(mat, mode='economic')
        T1 = T1.reshape(shape[:3] + (T1.shape[-1],))
        T2 = T2.reshape((T2.shape[0],) + shape[3:])
    return T1, T2


def merge_gate_with_mpo_pair(gate, mpo1, mpo2, gate_is_left=True):
    """Merge a 2-qubit gate (2,2,2,2) with adjacent MPO tensors.

    gate_is_left=True: gate below MPO (circuit side).
    gate_is_left=False: gate above MPO (environment side).

    Returns 6-index tensor (bl, k1, b1, k2, b2, br).
    """
    if gate_is_left:
        return np.einsum('iabc,cdef,begh->iadghf', mpo1, mpo2, gate)
    else:
        return np.einsum('abcd,icef,fdgh->iabegh', gate, mpo1, mpo2)
