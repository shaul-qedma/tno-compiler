"""MPO primitives: decomposition, merging, splitting, canonical forms.

Each tensor has shape (bond_l, phys_up, phys_dn, bond_r) with
dummy dim-1 bonds at boundaries.
"""

import numpy as np
from scipy.linalg import rq


def identity_mpo(n_sites):
    T = np.eye(2, dtype=complex)[np.newaxis, :, :, np.newaxis]
    return [T.copy() for _ in range(n_sites)]


def matrix_to_mpo(U):
    """Decompose a 2^n × 2^n matrix into an MPO via successive SVDs."""
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
    traced = [np.einsum('iaaj->ij', T) for T in mpo]
    result = traced[0]
    for T in traced[1:]:
        result = result @ T
    return np.einsum('ii->', result)


def canonicalize_tensor(T, left=True):
    """QR/RQ canonicalize. Returns (isometric_tensor, remainder)."""
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
    """Split (bl, k1, b1, k2, b2, br) → T1(bl,k1,b1,d), T2(d,k2,b2,br)."""
    A = np.moveaxis(T, 3, 2)  # (bl, k1, k2, b1, b2, br)
    shape = A.shape
    mat = A.reshape(shape[0] * shape[1] * shape[2],
                    shape[3] * shape[4] * shape[5])
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    if max_bond < len(s):
        u, s, v = u[:, :max_bond], s[:max_bond], v[:max_bond, :]
    if canonical == 'left':
        T1 = u.reshape(shape[:3] + (u.shape[-1],))
        T2 = (np.diag(s) @ v).reshape((len(s),) + shape[3:])
    else:
        T1 = (u @ np.diag(s)).reshape(shape[:3] + (len(s),))
        T2 = v.reshape((v.shape[0],) + shape[3:])
    return T1, T2


def merge_gate_with_mpo_pair(gate, mpo1, mpo2, gate_is_left=True):
    """Merge a (2,2,2,2) gate with two adjacent MPO tensors.
    Returns (bl, k1, b1, k2, b2, br)."""
    if gate_is_left:
        return np.einsum('iabc,cdef,begh->iadghf', mpo1, mpo2, gate,
                         optimize=True)
    else:
        return np.einsum('abcd,icef,fdgh->iabegh', gate, mpo1, mpo2,
                         optimize=True)
