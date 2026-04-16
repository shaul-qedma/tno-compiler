"""MPO array operations for the gradient computation.

The gradient engine works with raw numpy arrays in (bond_l, k, b, bond_r)
format. This module also provides the bridge from quimb MPOs to this format.
"""

import numpy as np
from scipy.linalg import rq


def mpo_to_arrays(mpo):
    """Convert a quimb MPO to list of (bond_l, k, b, bond_r) numpy arrays.

    Permutes axes from quimb's arbitrary order to the fixed convention
    needed by the gradient computation.
    """
    arrays = []
    for i in range(mpo.L):
        t = mpo[i]
        inds = t.inds
        ax_k = inds.index(f"k{i}")
        ax_b = inds.index(f"b{i}")
        bond_axes = [j for j in range(len(inds)) if j != ax_k and j != ax_b]

        if i == 0:
            perm = (ax_k, ax_b, bond_axes[0])
            data = t.data.transpose(perm)[np.newaxis, ...]
        elif i == mpo.L - 1:
            perm = (bond_axes[0], ax_k, ax_b)
            data = t.data.transpose(perm)[..., np.newaxis]
        else:
            bond_inds = [inds[j] for j in bond_axes]
            prev_inds = set(mpo[i - 1].inds)
            if bond_inds[0] in prev_inds:
                ax_bl, ax_br = bond_axes[0], bond_axes[1]
            else:
                ax_bl, ax_br = bond_axes[1], bond_axes[0]
            perm = (ax_bl, ax_k, ax_b, ax_br)
            data = t.data.transpose(perm)

        arrays.append(np.array(data, dtype=complex))
    return arrays


def identity_mpo(n_sites):
    T = np.eye(2, dtype=complex)[np.newaxis, :, :, np.newaxis]
    return [T.copy() for _ in range(n_sites)]


def canonicalize_tensor(T, left=True):
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
    A = np.moveaxis(T, 3, 2)
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
