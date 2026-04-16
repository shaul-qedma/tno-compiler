"""Gradient of the overlap Tr(V†U) w.r.t. each circuit gate.

Builds upper/lower environment MPOs by merging circuit layers, then
computes gate-wise partial derivatives within each layer.

Ported from rqcopt-mpo/tn_brickwall_methods.py.
"""

import numpy as np
from .mpo_ops import (
    identity_mpo, canonicalize_tensor,
    merge_gate_with_mpo_pair, split_merged_tensor,
)
from .brickwall import layer_structure, partition_gates


def _merge_layer_right_to_left(mpo, gates, odd, gate_is_left, max_bond):
    n = len(mpo)
    if not odd:
        Q, R = canonicalize_tensor(mpo[-1], left=False)
        result = [Q]
        i = n - 2
    else:
        result = []
        i = n - 1
    while i - 1 >= 0:
        m1, m2 = mpo[i - 1], mpo[i]
        m2 = m2 if i == n - 1 else np.einsum('iabj,jk->iabk', m2, R)
        merged = merge_gate_with_mpo_pair(
            gates[int((i - 1) / 2)], m1, m2, gate_is_left)
        T1, T2 = split_merged_tensor(merged, 'right', max_bond)
        if i - 1 == 0:
            result += [T2, T1]
        else:
            Q, R = canonicalize_tensor(T1, left=False)
            result += [T2, Q]
        i -= 2
    if i == 0:
        result.append(np.einsum('iabj,jk->iabk', mpo[0], R))
    result.reverse()
    return result


def _merge_layer_left_to_right(mpo, gates, odd, gate_is_left, max_bond):
    n = len(mpo)
    if not odd:
        Q, R = canonicalize_tensor(mpo[0], left=True)
        result = [Q]
        i = 1
    else:
        result = []
        i = 0
    while i + 1 < n:
        m1 = mpo[i] if i == 0 else np.einsum('ij,jabk->iabk', R, mpo[i])
        merged = merge_gate_with_mpo_pair(
            gates[int(i / 2)], m1, mpo[i + 1], gate_is_left)
        T1, T2 = split_merged_tensor(merged, 'left', max_bond)
        if i + 1 == n - 1:
            result += [T1, T2]
        else:
            Q, R = canonicalize_tensor(T2, left=True)
            result += [T1, Q]
        i += 2
    if i == n - 1:
        result.append(np.einsum('ij,jabk->iabk', R, mpo[-1]))
    return result


def _layer_envs(gates, odd, upper, lower):
    """Gate environments within a single layer."""
    n_gates = len(gates)
    if odd:
        i_mpo = len(upper)
        R, L = np.eye(1, dtype=complex), np.eye(1, dtype=complex)
    else:
        i_mpo = len(upper) - 1
        R = np.einsum('abcd,ecbd->ae', upper[-1], lower[-1])
        L = np.einsum('abcd,acbe->de', upper[0], lower[0])

    R_envs = [R.copy()]
    for gate in reversed(gates[1:]):
        i_mpo -= 2
        R = np.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai',
                       upper[i_mpo], upper[i_mpo + 1], gate,
                       lower[i_mpo], lower[i_mpo + 1], R)
        R_envs.append(R.copy())
    R_envs.reverse()

    grads = []
    i_mpo = 0 if odd else 1
    i_env = i_mpo
    for g in range(n_gates):
        A1, A2 = upper[i_mpo], upper[i_mpo + 1]
        B1, B2 = lower[i_mpo], lower[i_mpo + 1]
        i_mpo += 2
        if g > 0:
            R = R_envs[g]
            L = np.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl',
                           L, upper[i_env], upper[i_env + 1], gates[g - 1],
                           lower[i_env], lower[i_env + 1])
            i_env += 2
        grads.append(np.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                                L, A1, A2, B1, B2, R))

    return np.stack(grads).conj()


def compute_cost_and_grad(target_arrays, gates, n_qubits, n_layers,
                          max_bond=128, first_odd=True):
    """Compute Frobenius cost and Euclidean gradient w.r.t. each gate.

    Args:
        target_arrays: list of (bl, k, b, br) arrays (target MPO, stores V†).
        gates: flat list of (2,2,2,2) arrays (circuit gates).

    Returns:
        cost: 2 - 2·Re(Tr(V†U)) / 2^n
        grad: array (n_gates, 2, 2, 2, 2)
    """
    structure = layer_structure(n_qubits, n_layers, first_odd)
    gate_layers = partition_gates(gates, n_qubits, n_layers, first_odd)
    is_odd = [s[0] for s in structure]

    # Upper environments: target with circuit layers merged from above
    top = [a.copy() for a in target_arrays]
    upper_envs = [list(top)]
    sweep_right = False  # first merge is right-to-left
    for gl, odd in zip(reversed(gate_layers[1:]), reversed(is_odd[1:])):
        merge = _merge_layer_right_to_left if sweep_right else _merge_layer_left_to_right
        top = merge(top, gl, odd, True, max_bond)
        upper_envs.append([a.copy() for a in top])
        sweep_right = not sweep_right
    upper_envs.reverse()

    # Lower environments + gradients
    bottom = identity_mpo(n_qubits)
    sweep_right = False
    all_grads = []
    for layer in range(n_layers):
        if layer > 0:
            merge = _merge_layer_right_to_left if sweep_right else _merge_layer_left_to_right
            bottom = merge(bottom, gate_layers[layer - 1], is_odd[layer - 1],
                           False, max_bond)
            sweep_right = not sweep_right
        all_grads.append(_layer_envs(gate_layers[layer], is_odd[layer],
                                     upper_envs[layer], bottom))

    grad = np.concatenate(all_grads, axis=0)
    overlap = np.einsum('abcd,abcd->', all_grads[0][0].conj(), gate_layers[0][0])
    cost = 2.0 - 2.0 * overlap.real / (2 ** n_qubits)
    return float(cost), grad
