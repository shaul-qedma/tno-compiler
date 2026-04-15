"""Gradient computation for the MPO-based brickwall circuit compiler.

Direct port of rqcopt-mpo/tn_brickwall_methods.py core functions,
with Trotter-specific logic removed.
"""

import numpy as np
from .mpo_ops import (
    identity_mpo, right_canonicalize, canonicalize_tensor,
    merge_gate_with_mpo_pair, split_merged_tensor,
)
from .brickwall import layer_structure, partition_gates


def _merge_layer_right_to_left(mpo, gates, odd, gate_is_left, max_bond):
    """Port of merge_and_truncate_mpo_and_layer_right_to_left."""
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
        if i == n - 1:
            m2_R = m2
        else:
            m2_R = np.einsum('iabj,jk->iabk', m2, R)
        gate = gates[int((i - 1) / 2) if odd else int((i - 1) / 2)]
        # Gate index: for odd layers, pairs at (0,1),(2,3),... -> gate 0 at i=0,1
        # For even layers, pairs at (1,2),(3,4),... -> gate 0 at i=1,2
        g_idx = (i - 1) // 2 if odd else (i - 2) // 2
        merged = merge_gate_with_mpo_pair(gates[g_idx], m1, m2_R, gate_is_left)
        T1, T2 = split_merged_tensor(merged, canonical='right', max_bond=max_bond)

        if i - 2 < (0 if odd else 1):
            result.extend([T2, T1])
        else:
            Q, R = canonicalize_tensor(T1, left=False)
            result.extend([T2, Q])

        i -= 2

    if not odd and i == 0:
        m_R = np.einsum('iabj,jk->iabk', mpo[0], R)
        result.append(m_R)

    result.reverse()
    return result


def _merge_layer_left_to_right(mpo, gates, odd, gate_is_left, max_bond):
    """Port of merge_and_truncate_mpo_and_layer_left_to_right."""
    n = len(mpo)
    R = None

    if not odd:
        Q, R = canonicalize_tensor(mpo[0], left=True)
        result = [Q]
        i = 1
    else:
        result = []
        i = 0

    while i + 1 < n:
        m1 = mpo[i]
        if i == 0:
            m1_R = m1
        else:
            m1_R = np.einsum('ij,jabk->iabk', R, m1)

        g_idx = i // 2 if odd else (i - 1) // 2
        merged = merge_gate_with_mpo_pair(gates[g_idx], m1_R, mpo[i + 1], gate_is_left)
        T1, T2 = split_merged_tensor(merged, canonical='left', max_bond=max_bond)

        if i + 1 == n - 1:
            result.extend([T1, T2])
        else:
            Q, R = canonicalize_tensor(T2, left=True)
            result.extend([T1, Q])

        i += 2

    if i == n - 1:
        m_R = np.einsum('ij,jabk->iabk', R, mpo[-1])
        result.append(m_R)

    return result


def _layer_partial_derivatives(gates, odd, upper, lower):
    """Port of compute_partial_derivatives_in_layer."""
    n_gates = len(gates)

    if odd:
        i_mpo = len(upper)
        R = np.eye(1, dtype=complex)
        L = np.eye(1, dtype=complex)
    else:
        i_mpo = len(upper) - 1
        R = np.einsum('abcd,ecbd->ae', upper[-1], lower[-1])
        L = np.einsum('abcd,acbe->de', upper[0], lower[0])

    # Build right environments
    R_envs = [R.copy()]
    for g in reversed(range(1, n_gates)):
        A1, A2 = upper[i_mpo - 2], upper[i_mpo - 1]
        B1, B2 = lower[i_mpo - 2], lower[i_mpo - 1]
        R = np.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai',
                       A1, A2, gates[g], B1, B2, R)
        R_envs.append(R.copy())
        i_mpo -= 2
    R_envs.reverse()

    # Sweep left to right
    grads = []
    i_mpo = 0 if odd else 1
    i_env = i_mpo
    for g_idx in range(n_gates):
        A1, A2 = upper[i_mpo], upper[i_mpo + 1]
        B1, B2 = lower[i_mpo], lower[i_mpo + 1]
        i_mpo += 2

        if g_idx > 0:
            R = R_envs[g_idx]
            L = np.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl',
                           L,
                           upper[i_env], upper[i_env + 1],
                           gates[g_idx - 1],
                           lower[i_env], lower[i_env + 1])
            i_env += 2

        env = np.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                         L, A1, A2, B1, B2, R)
        grads.append(env)

    return np.stack(grads).conj()


def compute_cost_and_grad(target_arrays, gates, n_qubits, n_layers,
                          max_bond=128, first_odd=True):
    """Compute cost and Euclidean gradient.

    Port of compute_full_gradient + get_riemannian_gradient_and_cost_function
    (Euclidean gradient only; Riemannian projection done in optim.py).

    Args:
        target_arrays: list of (bl, k, b, br) arrays.
        gates: flat list of (2,2,2,2) arrays.
        n_qubits, n_layers, max_bond, first_odd: circuit config.

    Returns:
        cost: Frobenius distance 2 - 2*Re(overlap)/2^n.
        grad: array (n_gates, 2, 2, 2, 2).
    """
    structure = layer_structure(n_qubits, n_layers, first_odd)
    gate_layers = partition_gates(gates, n_qubits, n_layers, first_odd)
    is_odd = [s[0] for s in structure]

    # Upper environments: merge circuit layers into target from top
    top = right_canonicalize([a.copy() for a in target_arrays])
    upper_envs = [list(top)]
    merge_right_to_left = True

    for layer in reversed(range(1, n_layers)):
        if merge_right_to_left:
            top = _merge_layer_right_to_left(
                top, gate_layers[layer], is_odd[layer], True, max_bond)
        else:
            top = _merge_layer_left_to_right(
                top, gate_layers[layer], is_odd[layer], True, max_bond)
        upper_envs.append([a.copy() for a in top])
        merge_right_to_left = not merge_right_to_left
    upper_envs.reverse()

    # Lower environments and gradients
    bottom = identity_mpo(n_qubits)
    merge_right_to_left = True
    all_grads = []

    for layer in range(n_layers):
        if layer > 0:
            if merge_right_to_left:
                bottom = _merge_layer_right_to_left(
                    bottom, gate_layers[layer - 1], is_odd[layer - 1],
                    False, max_bond)
            else:
                bottom = _merge_layer_left_to_right(
                    bottom, gate_layers[layer - 1], is_odd[layer - 1],
                    False, max_bond)
            merge_right_to_left = not merge_right_to_left

        grads = _layer_partial_derivatives(
            gate_layers[layer], is_odd[layer],
            upper_envs[layer], bottom)
        all_grads.append(grads)

    grad = np.concatenate(all_grads, axis=0)
    overlap = np.einsum('abcd,abcd->', all_grads[0][0].conj(), gate_layers[0][0])
    cost = 2.0 - 2.0 * overlap.real / (2 ** n_qubits)

    return float(cost), grad
