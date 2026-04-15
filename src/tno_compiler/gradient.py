"""Gradient computation for the MPO-based brickwall circuit compiler.

Exact port of rqcopt-mpo/tn_brickwall_methods.py core functions,
with Trotter-specific logic replaced by generic brickwall partitioning.
"""

import numpy as np
from .mpo_ops import (
    identity_mpo, right_canonicalize, left_canonicalize, canonicalize_tensor,
    merge_gate_with_mpo_pair, split_merged_tensor,
)
from .brickwall import layer_structure, partition_gates


def _merge_layer_right_to_left(mpo_init, gates_in_layer, odd_layer, layer_is_below, max_bond):
    """Exact port of merge_and_truncate_mpo_and_layer_right_to_left."""
    n = len(mpo_init)
    if not odd_layer:
        Q, R = canonicalize_tensor(mpo_init[-1], left=False)
        mpo_res = [Q]
        i = n - 2
    else:
        mpo_res = []
        i = n - 1

    while i - 1 >= 0:
        mpo1, mpo2 = mpo_init[i - 1], mpo_init[i]
        if i == n - 1:
            mpo2_R = mpo2
        else:
            mpo2_R = np.einsum('iabj,jk->iabk', mpo2, R)
        gate = gates_in_layer[int((i - 1) / 2)]
        merged = merge_gate_with_mpo_pair(gate, mpo1, mpo2_R, gate_is_left=layer_is_below)
        T1, T2 = split_merged_tensor(merged, canonical='right', max_bond=max_bond)
        if i - 1 == 0:
            mpo_res += [T2, T1]
        else:
            Q, R = canonicalize_tensor(T1, left=False)
            mpo_res += [T2, Q]
        i -= 2

    if i == 0:
        mpo_R = np.einsum('iabj,jk->iabk', mpo_init[0], R)
        mpo_res += [mpo_R]

    mpo_res = mpo_res[::-1]
    return mpo_res


def _merge_layer_left_to_right(mpo_init, gates_in_layer, odd_layer, layer_is_below, max_bond):
    """Exact port of merge_and_truncate_mpo_and_layer_left_to_right."""
    n = len(mpo_init)
    if not odd_layer:
        Q, R = canonicalize_tensor(mpo_init[0], left=True)
        mpo_res = [Q]
        i = 1
    else:
        mpo_res = []
        i = 0

    while i + 1 < n:
        mpo1, mpo2 = mpo_init[i], mpo_init[i + 1]
        if i == 0:
            mpo1_R = mpo1
        else:
            mpo1_R = np.einsum('ij,jabk->iabk', R, mpo1)
        gate = gates_in_layer[int(i / 2)]
        merged = merge_gate_with_mpo_pair(gate, mpo1_R, mpo2, gate_is_left=layer_is_below)
        T1, T2 = split_merged_tensor(merged, canonical='left', max_bond=max_bond)
        if i + 1 == n - 1:
            mpo_res += [T1, T2]
        else:
            Q, R = canonicalize_tensor(T2, left=True)
            mpo_res += [T1, Q]
        i += 2

    if i == n - 1:
        mpo_R = np.einsum('ij,jabk->iabk', R, mpo_init[-1])
        mpo_res += [mpo_R]

    return mpo_res


def _layer_partial_derivatives(gates_in_layer, layer_odd, upper, lower):
    """Exact port of compute_partial_derivatives_in_layer."""
    n_gates = len(gates_in_layer)

    if layer_odd:
        i_mpo = len(upper)
        R = np.eye(1, dtype=complex)
        L = np.eye(1, dtype=complex)
    else:
        i_mpo = len(upper) - 1
        R = np.einsum('abcd,ecbd->ae', upper[-1], lower[-1])
        L = np.einsum('abcd,acbe->de', upper[0], lower[0])

    R_envs = [R.copy()]
    for gate in reversed(gates_in_layer[1:]):
        A1, A2 = upper[i_mpo - 2], upper[i_mpo - 1]
        B1, B2 = lower[i_mpo - 2], lower[i_mpo - 1]
        R = np.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai',
                       A1, A2, gate, B1, B2, R)
        R_envs.append(R.copy())
        i_mpo -= 2
    R_envs = R_envs[::-1]

    grads = []
    if layer_odd:
        i_mpo = 0
        i_env_mpo = 0
    else:
        i_mpo = 1
        i_env_mpo = 1

    for cut_out_gate in range(n_gates):
        A1, A2 = upper[i_mpo], upper[i_mpo + 1]
        B1, B2 = lower[i_mpo], lower[i_mpo + 1]
        i_mpo += 2

        if cut_out_gate > 0:
            R = R_envs[cut_out_gate]
            L = np.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl',
                           L, upper[i_env_mpo], upper[i_env_mpo + 1],
                           gates_in_layer[cut_out_gate - 1],
                           lower[i_env_mpo], lower[i_env_mpo + 1])
            i_env_mpo += 2

        env = np.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                         L, A1, A2, B1, B2, R)
        grads.append(env)

    return np.stack(grads).conj()


def compute_cost_and_grad(target_arrays, gates, n_qubits, n_layers,
                          max_bond=128, first_odd=True):
    """Compute cost and Euclidean gradient.

    Exact port of compute_full_gradient + cost computation.
    """
    structure = layer_structure(n_qubits, n_layers, first_odd)
    gate_layers = partition_gates(gates, n_qubits, n_layers, first_odd)
    is_odd = [s[0] for s in structure]

    # Left-canonicalize the target MPO (required for correct contraction)
    target_lc = left_canonicalize([a.copy() for a in target_arrays])

    # Upper environments
    bottom_env = identity_mpo(n_qubits)
    top_env = list(target_lc)
    upper_envs = [[a.copy() for a in top_env]]
    merge_right_to_left = True

    for gates_l, odd in zip(reversed(gate_layers[1:]), reversed(is_odd[1:])):
        if merge_right_to_left:
            top_env = _merge_layer_right_to_left(
                top_env, gates_l, odd, True, max_bond)
        else:
            top_env = _merge_layer_left_to_right(
                top_env, gates_l, odd, True, max_bond)
        upper_envs.append([a.copy() for a in top_env])
        merge_right_to_left = not merge_right_to_left
    upper_envs = upper_envs[::-1]

    # Gradient computation
    merge_right_to_left = True
    all_grads = []

    for layer in range(n_layers):
        if layer > 0:
            if merge_right_to_left:
                bottom_env = _merge_layer_right_to_left(
                    bottom_env, gate_layers[layer - 1], is_odd[layer - 1],
                    False, max_bond)
            else:
                bottom_env = _merge_layer_left_to_right(
                    bottom_env, gate_layers[layer - 1], is_odd[layer - 1],
                    False, max_bond)
            merge_right_to_left = not merge_right_to_left

        grads = _layer_partial_derivatives(
            gate_layers[layer], is_odd[layer],
            upper_envs[layer], bottom_env)
        all_grads.append(grads)

    grad = np.concatenate(all_grads, axis=0)
    overlap = np.einsum('abcd,abcd->', all_grads[0][0].conj(), gate_layers[0][0])
    cost = 2.0 - 2.0 * overlap.real / (2 ** n_qubits)

    return float(cost), grad
