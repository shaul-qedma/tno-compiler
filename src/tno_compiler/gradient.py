"""Gate environment computation and polar sweep for MPO-based compilation.

Environment computation ported from rqcopt-mpo/tn_brickwall_methods.py.
Polar sweep implements the Gibbs & Cincio (2025) algorithm: alternating
direction sweeps with incremental environment updates.
"""

import numpy as np
from .mpo_ops import (
    identity_mpo, canonicalize_tensor,
    merge_gate_with_mpo_pair, split_merged_tensor,
)
from .brickwall import brickwall_ansatz_gates


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
    """Environments for each gate in one layer. Returns conjugated envs.

    Single left-to-right pass. Used by compute_cost_and_grad.
    """
    return _layer_envs_onepass(gates, odd, upper, lower)


def _layer_envs_onepass(gates, odd, upper, lower):
    """One left-to-right pass collecting environments."""
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
                       lower[i_mpo], lower[i_mpo + 1], R,
                       optimize=True)
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
                           lower[i_env], lower[i_env + 1],
                           optimize=True)
            i_env += 2
        grads.append(np.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                                L, A1, A2, B1, B2, R,
                                optimize=True))

    return np.stack(grads).conj()


def _gate_env(g_idx, gates, odd, upper, lower):
    """Environment for a single gate given current L/R context."""
    n_gates = len(gates)
    i_start = 0 if odd else 1

    # Build L from left edge to g_idx
    if odd:
        L = np.eye(1, dtype=complex)
    else:
        L = np.einsum('abcd,acbe->de', upper[0], lower[0])
    i_mpo = i_start
    for g in range(g_idx):
        L = np.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl',
                       L, upper[i_mpo], upper[i_mpo + 1], gates[g],
                       lower[i_mpo], lower[i_mpo + 1],
                       optimize=True)
        i_mpo += 2

    # Build R from right edge to g_idx
    if odd:
        R = np.eye(1, dtype=complex)
    else:
        R = np.einsum('abcd,ecbd->ae', upper[-1], lower[-1])
    i_mpo = len(upper) - (1 if not odd else 0)
    for g in reversed(range(g_idx + 1, n_gates)):
        i_mpo -= 2
        R = np.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai',
                       upper[i_mpo], upper[i_mpo + 1], gates[g],
                       lower[i_mpo], lower[i_mpo + 1], R,
                       optimize=True)

    # Contract environment for gate g_idx
    i_mpo = i_start + 2 * g_idx
    A1, A2 = upper[i_mpo], upper[i_mpo + 1]
    B1, B2 = lower[i_mpo], lower[i_mpo + 1]
    env = np.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                     L, A1, A2, B1, B2, R, optimize=True)
    return env.conj()


def _optimize_layer_inplace(gates, odd, upper, lower, n_inner=3):
    """Optimize all gates in a layer via multi-pass polar sweeps.

    Multiple left-right passes within the layer, updating each gate
    and recomputing its environment from current neighbors.
    Matches Gibbs-Cincio (2025) within-layer convergence.
    """
    n_gates = len(gates)
    if n_gates <= 1:
        env = _gate_env(0, gates, odd, upper, lower)
        u, _, vh = np.linalg.svd(env.reshape(4, 4), full_matrices=False)
        gates[0] = (u @ vh).reshape(2, 2, 2, 2)
        return

    for _ in range(n_inner):
        # Left-to-right pass
        for g in range(n_gates):
            env = _gate_env(g, gates, odd, upper, lower)
            u, _, vh = np.linalg.svd(env.reshape(4, 4), full_matrices=False)
            gates[g] = (u @ vh).reshape(2, 2, 2, 2)
        # Right-to-left pass
        for g in reversed(range(n_gates)):
            env = _gate_env(g, gates, odd, upper, lower)
            u, _, vh = np.linalg.svd(env.reshape(4, 4), full_matrices=False)
            gates[g] = (u @ vh).reshape(2, 2, 2, 2)


def polar_sweep(target_arrays, gates, n_qubits, n_layers,
                max_bond=128, first_odd=True, n_inner=3):
    """One full polar decomposition sweep (Gibbs-Cincio 2025).

    Down sweep (L-1 → 0): upper env starts at target (exact), absorbs
    updated layers incrementally. Lower env pre-built from original gates.
    At each layer: multi-pass within-layer convergence.

    Up sweep (0 → L-1): lower env starts at identity (exact), absorbs
    updated layers incrementally. Upper env pre-built from current gates.
    At each layer: multi-pass within-layer convergence.

    MPO resets: upper env resets to target at start of each half-sweep.
    Lower env resets to identity at start of each half-sweep.

    Modifies gates in-place. Returns cost.
    """
    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    L = len(structure)
    gate_layers = _partition(gates, structure)

    # --- Down sweep (L-1 → 0) ---
    # Pre-build lower envs from original gates
    lower_envs = [identity_mpo(n_qubits)]
    sr = False
    for k in range(L - 1):
        merge = _merge_layer_right_to_left if sr else _merge_layer_left_to_right
        lower_envs.append(merge(lower_envs[-1], gate_layers[k],
                                structure[k][0], False, max_bond))
        sr = not sr

    # Sweep top to bottom with incremental upper env
    top = [a.copy() for a in target_arrays]  # exact reset
    sr = False
    for layer in reversed(range(L)):
        odd = structure[layer][0]
        _optimize_layer_inplace(gate_layers[layer], odd, top,
                                lower_envs[layer], n_inner)
        if layer > 0:
            merge = _merge_layer_right_to_left if sr else _merge_layer_left_to_right
            top = merge(top, gate_layers[layer], odd, True, max_bond)
            sr = not sr

    # --- Up sweep (0 → L-1) ---
    # Pre-build upper envs from current (post-down-sweep) gates
    upper_envs = [[a.copy() for a in target_arrays]]
    sr = False
    for k in reversed(range(1, L)):
        merge = _merge_layer_right_to_left if sr else _merge_layer_left_to_right
        upper_envs.append(merge(upper_envs[-1], gate_layers[k],
                                structure[k][0], True, max_bond))
        sr = not sr
    upper_envs.reverse()

    # Sweep bottom to top with incremental lower env
    bottom = identity_mpo(n_qubits)  # exact reset
    sr = False
    for layer in range(L):
        odd = structure[layer][0]
        _optimize_layer_inplace(gate_layers[layer], odd,
                                upper_envs[layer], bottom, n_inner)
        if layer < L - 1:
            merge = _merge_layer_right_to_left if sr else _merge_layer_left_to_right
            bottom = merge(bottom, gate_layers[layer], odd, False, max_bond)
            sr = not sr

    _flatten_into(gate_layers, gates)

    cost, _ = compute_cost_and_grad(target_arrays, gates, n_qubits, n_layers,
                                     max_bond, first_odd)
    return cost


def _update_gates_polar(layer_gates, envs):
    """Update each gate in a layer via polar decomposition of its environment."""
    for i in range(len(layer_gates)):
        env = envs[i].reshape(4, 4)
        u, _, vh = np.linalg.svd(env, full_matrices=False)
        layer_gates[i] = (u @ vh).reshape(2, 2, 2, 2)


def _partition(gates, structure):
    """Partition flat gate list into per-layer mutable lists."""
    result, idx = [], 0
    for _, pairs in structure:
        result.append(list(gates[idx:idx + len(pairs)]))
        idx += len(pairs)
    return result


def _flatten_into(gate_layers, gates):
    """Write per-layer gates back to flat list."""
    idx = 0
    for layer in gate_layers:
        for i, g in enumerate(layer):
            gates[idx + i] = g
        idx += len(layer)


def compute_cost_and_grad(target_arrays, gates, n_qubits, n_layers,
                          max_bond=128, first_odd=True):
    """Frobenius cost 2 - 2·Re(Tr(V†U))/2^n and its gradient."""
    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    gate_layers = _partition(gates, structure)

    # Build upper environments
    top = [a.copy() for a in target_arrays]
    upper_envs = [list(top)]
    sweep_right = False
    for gl, (odd, _) in zip(reversed(gate_layers[1:]), reversed(structure[1:])):
        merge = _merge_layer_right_to_left if sweep_right else _merge_layer_left_to_right
        top = merge(top, gl, odd, True, max_bond)
        upper_envs.append([a.copy() for a in top])
        sweep_right = not sweep_right
    upper_envs.reverse()

    # Build lower environments and compute gradients
    bottom = identity_mpo(n_qubits)
    sweep_right = False
    all_grads = []
    for layer, (odd, _) in enumerate(structure):
        if layer > 0:
            merge = _merge_layer_right_to_left if sweep_right else _merge_layer_left_to_right
            bottom = merge(bottom, gate_layers[layer - 1],
                           structure[layer - 1][0], False, max_bond)
            sweep_right = not sweep_right
        all_grads.append(_layer_envs(gate_layers[layer], odd,
                                     upper_envs[layer], bottom))

    grad = np.concatenate(all_grads, axis=0)
    overlap = np.einsum('abcd,abcd->', all_grads[0][0].conj(), gate_layers[0][0])
    return float(2.0 - 2.0 * overlap.real / (2 ** n_qubits)), grad
