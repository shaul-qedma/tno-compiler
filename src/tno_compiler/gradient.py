"""Gate environment computation and polar sweep for MPO-based compilation.

Environment computation ported from rqcopt-mpo/tn_brickwall_methods.py.
Polar sweep implements the Gibbs & Cincio (2025) algorithm: alternating
direction sweeps with incremental environment updates.

All hot-path operations use JAX for JIT compilation and fused contractions.
"""

import jax.numpy as jnp
import numpy as np
from .jax_ops import (
    _EYE1,
    identity_mpo, canonicalize_tensor,
    merge_gate_with_mpo_pair, split_merged_tensor,
    contract_R, contract_L, compute_gate_env,
    env_and_polar_update,
    absorb_R_left, absorb_R_right,
    init_L, init_R,
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
        m2 = m2 if i == n - 1 else absorb_R_right(m2, R)
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
        result.append(absorb_R_right(mpo[0], R))
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
        m1 = mpo[i] if i == 0 else absorb_R_left(R, mpo[i])
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
        result.append(absorb_R_left(R, mpo[-1]))
    return result


def _layer_envs(gates, odd, upper, lower):
    """Environments for each gate in one layer. Returns conjugated envs."""
    return _layer_envs_onepass(gates, odd, upper, lower)


def _layer_envs_onepass(gates, odd, upper, lower):
    """One left-to-right pass collecting environments."""
    n_gates = len(gates)
    if odd:
        i_mpo = len(upper)
        R, L = _EYE1, _EYE1
    else:
        i_mpo = len(upper) - 1
        R = init_R(upper[-1], lower[-1])
        L = init_L(upper[0], lower[0])

    R_envs = [R]
    for gate in reversed(gates[1:]):
        i_mpo -= 2
        R = contract_R(upper[i_mpo], upper[i_mpo + 1], gate,
                        lower[i_mpo], lower[i_mpo + 1], R)
        R_envs.append(R)
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
            L = contract_L(L, upper[i_env], upper[i_env + 1], gates[g - 1],
                           lower[i_env], lower[i_env + 1])
            i_env += 2
        grads.append(compute_gate_env(L, A1, A2, B1, B2, R))

    return jnp.stack(grads)


def _optimize_layer_inplace(gates, odd, upper, lower, n_inner=3,
                             drop_rate=0.0, rng=None):
    """Optimize all gates in a layer via multi-pass incremental sweeps.

    Uses fused env_and_polar_update: one JIT call per gate instead of
    separate environment contraction + SVD.

    `drop_rate` (0 to disable): per-gate probability of skipping the
    polar update on each visit, to break basin-commit via the coordinated
    equilibrium Gauss-Seidel converges to.
    """
    n_gates = len(gates)
    i_start = 0 if odd else 1
    drop = drop_rate > 0.0

    def _init_LR():
        if odd:
            return _EYE1, _EYE1
        return init_L(upper[0], lower[0]), init_R(upper[-1], lower[-1])

    for _ in range(n_inner):
        # --- Left-to-right pass ---
        L_init, R_init = _init_LR()
        R_envs = [R_init]
        i_mpo = len(upper) - (1 if not odd else 0)
        for g in reversed(range(1, n_gates)):
            i_mpo -= 2
            R_envs.append(contract_R(upper[i_mpo], upper[i_mpo + 1],
                                      gates[g], lower[i_mpo],
                                      lower[i_mpo + 1], R_envs[-1]))
        R_envs.reverse()

        L = L_init
        for g in range(n_gates):
            i_mpo = i_start + 2 * g
            if not (drop and rng.random() < drop_rate):
                gates[g] = env_and_polar_update(
                    L, upper[i_mpo], upper[i_mpo + 1],
                    lower[i_mpo], lower[i_mpo + 1], R_envs[g])
            if g < n_gates - 1:
                L = contract_L(L, upper[i_mpo], upper[i_mpo + 1],
                               gates[g], lower[i_mpo], lower[i_mpo + 1])

        # --- Right-to-left pass ---
        L_init, R_init = _init_LR()
        L_envs = [L_init]
        i_mpo = i_start
        for g in range(n_gates - 1):
            L_envs.append(contract_L(L_envs[-1], upper[i_mpo],
                                      upper[i_mpo + 1], gates[g],
                                      lower[i_mpo], lower[i_mpo + 1]))
            i_mpo += 2

        R = R_init
        for g in reversed(range(n_gates)):
            i_mpo = i_start + 2 * g
            if not (drop and rng.random() < drop_rate):
                gates[g] = env_and_polar_update(
                    L_envs[g], upper[i_mpo], upper[i_mpo + 1],
                    lower[i_mpo], lower[i_mpo + 1], R)
            if g > 0:
                R = contract_R(upper[i_mpo], upper[i_mpo + 1],
                               gates[g], lower[i_mpo],
                               lower[i_mpo + 1], R)


def polar_sweep(target_arrays, gates, n_qubits, n_layers,
                max_bond=128, first_odd=True, n_inner=3,
                drop_rate=0.0, rng=None):
    """One full polar decomposition sweep (Gibbs-Cincio 2025).

    Expects JAX arrays. Caller (polar_sweeps) handles conversion.
    Modifies gates in-place. Returns cost.
    `drop_rate`, `rng`: passed through to `_optimize_layer_inplace`.
    """
    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    L = len(structure)
    gate_layers = _partition(gates, structure)

    # --- Down sweep (L-1 → 0) ---
    lower_envs = [identity_mpo(n_qubits)]
    sr = False
    for k in range(L - 1):
        merge = _merge_layer_right_to_left if sr else _merge_layer_left_to_right
        lower_envs.append(merge(lower_envs[-1], gate_layers[k],
                                structure[k][0], False, max_bond))
        sr = not sr

    top = list(target_arrays)
    sr = False
    for layer in reversed(range(L)):
        odd = structure[layer][0]
        _optimize_layer_inplace(gate_layers[layer], odd, top,
                                lower_envs[layer], n_inner,
                                drop_rate=drop_rate, rng=rng)
        if layer > 0:
            merge = _merge_layer_right_to_left if sr else _merge_layer_left_to_right
            top = merge(top, gate_layers[layer], odd, True, max_bond)
            sr = not sr

    # --- Up sweep (0 → L-1) ---
    upper_envs = [list(target_arrays)]
    sr = False
    for k in reversed(range(1, L)):
        merge = _merge_layer_right_to_left if sr else _merge_layer_left_to_right
        upper_envs.append(merge(upper_envs[-1], gate_layers[k],
                                structure[k][0], True, max_bond))
        sr = not sr
    upper_envs.reverse()

    bottom = identity_mpo(n_qubits)
    sr = False
    for layer in range(L):
        odd = structure[layer][0]
        _optimize_layer_inplace(gate_layers[layer], odd,
                                upper_envs[layer], bottom, n_inner,
                                drop_rate=drop_rate, rng=rng)
        if layer < L - 1:
            merge = _merge_layer_right_to_left if sr else _merge_layer_left_to_right
            bottom = merge(bottom, gate_layers[layer], odd, False, max_bond)
            sr = not sr

    _flatten_into(gate_layers, gates)

    # Compute final cost (reuses the JAX arrays already in gates)
    cost = _compute_cost(target_arrays, gates, n_qubits, n_layers,
                         max_bond, first_odd)
    return cost


def _compute_cost(target_arrays, gates, n_qubits, n_layers,
                  max_bond, first_odd):
    """Cost only (no gradient). Avoids redundant environment build."""
    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    gate_layers = _partition(gates, structure)

    top = list(target_arrays)
    sweep_right = False
    for gl, (odd, _) in zip(reversed(gate_layers[1:]), reversed(structure[1:])):
        merge = _merge_layer_right_to_left if sweep_right else _merge_layer_left_to_right
        top = merge(top, gl, odd, True, max_bond)
        sweep_right = not sweep_right

    bottom = identity_mpo(n_qubits)
    odd = structure[0][0]
    envs = _layer_envs(gate_layers[0], odd, top, bottom)
    overlap = jnp.einsum('abcd,abcd->', envs[0].conj(), gate_layers[0][0])
    return float(2.0 - 2.0 * overlap.real / (2.0 ** n_qubits))


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
    """Frobenius cost 2 - 2·Re(Tr(V†U))/2^n and its gradient.

    Accepts numpy or JAX arrays. Returns (float, numpy array).
    """
    target_jax = [jnp.asarray(a) for a in target_arrays]
    gates_jax = [jnp.asarray(g) for g in gates]

    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    gate_layers = _partition(gates_jax, structure)

    top = list(target_jax)
    upper_envs = [list(top)]
    sweep_right = False
    for gl, (odd, _) in zip(reversed(gate_layers[1:]), reversed(structure[1:])):
        merge = _merge_layer_right_to_left if sweep_right else _merge_layer_left_to_right
        top = merge(top, gl, odd, True, max_bond)
        upper_envs.append(list(top))
        sweep_right = not sweep_right
    upper_envs.reverse()

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

    grad = np.asarray(jnp.concatenate(all_grads, axis=0))
    overlap = jnp.einsum('abcd,abcd->', all_grads[0][0].conj(), gate_layers[0][0])
    return float(2.0 - 2.0 * overlap.real / (2.0 ** n_qubits)), grad
