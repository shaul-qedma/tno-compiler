"""Polar-decomposition sweeps (Gibbs & Cincio 2025) over a brickwall
ansatz.

The compilation landscape is non-convex. Given a target MPO and an
initial gate set, `polar_sweep_batched` runs an alternating down-sweep
and up-sweep over the brickwall layers. Within each layer, a local
Gauss-Seidel sweep updates every gate to the closest unitary to its
exact-gradient environment (polar decomposition of the 4×4 env).

Two entry points:

  polar_sweep_batched - primary. Vectorizes over an ensemble batch dim
      so B independent members (different inits) optimize in one call,
      sharing the target and JIT cache. This is what `compile_ensemble`
      uses.

  compute_cost_and_grad - exposes the gate-environment gradient on the
      unbatched path. Kept for the Riemannian-ADAM method in optim.py,
      which needs a scalar cost + explicit gradient per call.

Environment computation ported from rqcopt-mpo/tn_brickwall_methods.py.
All hot-path tensor ops are JAX JIT-compiled; see jax_ops.py for the
kernels and their vmap'd siblings.
"""

import jax.numpy as jnp
import numpy as np

from .brickwall import brickwall_ansatz_gates
from .jax_ops import (
    _EYE1,
    # unbatched kernels (used by ADAM path via compute_cost_and_grad)
    identity_mpo, canonicalize_tensor,
    merge_gate_with_mpo_pair, split_merged_tensor,
    contract_R, contract_L, compute_gate_env,
    absorb_R_left, absorb_R_right,
    init_L, init_R,
    # batched kernels (used by polar_sweep_batched)
    identity_mpo_batched,
    contract_R_batched, contract_L_batched,
    env_and_polar_update_batched, compute_gate_env_batched,
    init_L_batched, init_R_batched,
    absorb_R_left_batched, absorb_R_right_batched,
    merge_gate_with_mpo_pair_batched,
    split_merged_tensor_batched,
    canonicalize_tensor_batched,
)


# ---------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------

def _partition(gates, structure):
    """Partition flat gate list into per-layer mutable lists."""
    result, idx = [], 0
    for _, pairs in structure:
        result.append(list(gates[idx:idx + len(pairs)]))
        idx += len(pairs)
    return result


def _flatten_into(gate_layers, gates):
    """Write per-layer gates back to flat list in place."""
    idx = 0
    for layer in gate_layers:
        for i, g in enumerate(layer):
            gates[idx + i] = g
        idx += len(layer)


# ---------------------------------------------------------------------
# Batched polar sweep (primary compilation path)
# ---------------------------------------------------------------------
#
# All tensors carry a leading batch dim B. B can be 1 for single-member
# compiles; there is no separate serial path — one code path, one test
# matrix. Layer-merge operations build MPO envelopes that diverge per
# batch element as gates evolve, so everything downstream of the first
# gate update must be batched.

def _merge_layer_right_to_left_batched(mpo, gates, odd, gate_is_left, max_bond):
    n = len(mpo)
    if not odd:
        Q, R = canonicalize_tensor_batched(mpo[-1], left=False)
        result = [Q]
        i = n - 2
    else:
        result = []
        i = n - 1
    while i - 1 >= 0:
        m1, m2 = mpo[i - 1], mpo[i]
        m2 = m2 if i == n - 1 else absorb_R_right_batched(m2, R)
        merged = merge_gate_with_mpo_pair_batched(
            gates[(i - 1) // 2], m1, m2, gate_is_left)
        T1, T2 = split_merged_tensor_batched(merged, 'right', max_bond)
        if i - 1 == 0:
            result += [T2, T1]
        else:
            Q, R = canonicalize_tensor_batched(T1, left=False)
            result += [T2, Q]
        i -= 2
    if i == 0:
        result.append(absorb_R_right_batched(mpo[0], R))
    result.reverse()
    return result


def _merge_layer_left_to_right_batched(mpo, gates, odd, gate_is_left, max_bond):
    n = len(mpo)
    if not odd:
        Q, R = canonicalize_tensor_batched(mpo[0], left=True)
        result = [Q]
        i = 1
    else:
        result = []
        i = 0
    while i + 1 < n:
        m1 = mpo[i] if i == 0 else absorb_R_left_batched(R, mpo[i])
        merged = merge_gate_with_mpo_pair_batched(
            gates[i // 2], m1, mpo[i + 1], gate_is_left)
        T1, T2 = split_merged_tensor_batched(merged, 'left', max_bond)
        if i + 1 == n - 1:
            result += [T1, T2]
        else:
            Q, R = canonicalize_tensor_batched(T2, left=True)
            result += [T1, Q]
        i += 2
    if i == n - 1:
        result.append(absorb_R_left_batched(R, mpo[-1]))
    return result


def _layer_envs_onepass_batched(gates, odd, upper, lower):
    """One L→R pass collecting gate environments for the cost read-out.

    Returns a list of (B, 2, 2, 2, 2) tensors, one per gate in the layer.
    Each is the conjugated environment `∂(Tr V†U)/∂gates[g]^*`.
    """
    n_gates = len(gates)
    B = gates[0].shape[0]
    eye = jnp.broadcast_to(_EYE1, (B, 1, 1))
    if odd:
        i_mpo = len(upper)
        R, L = eye, eye
    else:
        i_mpo = len(upper) - 1
        R = init_R_batched(upper[-1], lower[-1])
        L = init_L_batched(upper[0], lower[0])

    R_envs = [R]
    for gate in reversed(gates[1:]):
        i_mpo -= 2
        R = contract_R_batched(upper[i_mpo], upper[i_mpo + 1], gate,
                                lower[i_mpo], lower[i_mpo + 1], R)
        R_envs.append(R)
    R_envs.reverse()

    envs = []
    i_mpo = 0 if odd else 1
    i_env = i_mpo
    for g in range(n_gates):
        A1, A2 = upper[i_mpo], upper[i_mpo + 1]
        B1, B2 = lower[i_mpo], lower[i_mpo + 1]
        i_mpo += 2
        if g > 0:
            R = R_envs[g]
            L = contract_L_batched(L, upper[i_env], upper[i_env + 1],
                                    gates[g - 1],
                                    lower[i_env], lower[i_env + 1])
            i_env += 2
        envs.append(compute_gate_env_batched(L, A1, A2, B1, B2, R))
    return envs


def _compute_cost_batched(target_arrays_batched, gates, n_qubits, n_layers,
                           max_bond, first_odd):
    """Per-batch Frobenius cost. Returns a (B,) real array."""
    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    gate_layers = _partition(gates, structure)
    B = gates[0].shape[0]

    top = list(target_arrays_batched)
    sweep_right = False
    for gl, (odd, _) in zip(reversed(gate_layers[1:]),
                             reversed(structure[1:])):
        merge = (_merge_layer_right_to_left_batched if sweep_right
                 else _merge_layer_left_to_right_batched)
        top = merge(top, gl, odd, True, max_bond)
        sweep_right = not sweep_right

    bottom = identity_mpo_batched(n_qubits, B)
    odd = structure[0][0]
    envs = _layer_envs_onepass_batched(gate_layers[0], odd, top, bottom)
    overlap = jnp.einsum('Nabcd,Nabcd->N',
                         envs[0].conj(), gate_layers[0][0])
    return 2.0 - 2.0 * overlap.real / (2.0 ** n_qubits)


def _optimize_layer_inplace_batched(gates, odd, upper, lower, n_inner=3,
                                      drop_rate=0.0, rng=None):
    """Gauss-Seidel polar update over all gates in a single brickwall
    layer, vectorized over the ensemble batch dim.

    Shapes:
        gates: list of (B, 2, 2, 2, 2) arrays; mutated in place.
        upper, lower: lists of (B, bond_l, k, b, bond_r) MPO tensors.

    Within each inner pass, the L envelope advances as we update
    left-to-right (and R advances R→L). The "previous update" each gate
    sees is from neighbors on one side post-update, neighbors on the
    other side from pre-pass values — the classic Gauss-Seidel pattern.

    Dropout (`drop_rate` > 0): at each per-gate update opportunity, each
    of the B batch elements independently flips a Bernoulli(drop_rate)
    coin. The "skipped" elements keep their current gate value; others
    take the polar update. `jnp.where` fuses both cases into one XLA op
    so the batch dim stays tight.
    """
    n_gates = len(gates)
    i_start = 0 if odd else 1
    B = gates[0].shape[0]
    drop = drop_rate > 0.0

    def _init_LR():
        if odd:
            eye = jnp.broadcast_to(_EYE1, (B, 1, 1))
            return eye, eye
        return (init_L_batched(upper[0], lower[0]),
                init_R_batched(upper[-1], lower[-1]))

    def _apply_drop(old, new):
        if not drop:
            return new
        # Independent coins per batch element.
        mask = jnp.asarray(rng.random(B) >= drop_rate).reshape(B, 1, 1, 1, 1)
        return jnp.where(mask, new, old)

    for _ in range(n_inner):
        # --- Left-to-right pass ---
        L_init, R_init = _init_LR()
        R_envs = [R_init]
        i_mpo = len(upper) - (1 if not odd else 0)
        for g in reversed(range(1, n_gates)):
            i_mpo -= 2
            R_envs.append(contract_R_batched(
                upper[i_mpo], upper[i_mpo + 1], gates[g],
                lower[i_mpo], lower[i_mpo + 1], R_envs[-1]))
        R_envs.reverse()

        L = L_init
        for g in range(n_gates):
            i_mpo = i_start + 2 * g
            new_gate = env_and_polar_update_batched(
                L, upper[i_mpo], upper[i_mpo + 1],
                lower[i_mpo], lower[i_mpo + 1], R_envs[g])
            gates[g] = _apply_drop(gates[g], new_gate)
            if g < n_gates - 1:
                L = contract_L_batched(
                    L, upper[i_mpo], upper[i_mpo + 1], gates[g],
                    lower[i_mpo], lower[i_mpo + 1])

        # --- Right-to-left pass (mirror image) ---
        L_init, R_init = _init_LR()
        L_envs = [L_init]
        i_mpo = i_start
        for g in range(n_gates - 1):
            L_envs.append(contract_L_batched(
                L_envs[-1], upper[i_mpo], upper[i_mpo + 1], gates[g],
                lower[i_mpo], lower[i_mpo + 1]))
            i_mpo += 2

        R = R_init
        for g in reversed(range(n_gates)):
            i_mpo = i_start + 2 * g
            new_gate = env_and_polar_update_batched(
                L_envs[g], upper[i_mpo], upper[i_mpo + 1],
                lower[i_mpo], lower[i_mpo + 1], R)
            gates[g] = _apply_drop(gates[g], new_gate)
            if g > 0:
                R = contract_R_batched(
                    upper[i_mpo], upper[i_mpo + 1], gates[g],
                    lower[i_mpo], lower[i_mpo + 1], R)


def polar_sweep_batched(target_arrays_batched, gates, n_qubits, n_layers,
                         max_bond=128, first_odd=True, n_inner=3,
                         drop_rate=0.0, rng=None):
    """One full alternating-direction polar sweep (Gibbs & Cincio 2025),
    vectorized over an ensemble batch dim B.

    Args:
        target_arrays_batched: list of (B, bond_l, k, b, bond_r) JAX
            arrays — target MPO broadcast to B. All members share the
            target but carrying the batch dim lets downstream envelopes
            diverge per member as gates evolve.
        gates: list of (B, 2, 2, 2, 2) JAX arrays. Mutated in place.
        drop_rate, rng: see `_optimize_layer_inplace_batched`.

    Returns:
        (B,) cost vector (`2 − 2·Re Tr(V†U)/2ⁿ` per batch element).
    """
    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    L = len(structure)
    gate_layers = _partition(gates, structure)
    B = gates[0].shape[0]

    # --- Down sweep (L-1 → 0) ---
    lower_envs = [identity_mpo_batched(n_qubits, B)]
    sr = False
    for k in range(L - 1):
        merge = (_merge_layer_right_to_left_batched if sr
                 else _merge_layer_left_to_right_batched)
        lower_envs.append(merge(lower_envs[-1], gate_layers[k],
                                 structure[k][0], False, max_bond))
        sr = not sr

    top = list(target_arrays_batched)
    sr = False
    for layer in reversed(range(L)):
        odd = structure[layer][0]
        _optimize_layer_inplace_batched(
            gate_layers[layer], odd, top, lower_envs[layer], n_inner,
            drop_rate=drop_rate, rng=rng)
        if layer > 0:
            merge = (_merge_layer_right_to_left_batched if sr
                     else _merge_layer_left_to_right_batched)
            top = merge(top, gate_layers[layer], odd, True, max_bond)
            sr = not sr

    # --- Up sweep (0 → L-1) ---
    upper_envs = [list(target_arrays_batched)]
    sr = False
    for k in reversed(range(1, L)):
        merge = (_merge_layer_right_to_left_batched if sr
                 else _merge_layer_left_to_right_batched)
        upper_envs.append(merge(upper_envs[-1], gate_layers[k],
                                 structure[k][0], True, max_bond))
        sr = not sr
    upper_envs.reverse()

    bottom = identity_mpo_batched(n_qubits, B)
    sr = False
    for layer in range(L):
        odd = structure[layer][0]
        _optimize_layer_inplace_batched(
            gate_layers[layer], odd, upper_envs[layer], bottom, n_inner,
            drop_rate=drop_rate, rng=rng)
        if layer < L - 1:
            merge = (_merge_layer_right_to_left_batched if sr
                     else _merge_layer_left_to_right_batched)
            bottom = merge(bottom, gate_layers[layer], odd, False, max_bond)
            sr = not sr

    _flatten_into(gate_layers, gates)
    return _compute_cost_batched(target_arrays_batched, gates, n_qubits,
                                  n_layers, max_bond, first_odd)


# ---------------------------------------------------------------------
# Unbatched environment+gradient path (used by Riemannian ADAM only)
# ---------------------------------------------------------------------
#
# ADAM needs a scalar cost and an explicit gradient each step, so the
# polar path's "update-in-place-and-return-cost" shortcut doesn't apply.
# These helpers build the same environment tensors but return them to
# the caller instead of consuming them into a polar update.

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
            gates[(i - 1) // 2], m1, m2, gate_is_left)
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
            gates[i // 2], m1, mpo[i + 1], gate_is_left)
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


def _layer_envs_onepass(gates, odd, upper, lower):
    """Same as `_layer_envs_onepass_batched` but unbatched."""
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

    envs = []
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
        envs.append(compute_gate_env(L, A1, A2, B1, B2, R))
    return jnp.stack(envs)


def compute_cost_and_grad(target_arrays, gates, n_qubits, n_layers,
                           max_bond=128, first_odd=True):
    """Frobenius cost `2 − 2·Re Tr(V†U)/2ⁿ` and its per-gate gradient.

    Returns (cost: float, grad: numpy array of shape (n_gates, 2, 2, 2, 2)).
    Accepts numpy or JAX arrays. Used by Riemannian ADAM.
    """
    target_jax = [jnp.asarray(a) for a in target_arrays]
    gates_jax = [jnp.asarray(g) for g in gates]

    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    gate_layers = _partition(gates_jax, structure)

    top = list(target_jax)
    upper_envs = [list(top)]
    sweep_right = False
    for gl, (odd, _) in zip(reversed(gate_layers[1:]),
                             reversed(structure[1:])):
        merge = (_merge_layer_right_to_left if sweep_right
                 else _merge_layer_left_to_right)
        top = merge(top, gl, odd, True, max_bond)
        upper_envs.append(list(top))
        sweep_right = not sweep_right
    upper_envs.reverse()

    bottom = identity_mpo(n_qubits)
    sweep_right = False
    all_envs = []
    for layer, (odd, _) in enumerate(structure):
        if layer > 0:
            merge = (_merge_layer_right_to_left if sweep_right
                     else _merge_layer_left_to_right)
            bottom = merge(bottom, gate_layers[layer - 1],
                            structure[layer - 1][0], False, max_bond)
            sweep_right = not sweep_right
        all_envs.append(_layer_envs_onepass(gate_layers[layer], odd,
                                             upper_envs[layer], bottom))

    grad = np.asarray(jnp.concatenate(all_envs, axis=0))
    overlap = jnp.einsum('abcd,abcd->',
                         all_envs[0][0].conj(), gate_layers[0][0])
    cost = 2.0 - 2.0 * overlap.real / (2.0 ** n_qubits)
    return float(cost), grad
