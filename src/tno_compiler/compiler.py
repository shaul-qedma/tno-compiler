"""Compile a target `QuantumCircuit` into a shallower brickwall circuit.

The target is converted to an MPO (adjoint-reindexed into the
`rqcopt` convention) and then one of two optimizers runs to find the
brickwall gate set minimizing `‖U − V‖_F`:

- `method="polar"` (default) — Gibbs-Cincio polar-decomposition sweep.
  Uses the batched `polar_sweeps` with B=1 so the code path matches
  what `compile_ensemble` uses for B>1.
- `method="adam"` — Riemannian ADAM over U(4)^N.
"""

import numpy as np

from .brickwall import (
    brickwall_ansatz_gates, circuit_to_quimb_tn, gates_to_circuit,
)
from .compress import tn_to_mpo
from .gradient import compute_cost_and_grad
from .mpo_ops import mpo_to_arrays
from .optim import polar_sweeps, riemannian_adam


def build_target_arrays(target, max_bond=256, tol=1e-2):
    """Compress `target` to an MPO and produce the adjoint-reindexed
    arrays the compile's cost function expects.

    Returns (target_arrays, compress_error, actual_bond).
    `actual_bond` is the MPO's realized bond dimension after
    compression — always ≤ `max_bond` but often much smaller.
    """
    n_qubits = target.num_qubits
    tn = circuit_to_quimb_tn(target)
    mpo, compress_error = tn_to_mpo(
        tn, n_qubits, max_bond=max_bond,
        tol=tol * 0.1, norm="frobenius")
    actual_bond = max(mpo.bond_sizes())
    # Adjoint for rqcopt convention: the cost function expects V† contracted
    # against the circuit, so we swap bra/ket labels on the MPO.
    reindex_map = {f"k{i}": f"b{i}" for i in range(n_qubits)}
    reindex_map.update({f"b{i}": f"k{i}" for i in range(n_qubits)})
    target_arrays = mpo_to_arrays(mpo.conj().reindex(reindex_map))
    return target_arrays, compress_error, actual_bond


def compile_circuit(target, ansatz_depth, tol=1e-2,
                     max_bond=256, max_iter=500, lr=1e-3,
                     method="polar", first_odd=True,
                     init_gates=None, callback=None,
                     drop_rate=0.0, seed=0):
    """Compile `target` to a brickwall of depth `ansatz_depth`.

    Args:
        target: qiskit QuantumCircuit defining V.
        ansatz_depth: number of brickwall layers in the compiled circuit.
        tol: Frobenius error budget (10% allocated to MPO compression,
            90% remaining for the optimizer).
        max_bond: hard cap on MPO bond dimension during compression
            and envelope merging.
        max_iter: number of polar / ADAM iterations.
        lr: only used for `method="adam"`.
        method: "polar" (default) or "adam".
        first_odd: whether the brickwall ansatz starts with an odd-pair
            layer (gates at (0,1), (2,3), …).
        init_gates: optional list of (2,2,2,2) initial gate tensors.
            If None, an identity init is used.
        callback: optional callable(step, cost) for progress reporting.
        drop_rate: per-gate polar-sweep dropout probability (0 disables).
            Polar method only.
        seed: master RNG seed; drives dropout (init randomness is the
            caller's responsibility via `init_gates`).

    Returns:
        compiled: `QuantumCircuit` implementing the compiled brickwall.
        info: dict with keys `cost_history`, `compress_error`,
            `compile_error` (final cost), `gate_tensors` (list of
            (2,2,2,2) final gates).
    """
    n_qubits = target.num_qubits
    ansatz = brickwall_ansatz_gates(n_qubits, ansatz_depth, first_odd)

    target_arrays, compress_error, actual_bond = build_target_arrays(
        target, max_bond=max_bond, tol=tol)

    if init_gates is None:
        ng = sum(len(pairs) for _, pairs in ansatz)
        init_gates = [np.eye(4, dtype=complex).reshape(2, 2, 2, 2)] * ng

    if method == "polar":
        # B=1 wrapping: the batched path handles single-member compiles
        # uniformly, so there's one code path to maintain and test.
        opt_gates_list, hist_list = polar_sweeps(
            [init_gates], max_iter=max_iter, callback=callback,
            target_arrays=target_arrays, n_qubits=n_qubits,
            n_layers=ansatz_depth, max_bond=actual_bond,
            first_odd=first_odd,
            drop_rate=drop_rate, seed=seed)
        opt_gates = opt_gates_list[0]
        cost_history = hist_list[0]
    elif method == "adam":
        def cost_grad_fn(gates):
            return compute_cost_and_grad(
                target_arrays, gates, n_qubits, ansatz_depth,
                max_bond=actual_bond, first_odd=first_odd)

        opt_gates, cost_history = riemannian_adam(
            cost_grad_fn, init_gates, max_iter=max_iter, lr=lr,
            callback=callback)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'polar' or 'adam'.")

    compiled = gates_to_circuit(opt_gates, n_qubits, ansatz)
    return compiled, {
        'cost_history': cost_history,
        'compress_error': compress_error,
        'compile_error': cost_history[-1] if cost_history else float('inf'),
        'gate_tensors': opt_gates,
    }
