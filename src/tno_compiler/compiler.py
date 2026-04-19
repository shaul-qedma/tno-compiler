"""MPO-based circuit compiler: compile a target QuantumCircuit into a
shallower brickwall circuit.

Two optimization methods:
- "polar": polar decomposition sweeps (Gibbs & Cincio 2025).
  Analytic optimal update per gate, no learning rate.
- "adam": Riemannian ADAM (rqcopt, INMLe/rqcopt-mpo). Gradient-based,
  requires learning rate tuning.
"""

import numpy as np
from .brickwall import (
    circuit_to_quimb_tn, brickwall_ansatz_gates, gates_to_circuit,
)
from .compress import tn_to_mpo
from .mpo_ops import mpo_to_arrays
from .gradient import compute_cost_and_grad
from .optim import riemannian_adam, polar_sweeps


def compile_circuit(target, ansatz_depth, tol=1e-2,
                    max_bond=256, max_iter=500, lr=1e-3,
                    method="polar", first_odd=True,
                    init_gates=None, callback=None):
    """Compile a target QuantumCircuit into a brickwall circuit at ansatz_depth.

    Args:
        target: qiskit QuantumCircuit defining the target unitary V.
        ansatz_depth: number of brickwall layers in the compiled circuit.
        tol: Frobenius error budget (split between compression and compilation).
        max_bond: hard cap on MPO bond dimension (default 256, computational ceiling).
        max_iter, lr: optimizer parameters (lr only used for method="adam").
        method: "polar" (default, Gibbs-Cincio) or "adam" (Riemannian ADAM).
        first_odd: brickwall ansatz starts with odd layer.
        init_gates: optional list of (2,2,2,2) initial gate tensors.
        callback: optional callable(step, cost).

    Returns:
        compiled: QuantumCircuit (the compiled brickwall circuit).
        info: dict with cost_history, compress_error, compile_error.
    """
    n_qubits = target.num_qubits
    ansatz = brickwall_ansatz_gates(n_qubits, ansatz_depth, first_odd)

    # Build target MPO (allocate 10% of tol to compression)
    tn = circuit_to_quimb_tn(target)
    mpo, compress_error = tn_to_mpo(
        tn, n_qubits, max_bond=max_bond,
        tol=tol * 0.1, norm="frobenius")

    # Adjoint for rqcopt convention
    reindex_map = {f"k{i}": f"b{i}" for i in range(n_qubits)}
    reindex_map.update({f"b{i}": f"k{i}" for i in range(n_qubits)})
    target_arrays = mpo_to_arrays(mpo.conj().reindex(reindex_map))

    # Initialize ansatz gates
    if init_gates is None:
        ng = sum(len(pairs) for _, pairs in ansatz)
        init_gates = [np.eye(4, dtype=complex).reshape(2, 2, 2, 2)] * ng

    # Partition and optimize
    def _partition(gates):
        result, idx = [], 0
        for _, pairs in ansatz:
            result.append(gates[idx:idx + len(pairs)])
            idx += len(pairs)
        return result

    is_odd = [odd for odd, _ in ansatz]

    def cost_grad_fn(gates):
        return compute_cost_and_grad(
            target_arrays, gates, n_qubits, ansatz_depth,
            max_bond=max_bond or 128, first_odd=first_odd)

    if method == "polar":
        opt_gates, cost_history = polar_sweeps(
            cost_grad_fn, init_gates, max_iter=max_iter, callback=callback,
            target_arrays=target_arrays, n_qubits=n_qubits,
            n_layers=ansatz_depth, max_bond=max_bond or 128,
            first_odd=first_odd)
    elif method == "adam":
        opt_gates, cost_history = riemannian_adam(
            cost_grad_fn, init_gates, max_iter=max_iter, lr=lr, callback=callback)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'polar' or 'adam'.")

    compiled = gates_to_circuit(opt_gates, n_qubits, ansatz)
    return compiled, {
        'cost_history': cost_history,
        'compress_error': compress_error,
        'compile_error': cost_history[-1] if cost_history else float('inf'),
        'gate_tensors': opt_gates,
    }
