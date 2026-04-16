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
from .compress import tn_to_mpo, _contract_to_exact_mpo
from .mpo_ops import mpo_to_arrays
from .gradient import compute_cost_and_grad
from .optim import riemannian_adam, polar_sweeps


def compile_circuit(target, ansatz_depth, compress_fraction=0.0,
                    tol=1e-2, max_bond=None, max_iter=500, lr=1e-3,
                    method="polar", first_odd=True,
                    init_gates=None, callback=None):
    """Compile a target QuantumCircuit into a brickwall circuit at ansatz_depth.

    Args:
        target: qiskit QuantumCircuit defining the target unitary V.
        ansatz_depth: number of brickwall layers in the compiled circuit.
        compress_fraction: fraction of tol for MPO compression (0 = exact).
        tol: total Frobenius error budget.
        max_bond: hard cap on MPO bond dimension.
        max_iter, lr: optimizer parameters.
        first_odd: brickwall ansatz starts with odd layer.
        init_gates: optional list of (2,2,2,2) initial gate tensors.
        callback: optional callable(step, cost).

    Returns:
        compiled: QuantumCircuit (the compiled brickwall circuit).
        info: dict with cost_history, compress_error, compile_error.
    """
    n_qubits = target.num_qubits
    ansatz = brickwall_ansatz_gates(n_qubits, ansatz_depth, first_odd)

    # Build target MPO
    tn = circuit_to_quimb_tn(target)
    if compress_fraction > 0:
        mpo, compress_error = tn_to_mpo(
            tn, n_qubits, max_bond=max_bond,
            tol=tol * compress_fraction, norm="frobenius")
    else:
        mpo = _contract_to_exact_mpo(tn, n_qubits)
        compress_error = 0.0

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
            cost_grad_fn, init_gates, max_iter=max_iter, callback=callback)
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
