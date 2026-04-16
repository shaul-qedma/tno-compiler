"""MPO-based brickwall circuit compiler with error budget allocation."""

import numpy as np
from .brickwall import total_gates, circuit_to_tn, layer_structure, partition_gates
from .compress import tn_to_mpo, _contract_to_exact_mpo
from .mpo_ops import mpo_to_arrays
from .gradient import compute_cost_and_grad
from .optim import riemannian_adam


def compile_circuit(target_gates, n_qubits, n_layers, tol=1e-2,
                    compress_fraction=0.0, max_bond=None,
                    max_iter=500, lr=1e-3, first_odd=True,
                    init_gates=None, callback=None):
    """Compile a target brickwall circuit into an approximate brickwall circuit.

    Args:
        target_gates: list of (2,2,2,2) gate tensors defining the target V.
        n_qubits, n_layers: circuit dimensions.
        tol: total Frobenius error budget for the compilation.
        compress_fraction: fraction of tol allocated to MPO compression.
            0.0 (default) = no compression, use exact TN contracted to MPO.
            E.g. 0.1 = 10% of budget to compression, 90% to optimization.
        max_bond: hard cap on MPO bond dimension (None = no cap).
        max_iter, lr: optimizer parameters.
        init_gates: optional initial gates.
        callback: optional callable(step, cost).

    Returns:
        gates: compiled gate list.
        info: dict with 'cost_history', 'compress_error', 'compile_error'.
    """
    # Build target MPO (V†) with error budget split
    compress_tol = tol * compress_fraction
    tn = circuit_to_tn(target_gates, n_qubits, n_layers, first_odd)

    if compress_fraction > 0:
        mpo, compress_error = tn_to_mpo(
            tn, n_qubits, max_bond=max_bond, tol=compress_tol, norm="frobenius")
    else:
        mpo = _contract_to_exact_mpo(tn, n_qubits)
        compress_error = 0.0

    # Take adjoint for the rqcopt convention
    reindex_map = {f"k{i}": f"b{i}" for i in range(n_qubits)}
    reindex_map.update({f"b{i}": f"k{i}" for i in range(n_qubits)})
    target_mpo = mpo.conj().reindex(reindex_map)
    target_arrays = mpo_to_arrays(target_mpo)

    # Initialize gates
    if init_gates is None:
        ng = total_gates(n_qubits, n_layers, first_odd)
        init_gates = [np.eye(4, dtype=complex).reshape(2, 2, 2, 2)] * ng

    # Optimize
    def cost_grad_fn(gates):
        return compute_cost_and_grad(
            target_arrays, gates, n_qubits, n_layers,
            max_bond=max_bond or 128, first_odd=first_odd)

    gates, cost_history = riemannian_adam(
        cost_grad_fn, init_gates, max_iter=max_iter, lr=lr, callback=callback)

    return gates, {
        'cost_history': cost_history,
        'compress_error': compress_error,
        'compile_error': cost_history[-1] if cost_history else float('inf'),
    }
