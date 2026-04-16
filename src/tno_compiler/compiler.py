"""MPO-based brickwall circuit compiler."""

import numpy as np
from .brickwall import total_gates
from .gradient import compute_cost_and_grad
from .optim import riemannian_adam


def compile_circuit(target_mpo, n_qubits, n_layers, max_bond=128,
                    max_iter=500, lr=1e-3, first_odd=True,
                    init_gates=None, callback=None):
    """Compile a target MPO (from brickwall.target_mpo) into brickwall gates.
    Returns (gates, cost_history)."""
    if init_gates is None:
        ng = total_gates(n_qubits, n_layers, first_odd)
        init_gates = [np.eye(4, dtype=complex).reshape(2, 2, 2, 2)] * ng

    def cost_grad_fn(gates):
        return compute_cost_and_grad(
            target_mpo, gates, n_qubits, n_layers, max_bond, first_odd)

    return riemannian_adam(cost_grad_fn, init_gates,
                          max_iter=max_iter, lr=lr, callback=callback)
