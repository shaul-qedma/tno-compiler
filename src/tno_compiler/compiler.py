"""MPO-based brickwall circuit compiler.

Given a target unitary (as a quimb MPO) and ansatz depth, finds
brickwall gates maximizing |Tr(V†U)|² via Riemannian ADAM on U(4).
"""

import numpy as np
from .brickwall import total_gates, partition_gates
from .mpo_ops import matrix_to_mpo
from .gradient import compute_cost_and_grad
from .optim import riemannian_adam


def compile_circuit(target_mpo, n_qubits, n_layers, max_bond=128,
                    max_iter=500, lr=1e-3, first_odd=True,
                    init_gates=None, callback=None):
    """Compile a target MPO into a brickwall circuit.

    Args:
        target_mpo: quimb MatrixProductOperator.
        n_qubits: number of qubits.
        n_layers: depth of brickwall ansatz.
        max_bond: max MPO bond dimension during contraction.
        max_iter: optimization iterations.
        lr: learning rate.
        init_gates: optional list of (2,2,2,2) arrays. Defaults to identity.
        callback: optional callable(step, cost).

    Returns:
        gates: list of (2,2,2,2) numpy arrays.
        cost_history: list of costs.
    """
    target_arrays = target_mpo  # already a list of arrays

    if init_gates is None:
        ng = total_gates(n_qubits, n_layers, first_odd)
        init_gates = [np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
                      for _ in range(ng)]

    def cost_grad_fn(gates_tn):
        gates_list = list(gates_tn)
        cost, grad = compute_cost_and_grad(
            target_arrays, gates_list, n_qubits, n_layers,
            max_bond, first_odd)
        return cost, grad

    gates, cost_history = riemannian_adam(
        cost_grad_fn, init_gates, max_iter=max_iter, lr=lr,
        callback=callback)

    return list(gates), cost_history
