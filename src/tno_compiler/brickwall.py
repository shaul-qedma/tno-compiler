"""1D brickwall circuit: alternating layers of nearest-neighbor 2-qubit gates.

Odd layers:  (0,1), (2,3), (4,5), ...
Even layers: (1,2), (3,4), (5,6), ...

Gates are (2,2,2,2) tensors. Matrices use big-endian qubit ordering.
"""

import numpy as np
from qiskit.quantum_info import random_unitary


def layer_structure(n_qubits, n_layers, first_odd=True):
    """List of (is_odd, [(q1,q2), ...]) for each layer."""
    odd = first_odd
    result = []
    for _ in range(n_layers):
        start = 0 if odd else 1
        result.append((odd, [(i, i + 1) for i in range(start, n_qubits - 1, 2)]))
        odd = not odd
    return result


def total_gates(n_qubits, n_layers, first_odd=True):
    return sum(len(p) for _, p in layer_structure(n_qubits, n_layers, first_odd))


def partition_gates(gates, n_qubits, n_layers, first_odd=True):
    """Split a flat gate list into per-layer lists."""
    result, idx = [], 0
    for _, pairs in layer_structure(n_qubits, n_layers, first_odd):
        result.append(gates[idx:idx + len(pairs)])
        idx += len(pairs)
    return result


def random_haar_gates(n_qubits, n_layers, first_odd=True, seed=0):
    """Haar-random 2-qubit gates. Returns list of (2,2,2,2) arrays."""
    ng = total_gates(n_qubits, n_layers, first_odd)
    return [random_unitary(4, seed=seed + i).data.reshape(2, 2, 2, 2)
            for i in range(ng)]


def gates_to_unitary(gates, n_qubits, n_layers, first_odd=True):
    """Exact 2^n × 2^n unitary from gates (big-endian, for testing)."""
    d = 2 ** n_qubits
    U = np.eye(d, dtype=complex)
    idx = 0
    for _, pairs in layer_structure(n_qubits, n_layers, first_odd):
        layer_U = np.eye(d, dtype=complex)
        for q1, q2 in pairs:
            mat = np.asarray(gates[idx]).reshape(4, 4)
            left = np.eye(2 ** q1) if q1 > 0 else np.ones((1, 1))
            right = np.eye(2 ** (n_qubits - q2 - 1)) if q2 < n_qubits - 1 else np.ones((1, 1))
            layer_U = np.kron(np.kron(left, mat), right) @ layer_U
            idx += 1
        U = layer_U @ U
    return U


def target_mpo(gates, n_qubits, n_layers, first_odd=True):
    """Target MPO for compilation (stores V†, matching the rqcopt merge convention)."""
    from .mpo_ops import matrix_to_mpo
    return matrix_to_mpo(gates_to_unitary(gates, n_qubits, n_layers, first_odd).conj().T)
