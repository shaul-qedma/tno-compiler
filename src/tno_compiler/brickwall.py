"""1D brickwall circuit representation and conversion utilities.

A brickwall on n qubits at depth D has alternating layers:
  - Odd layers:  gates on (0,1), (2,3), (4,5), ...
  - Even layers: gates on (1,2), (3,4), (5,6), ...

Each gate is a (2,2,2,2) tensor reshaped from a 4x4 unitary.
All matrices use big-endian qubit ordering (site 0 = MSB).
"""

import numpy as np
from qiskit.quantum_info import random_unitary


def layer_pairs(n_qubits, odd):
    """Qubit pairs for an odd (True) or even (False) brickwall layer."""
    start = 0 if odd else 1
    return [(i, i + 1) for i in range(start, n_qubits - 1, 2)]


def layer_structure(n_qubits, n_layers, first_odd=True):
    """Return list of (is_odd, pairs) for each layer."""
    odd = first_odd
    result = []
    for _ in range(n_layers):
        result.append((odd, layer_pairs(n_qubits, odd)))
        odd = not odd
    return result


def total_gates(n_qubits, n_layers, first_odd=True):
    """Total 2-qubit gates in a brickwall circuit."""
    return sum(len(pairs) for _, pairs in
               layer_structure(n_qubits, n_layers, first_odd))


def partition_gates(gates, n_qubits, n_layers, first_odd=True):
    """Split flat gate list into per-layer lists."""
    result = []
    idx = 0
    for _, pairs in layer_structure(n_qubits, n_layers, first_odd):
        n = len(pairs)
        result.append(gates[idx:idx + n])
        idx += n
    return result


def random_haar_gates(n_qubits, n_layers, first_odd=True, seed=0):
    """Generate Haar-random 2-qubit gates for a brickwall circuit.

    Returns list of (2,2,2,2) numpy arrays.
    """
    ng = total_gates(n_qubits, n_layers, first_odd)
    return [random_unitary(4, seed=seed + i).data.reshape(2, 2, 2, 2)
            for i in range(ng)]


def gates_to_unitary(gates, n_qubits, n_layers, first_odd=True):
    """Build the exact 2^n x 2^n unitary from brickwall gates.

    Uses big-endian ordering (site 0 = MSB) to match matrix_to_mpo.
    """
    d = 2 ** n_qubits
    U = np.eye(d, dtype=complex)
    idx = 0
    for _, pairs in layer_structure(n_qubits, n_layers, first_odd):
        layer_U = np.eye(d, dtype=complex)
        for q1, q2 in pairs:
            gate_mat = np.asarray(gates[idx]).reshape(4, 4)
            left = np.eye(2 ** q1) if q1 > 0 else np.ones((1, 1))
            right = np.eye(2 ** (n_qubits - q2 - 1)) if q2 < n_qubits - 1 else np.ones((1, 1))
            layer_U = np.kron(np.kron(left, gate_mat), right) @ layer_U
            idx += 1
        U = layer_U @ U
    return U


def target_mpo(gates, n_qubits, n_layers, first_odd=True):
    """Build the target MPO for compilation (stores V†).

    The rqcopt merge convention computes Tr(MPO · circuit). Storing V†
    makes the overlap Tr(V† · U), so maximizing Re(overlap) minimizes
    the Frobenius distance ‖V - U‖_F.
    """
    from .mpo_ops import matrix_to_mpo
    V = gates_to_unitary(gates, n_qubits, n_layers, first_odd)
    return matrix_to_mpo(V.conj().T)
