"""1D brickwall circuit topology and quimb TN construction.

Odd layers:  (0,1), (2,3), (4,5), ...
Even layers: (1,2), (3,4), (5,6), ...
"""

import numpy as np
import quimb.tensor as qtn
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


def circuit_to_tn(gates, n_qubits, n_layers, first_odd=True):
    """Build a quimb TN from brickwall gates (split-gate: one tensor per site per gate)."""
    circ = qtn.Circuit(n_qubits)
    idx = 0
    for layer, (_, pairs) in enumerate(layer_structure(n_qubits, n_layers, first_odd)):
        for q1, q2 in pairs:
            circ.apply_gate_raw(
                np.asarray(gates[idx]).reshape(4, 4),
                (q1, q2),
                gate_round=layer,
                contract="split-gate",
            )
            idx += 1
    return circ.get_uni()


def circuit_to_mpo(gates, n_qubits, n_layers, first_odd=True,
                   max_bond=None, tol=1e-10):
    """Brickwall gates → quimb MPO with guaranteed operator norm tolerance."""
    from .compress import tn_to_mpo
    tn = circuit_to_tn(gates, n_qubits, n_layers, first_odd)
    return tn_to_mpo(tn, n_qubits, max_bond=max_bond, tol=tol)


def target_mpo(gates, n_qubits, n_layers, first_odd=True,
               max_bond=None, tol=1e-10):
    """Target MPO for compilation (stores V†). Returns (mpo, error_bound)."""
    mpo, error = circuit_to_mpo(gates, n_qubits, n_layers, first_odd,
                                max_bond, tol)
    reindex_map = {f"k{i}": f"b{i}" for i in range(n_qubits)}
    reindex_map.update({f"b{i}": f"k{i}" for i in range(n_qubits)})
    return mpo.conj().reindex(reindex_map), error
