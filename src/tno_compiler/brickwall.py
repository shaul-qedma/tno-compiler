"""1D brickwall circuit representation and conversion utilities.

A brickwall on n qubits at depth D has alternating layers:
  - Odd layers:  gates on (0,1), (2,3), (4,5), ...
  - Even layers: gates on (1,2), (3,4), (5,6), ...

Each gate is a 4x4 unitary matrix (numpy array).
"""

import numpy as np
from qiskit.quantum_info import random_unitary, Operator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import quimb.tensor as qtn


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
    """Generate Haar-random 2-qubit gates for a brickwall circuit."""
    ng = total_gates(n_qubits, n_layers, first_odd)
    return [random_unitary(4, seed=seed + i).data for i in range(ng)]


def gates_to_qiskit(gates, n_qubits, n_layers, first_odd=True):
    """Convert gate list to a Qiskit QuantumCircuit."""
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _, pairs in layer_structure(n_qubits, n_layers, first_odd):
        for q1, q2 in pairs:
            qc.append(UnitaryGate(gates[idx]), [q1, q2])
            idx += 1
    return qc


def gates_to_unitary(gates, n_qubits, n_layers, first_odd=True):
    """Get the exact 2^n x 2^n unitary matrix via Qiskit."""
    qc = gates_to_qiskit(gates, n_qubits, n_layers, first_odd)
    return Operator(qc).data


def unitary_to_mpo(U, n_qubits, max_bond=None):
    """Convert a 2^n x 2^n unitary matrix to a quimb MPO."""
    mpo = qtn.MatrixProductOperator.from_dense(U, dims=[2] * n_qubits)
    if max_bond is not None:
        mpo.compress(max_bond=max_bond)
    return mpo


def gates_to_mpo(gates, n_qubits, n_layers, max_bond=None, first_odd=True):
    """Convert brickwall gates to MPO via exact unitary (small n only)."""
    U = gates_to_unitary(gates, n_qubits, n_layers, first_odd)
    return unitary_to_mpo(U, n_qubits, max_bond)
