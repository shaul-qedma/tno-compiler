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
    """Generate Haar-random 2-qubit gates for a brickwall circuit.

    Returns list of (2,2,2,2) numpy arrays (tensor form).
    """
    ng = total_gates(n_qubits, n_layers, first_odd)
    return [random_unitary(4, seed=seed + i).data.reshape(2, 2, 2, 2)
            for i in range(ng)]


def gates_to_qiskit(gates, n_qubits, n_layers, first_odd=True):
    """Convert gate list to a Qiskit QuantumCircuit.

    Gates can be (2,2,2,2) or (4,4) arrays.
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _, pairs in layer_structure(n_qubits, n_layers, first_odd):
        for q1, q2 in pairs:
            mat = np.asarray(gates[idx]).reshape(4, 4)
            qc.append(UnitaryGate(mat), [q1, q2])
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


def target_mpo(gates, n_qubits, n_layers, first_odd=True):
    """Build the target MPO for compilation: stores V† (adjoint).

    The rqcopt convention computes Tr(MPO · circuit), so the MPO must
    hold V† so that maximizing Re Tr(V† U) minimizes ‖V - U‖_F.

    Uses rqcopt's matrix_to_mpo decomposition (not quimb) for index
    compatibility with the merge/gradient einsums.
    """
    from .mpo_ops import matrix_to_mpo
    V = gates_to_unitary(gates, n_qubits, n_layers, first_odd)
    return matrix_to_mpo(V.conj().T)
