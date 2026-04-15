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
    """Get the exact 2^n x 2^n unitary matrix by direct tensor contraction.

    Uses big-endian qubit ordering (qubit 0 = most significant bit)
    to match the matrix_to_mpo decomposition convention.
    """
    d = 2 ** n_qubits
    U = np.eye(d, dtype=complex)
    structure = layer_structure(n_qubits, n_layers, first_odd)
    idx = 0
    for _, pairs in structure:
        layer_U = np.eye(d, dtype=complex)
        for q1, q2 in pairs:
            gate_mat = np.asarray(gates[idx]).reshape(4, 4)
            # Build the full-system operator: I ⊗ ... ⊗ G ⊗ ... ⊗ I
            # with G acting on qubits (q1, q2) in big-endian order
            op = _embed_gate(gate_mat, q1, q2, n_qubits)
            layer_U = op @ layer_U
            idx += 1
        U = layer_U @ U
    return U


def _embed_gate(gate_4x4, q1, q2, n_qubits):
    """Embed a 2-qubit gate into the full Hilbert space (big-endian)."""
    assert q2 == q1 + 1, "Only adjacent qubits supported"
    d = 2 ** n_qubits
    left = np.eye(2 ** q1) if q1 > 0 else np.array([[1.0]])
    right = np.eye(2 ** (n_qubits - q2 - 1)) if q2 < n_qubits - 1 else np.array([[1.0]])
    return np.kron(np.kron(left, gate_4x4), right)


def unitary_to_mpo(U, n_qubits, max_bond=None):
    """Convert a 2^n x 2^n unitary matrix to a quimb MPO."""
    mpo = qtn.MatrixProductOperator.from_dense(U, dims=[2] * n_qubits)
    if max_bond is not None:
        mpo.compress(max_bond=max_bond)
    return mpo


def target_mpo(gates, n_qubits, n_layers, first_odd=True):
    """Build the target MPO for compilation: stores V (the target itself).

    The rqcopt convention: the MPO holds V, and the circuit holds U†
    (the adjoint of the approximation). The overlap Tr(V · U†) is
    maximized, which equals conj(Tr(V† U)).

    Uses rqcopt's matrix_to_mpo decomposition for index compatibility
    with the merge/gradient einsums.
    """
    from .mpo_ops import matrix_to_mpo
    V = gates_to_unitary(gates, n_qubits, n_layers, first_odd)
    return matrix_to_mpo(V.conj().T)
