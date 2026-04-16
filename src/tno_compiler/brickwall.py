"""1D brickwall circuit construction using Qiskit QuantumCircuit."""

import numpy as np
import quimb.tensor as qtn
from qiskit.quantum_info import random_unitary
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def random_brickwall(n_qubits, n_layers, first_odd=True, seed=0):
    """Generate a Haar-random brickwall QuantumCircuit."""
    qc = QuantumCircuit(n_qubits)
    odd = first_odd
    gate_idx = 0
    for layer in range(n_layers):
        start = 0 if odd else 1
        for i in range(start, n_qubits - 1, 2):
            U = random_unitary(4, seed=seed + gate_idx).data
            qc.append(UnitaryGate(U, label=f"g{gate_idx}"), [i, i + 1])
            gate_idx += 1
        odd = not odd
    return qc


def circuit_to_quimb_tn(qc):
    """Convert a QuantumCircuit to a quimb unitary TN (split-gate)."""
    n = qc.num_qubits
    circ = qtn.Circuit(n)
    for layer_idx, instruction in enumerate(qc.data):
        gate = instruction.operation
        qubits = [qc.find_bit(q).index for q in instruction.qubits]
        mat = np.array(gate.to_matrix())
        if len(qubits) == 2:
            circ.apply_gate_raw(mat, tuple(qubits),
                                gate_round=layer_idx, contract="split-gate")
        elif len(qubits) == 1:
            circ.apply_gate_raw(mat, tuple(qubits),
                                gate_round=layer_idx, contract=True)
    return circ.get_uni()


def circuit_to_mpo(qc, max_bond=None, tol=1e-10, norm="operator"):
    """QuantumCircuit → quimb MPO with guaranteed tolerance."""
    from .compress import tn_to_mpo
    tn = circuit_to_quimb_tn(qc)
    return tn_to_mpo(tn, qc.num_qubits, max_bond=max_bond, tol=tol, norm=norm)


def brickwall_ansatz_gates(n_qubits, n_layers, first_odd=True):
    """Layer structure for a brickwall ansatz: list of (is_odd, [(q1,q2)...])."""
    odd = first_odd
    result = []
    for _ in range(n_layers):
        start = 0 if odd else 1
        result.append((odd, [(i, i + 1) for i in range(start, n_qubits - 1, 2)]))
        odd = not odd
    return result


def gates_to_circuit(gate_tensors, n_qubits, ansatz_structure):
    """Convert optimized (2,2,2,2) gate tensors back to a QuantumCircuit."""
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _, pairs in ansatz_structure:
        for q1, q2 in pairs:
            mat = np.asarray(gate_tensors[idx]).reshape(4, 4)
            qc.append(UnitaryGate(mat), [q1, q2])
            idx += 1
    return qc
