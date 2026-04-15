"""Debug quimb gate_upper_ on MPO for brickwall circuits."""

import quimb.tensor as qtn
import quimb as qu
import numpy as np
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def test_case(n, gate_sequence, label):
    """Apply gate_sequence to n-qubit identity MPO and compare with Qiskit."""
    # Qiskit reference
    qc = QuantumCircuit(n)
    for gate, qubits in gate_sequence:
        qc.append(UnitaryGate(gate), list(qubits))
    U_exact = Operator(qc).data

    # quimb
    mpo = qtn.MPO_identity(n, dtype=complex)
    for gate, qubits in gate_sequence:
        mpo.gate_upper_(gate, qubits, contract='reduce-split')
    U_mpo = np.array(mpo.to_dense())

    overlap = abs(np.trace(U_mpo.conj().T @ U_exact)) / (2 ** n)
    match = np.allclose(U_mpo, U_exact, atol=1e-8)
    print(f"{label}: overlap={overlap:.8f}, match={match}, bonds={mpo.bond_sizes()}")

    # Also try from_dense for comparison
    mpo_ref = qtn.MatrixProductOperator.from_dense(U_exact, dims=[2] * n)
    U_ref = np.array(mpo_ref.to_dense())
    print(f"  from_dense roundtrip: {np.allclose(U_ref, U_exact, atol=1e-10)}")


g = [random_unitary(4, seed=i).data for i in range(10)]

# Case 1: 2 qubits, 1 gate (works)
test_case(2, [(g[0], (0, 1))], "2q, 1 gate")

# Case 2: 4 qubits, adjacent gates on same layer
test_case(4, [(g[0], (0, 1)), (g[1], (2, 3))], "4q, 1 odd layer")

# Case 3: 4 qubits, 2 layers
test_case(4, [(g[0], (0, 1)), (g[1], (2, 3)), (g[2], (1, 2))], "4q, 2 layers")

# Case 4: 6 qubits, 1 odd layer
test_case(6, [(g[0], (0, 1)), (g[1], (2, 3)), (g[2], (4, 5))], "6q, 1 odd layer")
