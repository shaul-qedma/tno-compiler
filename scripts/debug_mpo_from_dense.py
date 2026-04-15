"""Test building brickwall MPO via from_dense (exact small case)."""

import numpy as np
import quimb.tensor as qtn
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def brickwall_to_qiskit(gates, n, n_layers, first_odd=True):
    qc = QuantumCircuit(n)
    odd = first_odd
    idx = 0
    for _ in range(n_layers):
        start = 0 if odd else 1
        for i in range(start, n - 1, 2):
            qc.append(UnitaryGate(gates[idx]), [i, i + 1])
            idx += 1
        odd = not odd
    return qc


n, n_layers = 6, 4
seed = 42

# Count gates
odd = True
ng = 0
for _ in range(n_layers):
    start = 0 if odd else 1
    ng += len(range(start, n - 1, 2))
    odd = not odd
print(f"n={n}, layers={n_layers}, gates={ng}")

gates = [random_unitary(4, seed=seed + i).data for i in range(ng)]

# Get exact unitary
qc = brickwall_to_qiskit(gates, n, n_layers)
U_exact = Operator(qc).data
print(f"Exact unitary shape: {U_exact.shape}")

# Build MPO from dense
mpo = qtn.MatrixProductOperator.from_dense(U_exact, dims=[2] * n)
U_roundtrip = np.array(mpo.to_dense())
print(f"MPO bonds: {mpo.bond_sizes()}")
print(f"Roundtrip match: {np.allclose(U_roundtrip, U_exact, atol=1e-10)}")
print(f"Trace match: {np.allclose(mpo.trace(), np.trace(U_exact), atol=1e-10)}")

# Compress and check
for max_bond in [64, 32, 16, 8, 4]:
    mpo_c = mpo.copy()
    mpo_c.compress(max_bond=max_bond)
    U_c = np.array(mpo_c.to_dense())
    overlap = abs(np.trace(U_c.conj().T @ U_exact)) / (2 ** n)
    print(f"  max_bond={max_bond}: bonds={mpo_c.bond_sizes()}, overlap={overlap:.8f}")
