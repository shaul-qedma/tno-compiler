"""Verify matrix_to_mpo roundtrip and merge with rqcopt einsums."""

import numpy as np
from qiskit.quantum_info import random_unitary
from tno_compiler.mpo_ops import matrix_to_mpo, trace_mpo, merge_gate_with_mpo_pair, split_merged_tensor

# Test 1: matrix_to_mpo roundtrip
for n in [2, 4, 6]:
    U = random_unitary(2**n, seed=42).data
    mpo = matrix_to_mpo(U)
    # Reconstruct
    A = mpo[0]
    for B in mpo[1:]:
        C = np.einsum('iabj,jcdk->iacbdk', A, B)
        s = C.shape
        A = C.reshape(s[0], s[1]*s[2], s[3]*s[4], s[-1])
    U_recon = np.einsum('iabj->ab', A)
    print(f"n={n}: roundtrip={np.allclose(U_recon, U, atol=1e-10)}, "
          f"trace={np.allclose(trace_mpo(mpo), np.trace(U), atol=1e-10)}")

# Test 2: merge + trace for 2 qubits (should give Tr(V†G))
print("\n--- Merge test ---")
V = random_unitary(4, seed=0).data
G = random_unitary(4, seed=100).data

Vd_mpo = matrix_to_mpo(V.conj().T)
G_tn = G.reshape(2, 2, 2, 2)

merged = merge_gate_with_mpo_pair(G_tn, Vd_mpo[0], Vd_mpo[1], gate_is_left=True)
T0, T1 = split_merged_tensor(merged, max_bond=128)
ov = trace_mpo([T0, T1])
exact = np.trace(V.conj().T @ G)
print(f"2-qubit: ov={ov}, exact={exact}, match={np.allclose(ov, exact, atol=1e-8)}")

# Test 3: merge for 4 qubits, 1 odd layer
print("\n--- 4-qubit test ---")
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo

tg = random_haar_gates(4, 1, seed=0)
cg = random_haar_gates(4, 1, seed=100)

T = target_mpo(tg, 4, 1)
print(f"Target shapes: {[t.shape for t in T]}")

# Manually merge both gates
m0 = merge_gate_with_mpo_pair(cg[0], T[0], T[1], gate_is_left=True)
T0, T1 = split_merged_tensor(m0, max_bond=128)

# Need to carry R from T1 to T[2]
from tno_compiler.mpo_ops import canonicalize_tensor
Q1, R1 = canonicalize_tensor(T1, left=True)
T2_R = np.einsum('ij,jabk->iabk', R1, T[2])

m1 = merge_gate_with_mpo_pair(cg[1], T2_R, T[3], gate_is_left=True)
T2, T3 = split_merged_tensor(m1, max_bond=128)

ov = trace_mpo([T0, Q1, T2, T3])
exact = np.trace(gates_to_unitary(tg, 4, 1).conj().T @ gates_to_unitary(cg, 4, 1))
print(f"4-qubit: ov={ov}, exact={exact}, match={np.allclose(ov, exact, atol=1e-6)}")
