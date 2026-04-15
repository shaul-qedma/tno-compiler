"""Debug: verify merge for multi-gate layers on 4-qubit factored target.

The target is a single odd layer on 4 qubits: V = V0⊗V1 with V0 on (0,1)
and V1 on (2,3). So Tr(V†U) = Tr(V0†U0) × Tr(V1†U1).

We verify that merging each gate pair independently and tracing gives
the correct factored overlap.
"""

import numpy as np
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo as make_target
from tno_compiler.mpo_ops import quimb_mpo_to_arrays, trace_mpo

n, d = 4, 1
tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=100)

V = gates_to_unitary(tg, n, d)
U = gates_to_unitary(cg, n, d)
exact = np.trace(V.conj().T @ U)
print(f"Exact Tr(V†U) = {exact}")

# Factor check
V0, V1 = tg[0].reshape(4, 4), tg[1].reshape(4, 4)
U0, U1 = cg[0].reshape(4, 4), cg[1].reshape(4, 4)
tr01 = np.trace(V0.conj().T @ U0)
tr23 = np.trace(V1.conj().T @ U1)
print(f"Tr(V0†U0) = {tr01}")
print(f"Tr(V1†U1) = {tr23}")
print(f"Product = {tr01 * tr23} (should match exact)")

# Target MPO stores V†
T = quimb_mpo_to_arrays(make_target(tg, n, d))
print(f"\nTarget MPO shapes: {[t.shape for t in T]}")

# Verify target MPO represents V†
# Reconstruct V† from tensors and compare
from tno_compiler.brickwall import unitary_to_mpo
import quimb.tensor as qtn
Vd = V.conj().T
mpo_check = qtn.MatrixProductOperator.from_dense(Vd, dims=[2]*n)
Vd_recon = np.array(mpo_check.to_dense())
print(f"V† stored correctly: {np.allclose(Vd_recon, Vd, atol=1e-10)}")

# Now manually compute Tr(V†·U) using tensors
# Tr(V†U) = sum_{all indices} V†[rows, cols] U[cols, rows]
# In MPO tensor form for the whole 4-qubit system:
# V†[(up0,up1,up2,up3),(dn0,dn1,dn2,dn3)] = T0*T1*T2*T3
# U[(k0,k1,k2,k3),(b0,b1,b2,b3)] = U0[k0,k1,b0,b1] * U1[k2,k3,b2,b3]
# Tr = sum V†[r,c] U[c,r] = sum T_tensors * U[dn,up]
# where dn = lower indices of V† = ket of V, up = upper of V† = bra of V

# The factored form: since bond between sites 1-2 is 1,
# trace = product of traces over each side.
# For the (0,1) block:
# sum_{up0,dn0,up1,dn1} T0[1,up0,dn0,d] T1[d,up1,dn1,1] U0[dn0,dn1,up0,up1]
# This is: np.einsum('iabd,dcek,beac->', T0, T1, G0_reshaped)

G0 = cg[0]  # (k0, k1, b0, b1) = (2,2,2,2)
G1 = cg[1]

# For Tr(V†U) at sites (0,1):
# V†[up,dn] * U[dn,up] = T0[up0,dn0,d] * T1[d,up1,dn1] * U0[(dn0,dn1),(up0,up1)]
# U0 in tensor form: U0[dn0,dn1,up0,up1] = G0 reindexed
# If G0 is (k0,k1,b0,b1) and we need U0[col_0, col_1, row_0, row_1]:
# col = ket index of U, row = bra index of U
# So U0[dn0,dn1,up0,up1] with dn = col of U = ket of U and up = row of U = bra of U
# Meaning G0[dn0,dn1,up0,up1] -- but G0's convention is (k0,k1,b0,b1)
# where k=ket=row, b=bra=col. So G0[row0,row1,col0,col1] = G0[k0,k1,b0,b1]
# and we need G0[col0,col1,row0,row1] = G0 transposed to (b0,b1,k0,k1)

tr_manual_01 = np.einsum('iabd,dcek,beac->ik', T[0], T[1],
                          G0.transpose(2, 3, 0, 1))
print(f"\nManual Tr_01 = {tr_manual_01[0, 0]}")
print(f"Expected = {tr01}")
print(f"Match: {np.allclose(tr_manual_01[0, 0], tr01, atol=1e-10)}")

tr_manual_23 = np.einsum('iabd,dcek,beac->ik', T[2], T[3],
                          G1.transpose(2, 3, 0, 1))
print(f"\nManual Tr_23 = {tr_manual_23[0, 0]}")
print(f"Expected = {tr23}")
print(f"Match: {np.allclose(tr_manual_23[0, 0], tr23, atol=1e-10)}")
