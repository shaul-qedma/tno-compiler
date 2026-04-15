"""Derive the correct merge einsum from first principles."""

import numpy as np
from qiskit.quantum_info import random_unitary
import quimb.tensor as qtn

# Setup: V and U are 2-qubit unitaries
V0_mat = random_unitary(4, seed=0).data  # acts on qubits (0,1)
U0_mat = random_unitary(4, seed=100).data  # circuit gate on (0,1)

# We want Tr(V0† U0) using MPO representation of V†.
exact = np.trace(V0_mat.conj().T @ U0_mat)
print(f"Exact = {exact}")

# V†_mat = V0_mat.conj().T
Vd = V0_mat.conj().T

# MPO for V†: from_dense stores it with upper=row, lower=col
mpo = qtn.MatrixProductOperator.from_dense(Vd, dims=[2, 2])
# T0: (upper0, lower0, bond) where upper=row of Vd, lower=col of Vd
# i.e., upper = bra-of-V, lower = ket-of-V
T0_raw = np.array(mpo[0].data)  # (up0, dn0, bond)
T1_raw = np.array(mpo[1].data)  # (bond, up1, dn1)

# Verify: Vd_{(up0 up1),(dn0 dn1)} = T0[up0,dn0,d] T1[d,up1,dn1]
Vd_recon = np.einsum('abd,dce->acbe', T0_raw, T1_raw).reshape(4, 4)
print(f"V† reconstruction: {np.allclose(Vd_recon, Vd, atol=1e-10)}")

# With dummy bonds: T0: (1, up0, dn0, bond), T1: (bond, up1, dn1, 1)
T0 = T0_raw[np.newaxis, ...]
T1 = T1_raw[..., np.newaxis]

# U0 as tensor: U_{(k0 k1),(b0 b1)} -> G[k0,k1,b0,b1]
G = U0_mat.reshape(2, 2, 2, 2)

# Tr(V†U) = sum_{ij} Vd_{ij} U_{ji} = sum_{ij} Vd_{ij} U^T_{ij}
# = sum_{row,col} V†[row,col] U[col,row]
# = sum (up0,up1,dn0,dn1) T0[up0,dn0,d] T1[d,up1,dn1] U[(dn0,dn1),(up0,up1)]
# In tensor form: U^T[up0,up1,dn0,dn1] = G[dn0,dn1,up0,up1]
# So: Tr(V†U) = T0[up0,dn0,d] T1[d,up1,dn1] G[dn0,dn1,up0,up1]

tr_direct = np.einsum('abd,dce,bcae->', T0_raw, T1_raw, G)
print(f"Direct (no dummy): {tr_direct}")
print(f"Match: {np.allclose(tr_direct, exact, atol=1e-10)}")

# With dummy bonds:
tr_dummy = np.einsum('iabd,dcek,bcae->', T0, T1, G)
print(f"Direct (with dummy): {tr_dummy}")

# OK so the contraction is:
# T0[i,a,b,d] T1[d,c,e,k] G[b,e,a,c] -> scalar (sum over everything)
# For the MERGE (leaving physical open as a 6-index tensor):
# merged[i, open_up0, open_dn0, open_up1, open_dn1, k]
# where open indices are the ones NOT contracted between T and G.
# T has indices a=up0, b=dn0, c=up1, e=dn1
# G has indices b_G=k0, e_G=k1, a_G=b0, c_G=b1 (from G[dn0,dn1,up0,up1])
# Wait, G's indices in the contraction are G[b,e,a,c] so:
# G first index = b = dn0_T = lower0_V†
# G second index = e = dn1_T = lower1_V†
# G third index = a = up0_T = upper0_V†
# G fourth index = c = up1_T = upper1_V†
# So G is indexed as G[lower0_V†, lower1_V†, upper0_V†, upper1_V†]
# But G[k0,k1,b0,b1] = U[(k0 k1),(b0 b1)]
# So: lower0_V† = k0_U, lower1_V† = k1_U, upper0_V† = b0_U, upper1_V† = b1_U
# Meaning: the lower indices of V† contract with the ket indices of U,
# and the upper indices of V† contract with the bra indices of U.
# This makes sense: Tr(V†U) = sum V†[row,col] U[col,row]
# row_V† = upper_V† = bra_U, col_V† = lower_V† = ket_U.

# For the MERGE, we want to contract lower_V† with ket_U, leaving
# upper_V† and bra_U open:
# merged[bl, upper0_V†, bra0_U, upper1_V†, bra1_U, br]
# Contract: lower0_V† = ket0_U, lower1_V† = ket1_U
# T0[i, a=up0, b=dn0, d=bond], G[b=ket0, e_=ket1, g=bra0, h=bra1]
# But we need T1 to get dn1. So:
# T0[i, a, b, d] * T1[d, c, e, k] * G[b, e, g, h]
# -> merged[i, a, g, c, h, k] = (bl, up0_V†, bra0_U, up1_V†, bra1_U, br)
# Trace: up0=bra0, up1=bra1 -> gives sum Vd * U^T = Tr(V†U) ✓

merged_correct = np.einsum('iabd,dcek,begh->iaghck', T0, T1, G)
print(f"\nCorrect merged shape: {merged_correct.shape}")
tr_merged = 0
for a in range(2):
    for c in range(2):
        tr_merged += merged_correct[0, a, a, c, c, 0]
print(f"Trace of merged: {tr_merged}")
print(f"Match: {np.allclose(tr_merged, exact, atol=1e-10)}")

# Now verify split+trace_mpo works:
from tno_compiler.mpo_ops import split_merged_tensor, trace_mpo
T0n, T1n = split_merged_tensor(merged_correct, max_bond=128)
ov = trace_mpo([T0n, T1n])
print(f"Split+trace: {ov}")
print(f"Match: {np.allclose(ov, exact, atol=1e-8)}")

# So the correct einsum for gate_is_left is:
# 'iabd,dcek,begh->iaghck'
print("\n=== CORRECT EINSUM for gate_is_left: 'iabd,dcek,begh->iaghck' ===")
