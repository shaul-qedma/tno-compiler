"""Determine the exact index mapping between quimb MPO and gate tensors."""

import numpy as np
import quimb.tensor as qtn
from qiskit.quantum_info import random_unitary

# 2 qubits: V and U are 4x4 matrices
V = random_unitary(4, seed=0).data
U = random_unitary(4, seed=100).data
Vd = V.conj().T

exact = np.trace(Vd @ U)
print(f"Tr(V†U) = {exact}")

# V† as MPO
mpo = qtn.MatrixProductOperator.from_dense(Vd, dims=[2, 2])
T0 = np.array(mpo[0].data)  # (up0, dn0, bond)
T1 = np.array(mpo[1].data)  # (bond, up1, dn1)

# V† reconstruction: Vd[(up0,up1),(dn0,dn1)] = sum_d T0[up0,dn0,d] T1[d,up1,dn1]
Vd_check = np.einsum('abd,dce->acbe', T0, T1).reshape(4, 4)
print(f"V† correct: {np.allclose(Vd_check, Vd, atol=1e-10)}")

# U as gate tensor: U[(k0,k1),(b0,b1)] reshaped to G[k0,k1,b0,b1]
G = U.reshape(2, 2, 2, 2)

# Tr(V†U) = sum_{r,c} Vd[r,c] U[c,r]
# r = (up0, up1), c = (dn0, dn1)
# = sum T0[up0,dn0,d] T1[d,up1,dn1] U[(dn0,dn1),(up0,up1)]
# U[(dn0,dn1),(up0,up1)] = G indexed with rows=(dn0,dn1) and cols=(up0,up1)
# G[k0,k1,b0,b1] with k=row, b=col
# So G[dn0,dn1,up0,up1] maps to G[k0=dn0, k1=dn1, b0=up0, b1=up1]

# Try ALL possible index permutations of G to find which one gives exact
from itertools import permutations
for perm in permutations(range(4)):
    G_perm = G.transpose(perm)
    val = np.einsum('abd,dce,bcae->', T0, T1, G_perm)
    if np.allclose(val, exact, atol=1e-10):
        # Map perm back to meaning: G_perm[b_idx, e_idx, a_idx, c_idx]
        # where a=up0, b=dn0, c=up1, e=dn1
        # G_perm has indices from G rearranged as perm
        # G_perm[i,j,k,l] = G[perm^-1[i], perm^-1[j], ...]
        # Actually G_perm[i,j,k,l] = G[perm[i], perm[j], perm[k], perm[l]] -- no, transpose works as:
        # G.transpose(perm)[i0,i1,i2,i3] = G[i_{perm[0]}, i_{perm[1]}, i_{perm[2]}, i_{perm[3]}] -- no
        # Actually: G.transpose((p0,p1,p2,p3))[i,j,k,l] = G[..] where axis p0 becomes axis 0 etc.
        # So G.transpose(perm)[i,j,k,l] means the original axis perm[0] is now axis 0, etc.
        # = G[i_old] where i_old[perm[m]] = i_new[m]
        # So G_perm[b, e, a, c] = G[x,y,z,w] where x=original axis at position perm.index(where b goes)
        # This is getting confusing. Let me just print which works.
        print(f"  MATCH: G.transpose{perm} -> einsum gives {val}")

# Also try: what does the working debug_first_principles einsum produce?
# That script used: 'iabd,dcek,begh->iaghck' on T0_with_dummy, T1_with_dummy, G
# Where G was U.reshape(2,2,2,2) = G[k0,k1,b0,b1] (NOT transposed)
T0d = T0[np.newaxis, ...]
T1d = T1[..., np.newaxis]
merged = np.einsum('iabd,dcek,begh->iaghck', T0d, T1d, G)
tr = 0
for a in range(2):
    for c in range(2):
        tr += merged[0, a, a, c, c, 0]
print(f"\ndebug_first_principles einsum trace: {tr}")
print(f"Match: {np.allclose(tr, exact, atol=1e-10)}")

# So this einsum contracts: b=dn0 with G's first index (b=k0_G)
# and e=dn1 with G's second index (e=k1_G)
# meaning lower_V† = ket_U (b0=k0, b1=k1)
# Result open: a=up0_V†, g=b0_U, c=up1_V†, h=b1_U
# Trace: up0_V† = b0_U and up1_V† = b1_U
# This means: upper_V† contracts with bra_U for the trace.
# upper_V† = bra_V, bra_U = col of U
# Tr(V†U) = sum V†[row,col] U[col,row]
# row_V† = upper = bra_V, col_V† = lower = ket_V
# So: lower_V† = ket_V contracts with ket_U (from the merge)
# And trace requires: upper_V† = bra_V = bra_U
# That gives: sum V*[bra_V, ket_V] U[ket_V, bra_V]
# = sum conj(V)[bra,ket] U[ket,bra]
# = sum conj(V[bra,ket]) U[ket,bra]
# Hmm wait: V†[row,col] = V*[col,row] = conj(V[col,row])
# Tr(V†U) = sum_{r,c} V†[r,c] U[c,r] = sum conj(V[c,r]) U[c,r]
# = <V, U>_F (Frobenius inner product)
# = sum V*[i,j] U[i,j]

# So: the merge contracts lower_V† (=ket_V) with ket_U,
# and the trace contracts upper_V† (=bra_V) with bra_U.
# V†'s lower = V's ket = V*'s ket. U's ket = U's ket.
# Contracting V's ket with U's ket and V's bra with U's bra gives
# sum V*[bra,ket] U[bra,ket]... no wait.
# V†[upper, lower] = V*[lower, upper]
# upper = index 0 in MPO (k), lower = index 1 (b)
# V†[k, b] = V*[b, k]
# Merging contracts b (lower of V†) with ket of G (first two indices)
# Tracing contracts k (upper of V†) with bra of G (last two indices)
# sum V†[k,b] G[b,k] = sum V*[b,k] G[b,k] = Tr(V^T G) ?
# No: sum_{k,b} V†[k,b] G[b,k] = Tr(V† G.T) ??
# Actually it's just: sum_{r,c} V†[r,c] U[c,r] = Tr(V† U). Which we verified works.

# The KEY insight: the gate G[k0,k1,b0,b1] = U[(k0 k1),(b0 b1)] and
# in the merge, G's k-indices (first two) contract with V†'s lower (b) indices.
# This is correct because V†'s lower = col of V† and U's row = ket = k.
# Tr(V†U) = sum V†[r,c] U[c,r] : c contracts with ket of U (G's k)
# and c = col_V† = lower_V† = b indices in MPO.
# ✓

# So the einsum 'iabd,dcek,begh->iaghck' IS correct for gate_is_left.
# Let me verify it ALSO works for the 4-qubit factored case:
print("\n=== 4-qubit test ===")
from tno_compiler.brickwall import random_haar_gates as rhg, gates_to_unitary as gtu, target_mpo as mktgt
from tno_compiler.mpo_ops import quimb_mpo_to_arrays

tg4 = rhg(4, 1, seed=0)
cg4 = rhg(4, 1, seed=100)

T4 = quimb_mpo_to_arrays(mktgt(tg4, 4, 1))
G4_0, G4_1 = cg4[0], cg4[1]

# Merge gate0 with T[0],T[1]
m0 = np.einsum('iabd,dcek,begh->iaghck', T4[0], T4[1], G4_0)
# Split
from tno_compiler.mpo_ops import split_merged_tensor
T0n, T1n = split_merged_tensor(m0, max_bond=128)

# Merge gate1 with T[2],T[3]
m1 = np.einsum('iabd,dcek,begh->iaghck', T4[2], T4[3], G4_1)
T2n, T3n = split_merged_tensor(m1, max_bond=128)

# Trace
from tno_compiler.mpo_ops import trace_mpo
ov = trace_mpo([T0n, T1n, T2n, T3n])
exact4 = np.trace(gtu(tg4, 4, 1).conj().T @ gtu(cg4, 4, 1))
print(f"4-qubit trace: {ov}")
print(f"Exact: {exact4}")
print(f"Match: {np.allclose(ov, exact4, atol=1e-6)}")
