"""Check bond dimensions in the 4-qubit merge."""

import numpy as np
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo as mktgt
from tno_compiler.mpo_ops import quimb_mpo_to_arrays, split_merged_tensor, trace_mpo

tg = random_haar_gates(4, 1, seed=0)
cg = random_haar_gates(4, 1, seed=100)

T = quimb_mpo_to_arrays(mktgt(tg, 4, 1))
print("Original MPO:")
for i, t in enumerate(T):
    print(f"  T[{i}]: {t.shape}")

# Merge each pair
m0 = np.einsum('iabd,dcek,begh->iaghck', T[0], T[1], cg[0])
m1 = np.einsum('iabd,dcek,begh->iaghck', T[2], T[3], cg[1])
print(f"\nMerged tensors: m0={m0.shape}, m1={m1.shape}")

T0, T1 = split_merged_tensor(m0, max_bond=128)
T2, T3 = split_merged_tensor(m1, max_bond=128)
print(f"Split: T0={T0.shape}, T1={T1.shape}, T2={T2.shape}, T3={T3.shape}")
print(f"Bond T1-right={T1.shape[-1]}, T2-left={T2.shape[0]}")

# The issue: T1 has right bond from the SVD of the merged (0,1) block,
# and T2 has left bond from the SVD of the merged (2,3) block.
# These are INDEPENDENT splits -- they don't share a common bond.
# The original bond between sites 1 and 2 was 1, but after merging,
# each side creates its own internal bond.

# The merged tensor m0 has shape (1, up0, bra0_U, up1, bra1_U, 1)
# The last dim (1) is the bond connecting to site 2.
# After splitting: T0=(1, up0, bra0_U, d), T1=(d, up1, bra1_U, 1)
# T1's right bond is still 1 -- good.
# And m1 has shape (1, up2, bra2_U, up3, bra3_U, 1)
# T2=(1, up2, bra2_U, d'), T3=(d', up3, bra3_U, 1)
# T2's left bond is 1 -- matches T1's right bond.

# So bonds DO connect. Let me check the trace more carefully.
# trace_mpo traces physical: einsum('iaaj->ij', T)
# For T0=(1, up0, bra0_U, d): trace means up0=bra0_U
# But up0 = upper_V† = bra_V and bra0_U = bra_U
# For Tr(V†U): we need bra_V = bra_U at each site. ✓

# So the trace SHOULD work. Let me check the numeric values.
print("\n--- Numeric check ---")
# T0 after trace: (1, d) matrix
t0 = np.einsum('iaaj->ij', T0)
t1 = np.einsum('iaaj->ij', T1)
t2 = np.einsum('iaaj->ij', T2)
t3 = np.einsum('iaaj->ij', T3)
print(f"Traced shapes: {t0.shape}, {t1.shape}, {t2.shape}, {t3.shape}")
result = t0 @ t1 @ t2 @ t3
print(f"Product: {result}")
print(f"Trace: {np.trace(result)}")

# Compare with manual computation
V = gates_to_unitary(tg, 4, 1)
U = gates_to_unitary(cg, 4, 1)
exact = np.trace(V.conj().T @ U)
print(f"Exact: {exact}")

# What if we DON'T split and instead trace the merged tensors directly?
# m0: (1, up0, bra0, up1, bra1, 1) -> trace: up0=bra0, up1=bra1
t_m0 = np.einsum('iabcdi->', m0[..., :1])  # Wrong indexing
# Actually m0[bl, up0, bra0, up1, bra1, br]
# Trace both physical pairs: up0=bra0, up1=bra1
t_m0 = 0
for a in range(2):
    for b in range(2):
        t_m0 += m0[0, a, a, b, b, 0]
t_m1 = 0
for a in range(2):
    for b in range(2):
        t_m1 += m1[0, a, a, b, b, 0]
print(f"\nDirect merged traces: m0={t_m0}, m1={t_m1}")
print(f"Product: {t_m0 * t_m1}")

# Factor the exact
V0, V1 = tg[0].reshape(4, 4), tg[1].reshape(4, 4)
U0, U1 = cg[0].reshape(4, 4), cg[1].reshape(4, 4)
print(f"Tr(V0†U0) = {np.trace(V0.conj().T @ U0)}")
print(f"Tr(V1†U1) = {np.trace(V1.conj().T @ U1)}")
